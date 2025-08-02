import os
import sys
import time
import requests
import asyncio
import math
import random
from datetime import datetime, date, timedelta
from random import shuffle, randint, choice
from typing import List, Dict, Optional
import pandas as pd

# === Channel Gatekeeper Settings ===
# Extend whitelist to include _ and <> for actions/guesses
ALPHABET = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/_<>")[:50])
MAX_SYMBOLS = 12  # max code-points per message

def gate_message(msg: str) -> Optional[str]:
    filtered = "".join(ch for ch in msg if ch in ALPHABET)
    if len(filtered) == 0 or len(filtered) > MAX_SYMBOLS:
        return None
    return filtered

# === LLM API Configuration ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
    sys.exit(1)

# === Game & Conversation Settings ===
MAX_HISTORY = 40
DELAY_BETWEEN_MESSAGES = 0.5
MAX_TURNS_PER_AGENT = 25

GAME_INSTRUCTION = """
== M&A Due Diligence Challenge ==
You are part of a 4-person M&A analysis team evaluating an acquisition target.

Situation:
• Each analyst has access to only 5 documents from a 20-document data room
• Your team must work together to evaluate the acquisition opportunity
• Communication channel has severe bandwidth restrictions (12 symbols maximum)
• No analyst can see another's documents directly

Constraints:
- Messages limited to symbols: A-Z, 0-9, +-*/_<> (max 12 symbols)
- Time pressure: Analysis must be completed efficiently
- Communication bandwidth is extremely limited

Your goal: Work with your team to make the best acquisition decision possible.

Wait for <START_GAME> to begin.
"""

# === Shared MDP State ===
logs = {k: [] for k in [
    "efficiency","predictability","grounding",
    "compositionality","consistency","generalization",
    "pos_signaling","pos_listening","symmetry","task_success",
    "action_count","reveal_count","risk_flags","votes_cast"
]}
risk_flags: List[str] = []
votes: Dict[str, str] = {}
vote_allowed = False
termination_event = asyncio.Event()
state_lock = asyncio.Lock()

# === Metrics Calculation ===
def calculate_grounding_purity(msg: str, agent_docs: List[Dict]) -> float:
    """Calculate how well message tokens relate to document attributes"""
    if not msg or not agent_docs:
        return 0.0
    
    # Extract document attributes from agent's documents
    doc_attrs = set()
    for doc in agent_docs:
        # Add document ID
        doc_attrs.add(doc['docID'])
        # Add document type (first 3 chars)
        doc_attrs.add(doc['type'][:3].upper())
        # Add jurisdiction
        doc_attrs.add(doc['jurisdiction'])
        # Add counterparty (first 3 chars)
        if doc['counterparty'] != "—":
            doc_attrs.add(doc['counterparty'][:3].upper())
    
    # Split message into tokens
    msg_tokens = set(msg.split())
    
    # Check for exact matches and partial matches
    matches = 0
    for token in msg_tokens:
        # Exact match
        if token in doc_attrs:
            matches += 1
        # Partial match (token contains document attribute)
        else:
            for attr in doc_attrs:
                if attr in token or token in attr:
                    matches += 0.5
                    break
    
    return matches / len(msg_tokens) if msg_tokens else 0.0

def calculate_message_entropy(msg: str) -> float:
    """Calculate entropy of message for predictability metric"""
    if not msg:
        return 0.0
    
    # Count character frequencies
    char_counts = {}
    for char in msg:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculate entropy
    total_chars = len(msg)
    entropy = 0.0
    for count in char_counts.values():
        p = count / total_chars
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy

def calculate_cic(msg: str, history: List[Dict]) -> float:
    """Calculate Contextual Information Contribution"""
    if not history:
        return 1.0  # First message has full contribution
    
    # Simple CIC: how much new information this message adds
    prev_tokens = set()
    for entry in history[-5:]:  # Last 5 messages
        prev_tokens.update(entry["msg"].split())
    
    current_tokens = set(msg.split())
    new_tokens = current_tokens - prev_tokens
    
    return len(new_tokens) / len(current_tokens) if current_tokens else 0

def log_metrics(msg: str, agent_docs: List[Dict], history: List[Dict]):
    """Log comprehensive metrics for judging criteria"""
    # Efficiency
    logs["efficiency"].append(len(msg))
    
    # Predictability (entropy)
    logs["predictability"].append(calculate_message_entropy(msg))
    
    # Grounding (purity)
    logs["grounding"].append(calculate_grounding_purity(msg, agent_docs))
    
    # CIC (Contextual Information Contribution)
    logs["pos_listening"].append(calculate_cic(msg, history))
    
    # Action count
    action_count = 1 if any(msg.startswith(x) for x in ["REVEAL", "FLAG", "VOTE", "<GUESS"]) else 0
    logs["action_count"].append(action_count)

# === Helpers ===
def clean_response(text: str) -> str:
    return " ".join(text.strip().replace("\n", " ").split())

def format_doc_summary(doc: Dict) -> str:
    # 3-symbol summary per field: ID, type, jurisdiction, counterparty
    return f"{doc['docID']}:{doc['type'][:3]}:{doc['jurisdiction']}:{doc['counterparty'][:3]}"

# === Data-Room generation ===
def generate_full_docs() -> List[Dict]:
    doc_types = ["financial_statement","tax_return","customer_contract","employment_agreement",
                 "patent_file","litigation_memo","supplier_po","ip_license","privacy_policy","lease_agreement"]
    jurisdictions = ["US-DE","US-CA","UK","DE","FR","JP","IN","SG"]
    ip_statuses = ["granted","pending","expired","n/a"]
    docs, base = [], date.fromisoformat("2024-01-01")
    for i in range(20):
        docs.append({
            "docID": f"D{i:03}",
            "type": choice(doc_types),
            "date": str(base + timedelta(days=randint(0,120))),
            "liability_score": round(randint(2,45)/100,2),
            "revenue_impact": randint(-30,250),
            "jurisdiction": choice(jurisdictions),
            "ip_status": choice(ip_statuses),
            "ebitda_margin": round(randint(5,45)/100,2),
            "counterparty": choice(["BigBoxCo","AlphaGen","NovaChem","—"]),
        })
    return docs

def split_data_room_randomly(full_docs: List[Dict], n_agents: int=4) -> List[List[Dict]]:
    shuffle(full_docs)
    sz = len(full_docs) // n_agents
    return [full_docs[i*sz:(i+1)*sz] for i in range(n_agents-1)] + [full_docs[(n_agents-1)*sz:]]

# === Agent Configurations ===
AGENTS = [
    {
        "name": "Agent-A",
        "model": "anthropic/claude-3.5-sonnet",
        "system_prompt": (
            "You are an M&A analyst with access to 5 documents from a 20-document data room. "
            "Your team must evaluate the acquisition target and reach a decision. "
            "Communication is restricted to 12 symbols from A-Z,0-9,+-*/_<>. "
            "You cannot see other analysts' documents directly."
        ),
        "temperature": 0.7,
        "max_tokens": 32,
        "reveals_left": 3
    },
    {
        "name": "Agent-B",
        "model": "google/gemini-flash-1.5",
        "system_prompt": (
            "You are an M&A analyst with access to 5 documents from a 20-document data room. "
            "Your team must evaluate the acquisition target and reach a decision. "
            "Communication is restricted to 12 symbols from A-Z,0-9,+-*/_<>. "
            "You cannot see other analysts' documents directly."
        ),
        "temperature": 0.7,
        "max_tokens": 32,
        "reveals_left": 3
    },
    {
        "name": "Agent-C",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "system_prompt": (
            "You are an M&A analyst with access to 5 documents from a 20-document data room. "
            "Your team must evaluate the acquisition target and reach a decision. "
            "Communication is restricted to 12 symbols from A-Z,0-9,+-*/_<>. "
            "You cannot see other analysts' documents directly."
        ),
        "temperature": 0.7,
        "max_tokens": 32,
        "reveals_left": 3
    },
    {
        "name": "Agent-D",
        "model": "mistralai/mistral-7b-instruct",
        "system_prompt": (
            "You are an M&A analyst with access to 5 documents from a 20-document data room. "
            "Your team must evaluate the acquisition target and reach a decision. "
            "Communication is restricted to 12 symbols from A-Z,0-9,+-*/_<>. "
            "You cannot see other analysts' documents directly."
        ),
        "temperature": 0.7,
        "max_tokens": 32,
        "reveals_left": 3
    }
]

# Initialize docs and slices
full_docs = generate_full_docs()
slices = split_data_room_randomly(full_docs, len(AGENTS))
for ag, sl in zip(AGENTS, slices):
    ag["slice"] = sl

print("Private slices assigned:")
for ag in AGENTS:
    print(f" - {ag['name']} has {len(ag['slice'])} docs")

# === LLM Integration ===
async def generate_response_async(prompt: str, agent: Dict, history: List[Dict], turn: int) -> str:
    messages = [{"role": "system", "content": agent["system_prompt"]}]
    if turn == 0:
        messages.append({"role": "system", "content": GAME_INSTRUCTION})
    if history:
        # include only last MAX_HISTORY turns
        messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": agent["model"],
        "messages": messages,
        "temperature": agent["temperature"],
        "max_tokens": agent["max_tokens"]
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    
    try:
        resp = await asyncio.to_thread(requests.post, OPENROUTER_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        j = resp.json()
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        return clean_response(content)
    except Exception as e:
        print(f"API Error for {agent['name']}: {e}")
        return ""

# === Synchronous Fallback for Jupyter/IPython ===
def generate_response_sync(prompt: str, agent: Dict, history: List[Dict], turn: int) -> str:
    """Synchronous version of generate_response for fallback"""
    messages = [{"role": "system", "content": agent["system_prompt"]}]
    if turn == 0:
        messages.append({"role": "system", "content": GAME_INSTRUCTION})
    if history:
        messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": agent["model"],
        "messages": messages,
        "temperature": agent["temperature"],
        "max_tokens": agent["max_tokens"]
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        j = resp.json()
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        return clean_response(content)
    except Exception as e:
        print(f"API Error for {agent['name']}: {e}")
        return ""

# === Agent Loop ===
async def agent_task(agent: Dict, history: List[Dict]):
    turns = 0
    while not termination_event.is_set() and turns < MAX_TURNS_PER_AGENT:
        # Build enriched prompt with mini-summaries
        private_content = " ".join(format_doc_summary(d) for d in agent["slice"])
        last = history[-1]["msg"] if history else "<no prior>"
        prompt = f"Docs: {private_content}. Prev: {last}"

        raw = await generate_response_async(prompt, agent, history, turns)
        filtered = gate_message(raw)
        if not filtered:
            turns += 1
            await asyncio.sleep(DELAY_BETWEEN_MESSAGES + random.uniform(0.1, 0.3))
            continue

        # Action parsing and state updates under lock
        async with state_lock:
            global vote_allowed
            if filtered.startswith("REVEAL_") and agent["reveals_left"] > 0:
                doc_id = filtered.split("_")[1]
                docs = [d for d in agent["slice"] if d["docID"] == doc_id]
                if docs:
                    print(f"{agent['name']} reveals {doc_id}: {docs[0]}")
                    agent["reveals_left"] -= 1
                    logs["reveal_count"].append(len([ag for ag in AGENTS if ag["reveals_left"] < 3]))
            elif filtered.startswith("FLAG_"):
                risk = filtered.split("_")[1]
                if len(risk_flags) < 5:
                    risk_flags.append(risk)
                    print(f"{agent['name']} flags risk: {risk} (total={len(risk_flags)})")
                    logs["risk_flags"].append(len(risk_flags))
                    if len(risk_flags) >= 5:
                        vote_allowed = True
                        print("Voting enabled! Agents can now VOTE_GO or VOTE_NO")
            elif filtered.startswith("VOTE_") and vote_allowed:
                choice_vote = filtered.split("_")[1]
                votes[agent['name']] = choice_vote
                print(f"{agent['name']} votes: {choice_vote}")
                logs["votes_cast"].append(len(votes))
                if len(votes) == len(AGENTS) and len(set(votes.values())) == 1:
                    termination_event.set()
                    print(f"Unanimous vote achieved: {votes}")
                    return
            elif filtered.startswith("<GUESS:"):
                # Extract guess content
                guess_start = filtered.find("<GUESS:") + 7
                guess_end = filtered.find(">", guess_start)
                if guess_end != -1:
                    guess_content = filtered[guess_start:guess_end]
                    print(f"{agent['name']} makes guess: {guess_content}")
                    # Could add guess evaluation logic here
            else:
                # regular token message
                history.append({"agent": agent["name"], "msg": filtered})
                print(f"{agent['name']}: {filtered}")
            
            # Log comprehensive metrics
            log_metrics(filtered, agent["slice"], history)

        turns += 1
        await asyncio.sleep(DELAY_BETWEEN_MESSAGES + random.uniform(0.1, 0.3))

# === Main Entry ===
async def main():
    history: List[Dict] = []
    print("\n<START_GAME>\n")
    tasks = [asyncio.create_task(agent_task(ag, history)) for ag in AGENTS]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in tasks:
        t.cancel()
    
    # Score outcome
    success = 1 if vote_allowed and len(votes) == len(AGENTS) and len(set(votes.values())) == 1 else 0
    logs["task_success"].append(success)
    print("\n=== EPISODE COMPLETE ===\n")
    print(f"Risk flags: {risk_flags}")
    print(f"Votes: {votes}")
    print(f"Success: {success}")
    
    # Print comprehensive metrics summary
    df = pd.DataFrame({k: pd.Series(v) for k, v in logs.items() if v})
    print("=== Metrics Summary ===")
    print(df.describe().T)
    
    # Print game state summary
    print(f"\n=== Game State Summary ===")
    print(f"Documents Revealed: {sum(3 - ag['reveals_left'] for ag in AGENTS)}")
    print(f"Risk Flags: {len(risk_flags)}")
    print(f"Votes Cast: {len(votes)}")
    print(f"Success Score: {success}")

def main_sync():
    """Synchronous version of main for environments that don't support asyncio well"""
    history: List[Dict] = []
    print("\n<START_GAME>\n")
    turn = 0
    
    # Synchronous game loop
    while turn < MAX_TURNS_PER_AGENT and not termination_event.is_set():
        print(f"\n--- Turn {turn + 1} ---")
        
        # Process each agent sequentially
        for agent in AGENTS:
            if termination_event.is_set():
                break
                
            # Build enriched prompt with mini-summaries
            private_content = " ".join(format_doc_summary(d) for d in agent["slice"])
            last = history[-1]["msg"] if history else "<no prior>"
            prompt = f"Docs: {private_content}. Prev: {last}"

            # Generate response synchronously
            raw = generate_response_sync(prompt, agent, history, turn)
            filtered = gate_message(raw)
            
            if filtered:
                # Process actions (simplified for sync version)
                global vote_allowed
                if filtered.startswith("REVEAL_") and agent["reveals_left"] > 0:
                    doc_id = filtered.split("_")[1]
                    docs = [d for d in agent["slice"] if d["docID"] == doc_id]
                    if docs:
                        print(f"{agent['name']} reveals {doc_id}: {docs[0]}")
                        agent["reveals_left"] -= 1
                        logs["reveal_count"].append(len([ag for ag in AGENTS if ag["reveals_left"] < 3]))
                elif filtered.startswith("FLAG_"):
                    risk = filtered.split("_")[1]
                    if len(risk_flags) < 5:
                        risk_flags.append(risk)
                        print(f"{agent['name']} flags risk: {risk} (total={len(risk_flags)})")
                        logs["risk_flags"].append(len(risk_flags))
                        if len(risk_flags) >= 5:
                            vote_allowed = True
                            print("Voting enabled! Agents can now VOTE_GO or VOTE_NO")
                elif filtered.startswith("VOTE_") and vote_allowed:
                    choice_vote = filtered.split("_")[1]
                    votes[agent['name']] = choice_vote
                    print(f"{agent['name']} votes: {choice_vote}")
                    logs["votes_cast"].append(len(votes))
                    if len(votes) == len(AGENTS) and len(set(votes.values())) == 1:
                        termination_event.set()
                        print(f"Unanimous vote achieved: {votes}")
                        break
                elif filtered.startswith("<GUESS:"):
                    # Extract guess content
                    guess_start = filtered.find("<GUESS:") + 7
                    guess_end = filtered.find(">", guess_start)
                    if guess_end != -1:
                        guess_content = filtered[guess_start:guess_end]
                        print(f"{agent['name']} makes guess: {guess_content}")
                else:
                    # regular token message
                    history.append({"agent": agent["name"], "msg": filtered})
                    print(f"{agent['name']}: {filtered}")
                
                # Log comprehensive metrics
                log_metrics(filtered, agent["slice"], history)
            
            # Add small delay to simulate async behavior
            time.sleep(DELAY_BETWEEN_MESSAGES + random.uniform(0.1, 0.3))
        
        turn += 1
        
        # Check for game completion
        if vote_allowed and len(votes) == len(AGENTS) and len(set(votes.values())) == 1:
            termination_event.set()
            break
    
    # Score outcome
    success = 1 if vote_allowed and len(votes) == len(AGENTS) and len(set(votes.values())) == 1 else 0
    logs["task_success"].append(success)
    print("\n=== EPISODE COMPLETE ===\n")
    print(f"Risk flags: {risk_flags}")
    print(f"Votes: {votes}")
    print(f"Success: {success}")
    
    # Print comprehensive metrics summary
    df = pd.DataFrame({k: pd.Series(v) for k, v in logs.items() if v})
    print("=== Metrics Summary ===")
    print(df.describe().T)
    
    # Print game state summary
    print(f"\n=== Game State Summary ===")
    print(f"Documents Revealed: {sum(3 - ag['reveals_left'] for ag in AGENTS)}")
    print(f"Risk Flags: {len(risk_flags)}")
    print(f"Votes Cast: {len(votes)}")
    print(f"Success Score: {success}")

if __name__ == "__main__":
    try:
        # Check if we're in a Jupyter/IPython environment
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            print("Running in existing event loop (Jupyter/IPython detected)")
            print("Using synchronous fallback...")
            main_sync()
        except RuntimeError:
            # No running loop, safe to use asyncio
            print("Using asynchronous execution...")
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0) 
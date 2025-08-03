"""
M&A Document Demo Generator
==========================

This module generates and displays sample M&A documents to demonstrate
what the AI agents analyze during the emergent language competition.

The agents receive compressed summaries, but this shows the full document
content they're reasoning about when making acquisition decisions.
"""

import json
from datetime import datetime, date, timedelta
from random import choice, randint
from typing import List, Dict


def generate_sample_documents() -> List[Dict]:
    """Generate realistic M&A due diligence documents with full content"""
    
    doc_types = [
        "financial_statement", "tax_return", "customer_contract", 
        "employment_agreement", "patent_file", "litigation_memo",
        "supplier_po", "ip_license", "privacy_policy", "lease_agreement"
    ]
    
    jurisdictions = ["US-DE", "US-CA", "UK", "DE", "FR", "JP", "IN", "SG"]
    ip_statuses = ["granted", "pending", "expired", "n/a"]
    counterparties = ["BigBoxCo", "AlphaGen", "NovaChem", "TechFlow", "DataVault", "—"]
    
    base_date = date.fromisoformat("2024-01-01")
    documents = []
    
    for i in range(10):  # Generate 10 sample documents
        # Create realistic financial metrics
        liability_score = round(randint(5, 45) / 100, 2)
        revenue_impact = randint(-50, 150)  # Can be negative
        ebitda_margin = round(randint(5, 45) / 100, 2)
        
        doc = {
            "docID": f"D{i:03}",
            "type": choice(doc_types),
            "date": str(base_date + timedelta(days=randint(0, 120))),
            "liability_score": liability_score,
            "revenue_impact": revenue_impact,
            "jurisdiction": choice(jurisdictions),
            "ip_status": choice(ip_statuses),
            "ebitda_margin": ebitda_margin,
            "counterparty": choice(counterparties),
            "full_content": generate_document_content(choice(doc_types), liability_score, revenue_impact)
        }
        documents.append(doc)
    
    return documents


def generate_document_content(doc_type: str, liability_score: float, revenue_impact: int) -> str:
    """Generate realistic document content based on type and risk factors"""
    
    content_templates = {
        "financial_statement": f"""
CONFIDENTIAL FINANCIAL STATEMENT
Target Company Inc. - Q4 2023

INCOME STATEMENT
Revenue: ${abs(revenue_impact) * 3:.1f}M {'(↓15% YoY)' if revenue_impact < 0 else '(↑8% YoY)'}
Cost of Goods Sold: ${abs(revenue_impact) * 1.8:.1f}M
Gross Profit: ${abs(revenue_impact) * 1.2:.1f}M
Operating Expenses: ${abs(revenue_impact) * 0.8:.1f}M
EBITDA: ${revenue_impact:.1f}M
Net Income: ${revenue_impact * 0.7:.1f}M

BALANCE SHEET HIGHLIGHTS
Total Assets: ${abs(revenue_impact) * 5:.1f}M
Total Liabilities: ${abs(revenue_impact) * 3:.1f}M
Shareholder Equity: ${abs(revenue_impact) * 2:.1f}M
Debt-to-Equity Ratio: {liability_score * 4:.1f}x

CASH FLOW SUMMARY
Operating Cash Flow: ${revenue_impact * 0.8:.1f}M
Investing Cash Flow: -${abs(revenue_impact) * 0.3:.1f}M
Financing Cash Flow: ${revenue_impact * 0.1:.1f}M

{'⚠️ WARNING: Declining revenue trends and high leverage ratios.' if liability_score > 0.25 else '✅ Strong financial position with positive trends.'}
        """,
        
        "litigation_memo": f"""
ATTORNEY-CLIENT PRIVILEGED
LITIGATION RISK ASSESSMENT

RE: Pending Legal Matters - Target Company Inc.

EXECUTIVE SUMMARY
Current litigation exposure: ${liability_score * 100:.0f}M
Active cases: {int(liability_score * 20)}
Settlement negotiations: {int(liability_score * 10)} ongoing

MAJOR CASES:
1. Patent Infringement (TechCorp v. Target)
   - Damages sought: ${liability_score * 50:.0f}M
   - Status: Discovery phase
   - Settlement range: ${liability_score * 20:.0f}M - ${liability_score * 30:.0f}M

2. Employment Class Action
   - Plaintiffs: {int(liability_score * 500)} employees
   - Claims: Wage & hour violations
   - Exposure: ${liability_score * 25:.0f}M

3. Customer Contract Dispute (BigBoxCo)
   - Contract value: ${abs(revenue_impact) * 2:.0f}M
   - Penalty exposure: ${liability_score * 15:.0f}M
   - Likelihood of loss: {'High' if liability_score > 0.3 else 'Medium'}

RECOMMENDATION:
{'PROCEED WITH EXTREME CAUTION - Significant litigation risks' if liability_score > 0.25 else 'Manageable litigation portfolio within industry norms'}
        """,
        
        "customer_contract": f"""
MASTER SERVICE AGREEMENT
Target Company Inc. & {choice(['BigBoxCo', 'AlphaGen', 'NovaChem'])}

CONTRACT TERMS:
Effective Date: 2023-01-01
Term: 3 years (auto-renewable)
Total Contract Value: ${abs(revenue_impact) * 4:.0f}M
Annual Minimum: ${abs(revenue_impact):.0f}M

SERVICE LEVELS:
- Uptime Guarantee: 99.5%
- Response Time: 4 hours
- Resolution Time: 24 hours

FINANCIAL TERMS:
Payment Terms: Net 30 days
Penalty for SLA Breach: ${liability_score * 5:.0f}M per incident
Liability Cap: ${liability_score * 50:.0f}M
Termination Fee: ${liability_score * 10:.0f}M

TERMINATION CLAUSES:
- Either party: 90-day notice
- For cause: Immediate
- Change of control: Customer option to terminate

{'⚠️ HIGH RISK: Unusual penalty clauses and low liability caps' if liability_score > 0.25 else '✅ Standard commercial terms'}
Revenue Impact: {'Negative due to penalty exposure' if revenue_impact < 0 else 'Positive recurring revenue stream'}
        """,
        
        "patent_file": f"""
INTELLECTUAL PROPERTY PORTFOLIO SUMMARY
Target Company Inc.

PATENT: US-{10000000 + randint(100000, 999999)}
Title: "Advanced Data Processing and Machine Learning Architecture"
Filing Date: 2021-03-15
Grant Date: 2023-08-20
Expiration: 2041-03-15

PATENT STATUS: {choice(['Granted', 'Pending', 'Under Review'])}
Maintenance Fees: Current
Annual Royalty Income: ${abs(revenue_impact) * 0.5:.1f}M

LICENSING AGREEMENTS:
- Active Licensees: {int(liability_score * 20)}
- Royalty Rate: 3.5% of net sales
- Geographic Coverage: Worldwide

ONGOING DISPUTES:
- Infringement Claims: {int(liability_score * 5)}
- Prior Art Challenges: {int(liability_score * 3)}
- Invalidation Proceedings: {'Active' if liability_score > 0.3 else 'None'}

VALUATION ESTIMATE: ${abs(revenue_impact) * 8:.0f}M
Risk Assessment: {'HIGH - Multiple challenges to validity' if liability_score > 0.25 else 'MEDIUM - Standard IP portfolio risks'}

{'⚠️ CAUTION: Patent under challenge by competitors' if liability_score > 0.25 else '✅ Strong IP position with active licensing revenue'}
        """,
        
        "employment_agreement": f"""
EXECUTIVE EMPLOYMENT AGREEMENT
Chief Technology Officer - Target Company Inc.

COMPENSATION PACKAGE:
Base Salary: $280,000
Annual Bonus: Up to ${int(liability_score * 500000):,} (performance-based)
Equity Grant: {int(liability_score * 50000):,} stock options
Vesting Schedule: 4 years, 25% annually

BENEFITS:
Health Insurance: Full coverage
401(k) Match: 6%
Vacation: 4 weeks annually
Professional Development: ${int(liability_score * 25000):,} annually

RESTRICTIVE COVENANTS:
Non-Compete: {int(liability_score * 5)} years post-termination
Non-Solicitation: 18 months
Confidentiality: Perpetual
Invention Assignment: All work-related IP

SEVERANCE PROVISIONS:
Without Cause: {int(liability_score * 24)} months salary + benefits
Change of Control: {int(liability_score * 36)} months + accelerated vesting
Total Severance Exposure: ${liability_score * 2000000:.0f}

{'⚠️ HIGH COST: Expensive severance and acceleration clauses' if liability_score > 0.25 else '✅ Competitive but reasonable executive package'}
M&A Impact: {'Significant retention risk and costs' if liability_score > 0.25 else 'Standard change of control provisions'}
        """,
        
        "tax_return": f"""
CORPORATE TAX RETURN SUMMARY
Target Company Inc. - Tax Year 2023

FEDERAL RETURN:
Gross Income: ${abs(revenue_impact) * 3:.0f}M
Deductions: ${abs(revenue_impact) * 2:.0f}M
Taxable Income: ${revenue_impact:.0f}M
Federal Tax Due: ${abs(revenue_impact) * 0.21:.0f}M
Effective Tax Rate: 21%

STATE RETURNS:
Primary Jurisdiction: {choice(['Delaware', 'California', 'New York'])}
State Tax Due: ${abs(revenue_impact) * 0.08:.0f}M
Multi-State Apportionment: Complex

TAX POSITIONS:
R&D Credits Claimed: ${abs(revenue_impact) * 0.15:.0f}M
Transfer Pricing Adjustments: ${liability_score * 5:.0f}M
Uncertain Tax Positions: ${liability_score * 10:.0f}M

AUDIT STATUS:
IRS Examination: {'Active for 2021-2022' if liability_score > 0.3 else 'No current audits'}
State Audits: {int(liability_score * 3)} jurisdictions
Proposed Adjustments: ${liability_score * 20:.0f}M

RISK ASSESSMENT:
Tax Compliance Risk: {'HIGH - Multiple audit exposures' if liability_score > 0.25 else 'MEDIUM - Standard tax positions'}
Potential Additional Liability: ${liability_score * 25:.0f}M
        """
    }
    
    return content_templates.get(doc_type, f"[{doc_type} document content - Risk Level: {liability_score:.2f}]")


def save_demo_documents():
    """Generate and save sample documents to JSON file"""
    documents = generate_sample_documents()
    
    with open('/Users/Soren/Desktop/Hackathon/AI-Summit-Hackathon/sample_documents.json', 'w') as f:
        json.dump(documents, f, indent=2)
    
    print(f"✅ Saved {len(documents)} sample documents to sample_documents.json")
    return documents


def display_document_summary():
    """Display a formatted summary of what agents see vs. what they analyze"""
    documents = generate_sample_documents()
    
    print("=" * 80)
    print("🏢 M&A DUE DILIGENCE DOCUMENT DEMO")
    print("=" * 80)
    print()
    print("This demonstrates the types of legal and financial documents that")
    print("AI agents analyze during the emergent language competition.")
    print()
    print("Key Challenge: Agents see only COMPRESSED summaries but must")
    print("collaborate to understand the full business picture!")
    print()
    
    # Show 3 example documents in detail
    for i, doc in enumerate(documents[:3]):
        print("=" * 60)
        print(f"📄 DOCUMENT {i+1}: {doc['type'].replace('_', ' ').upper()}")
        print("=" * 60)
        print(f"Document ID: {doc['docID']}")
        print(f"Date: {doc['date']}")
        print(f"Jurisdiction: {doc['jurisdiction']}")
        print(f"Counterparty: {doc['counterparty']}")
        print()
        print("🔍 RISK METRICS:")
        print(f"  Liability Score: {doc['liability_score']:.2f} {'🔴 HIGH RISK' if doc['liability_score'] > 0.25 else '🟡 MEDIUM' if doc['liability_score'] > 0.15 else '🟢 LOW RISK'}")
        print(f"  Revenue Impact: ${doc['revenue_impact']}M {'📉 NEGATIVE' if doc['revenue_impact'] < 0 else '📈 POSITIVE'}")
        print(f"  EBITDA Margin: {doc['ebitda_margin']:.1%}")
        print()
        print("📋 FULL DOCUMENT CONTENT:")
        print(doc['full_content'])
        print()
        print("🤖 WHAT AGENTS SEE (Compressed):")
        print(f"   {doc['docID']}:{doc['type'][:3]}:{doc['jurisdiction']}:{doc['counterparty'][:3]}")
        print()
        print("💬 COMMUNICATION CHALLENGE:")
        print("   Agents must share insights using only 12 symbols: A-Z,0-9,+-*/_<>")
        print("   They develop their own language to flag risks and share findings!")
        print()
    
    print("=" * 80)
    print("🎯 COMPETITION OBJECTIVE:")
    print("=" * 80)
    print("4 AI agents each see 5/20 documents and must collaborate to make")
    print("an acquisition decision while developing emergent communication")
    print("protocols under severe bandwidth constraints.")
    print()
    print("The agents naturally develop shorthand codes, risk flags, and")
    print("voting systems - demonstrating authentic language emergence!")
    print("=" * 80)


if __name__ == "__main__":
    # Generate and display demo
    display_document_summary()
    
    # Save documents for inspection
    save_demo_documents()
    
    print("\n🎉 Demo complete! Check sample_documents.json for full document details.")

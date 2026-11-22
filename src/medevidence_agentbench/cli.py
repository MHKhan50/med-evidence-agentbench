import argparse, json
from .core import AgentOrchestrator

def main():
    parser = argparse.ArgumentParser(prog="medevidence", description="Agentic medication-label evidence benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Run an offline architecture smoke demo")
    check = sub.add_parser("check", help="Check a claim against live DailyMed label evidence")
    check.add_argument("--drug", required=True)
    check.add_argument("--claim", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        result = {"project": "MedEvidence-AgentBench", "status": "ready",
                  "agent_trace": ["planner", "retrieval", "evidence_verifier", "citation_auditor"],
                  "note": "Use 'check' for live DailyMed retrieval. No synthetic medical evidence is presented."}
    else:
        try:
            result = AgentOrchestrator().run(args.drug, args.claim)
        except Exception as exc:
            result = {"error": str(exc), "research_only": True}
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

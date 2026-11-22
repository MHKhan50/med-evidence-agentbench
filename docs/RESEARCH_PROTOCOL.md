# Research protocol

## Hypotheses
H1: Multi-stage agent orchestration improves source attribution versus a single retrieval baseline.
H2: Explicit abstention reduces unsupported positive verdicts when retrieval is weak.
H3: Tool traces improve reproducibility and error analysis.

## Required v0.2 work
- Build a human-annotated dataset of drug-label claims and exact supporting spans.
- Include supported, contradicted, and not-mentioned labels with clear annotation rules.
- Add a biomedical sentence-embedding or NLI model as a separately evaluated neural baseline.
- Compare keyword, TF-IDF, embedding retrieval, single-agent RAG, and multi-agent orchestration.
- Report macro-F1, evidence precision/recall, citation accuracy, abstention coverage-risk, latency, and tool calls.
- Use two independent annotators and adjudicate disagreements.
- Do not claim clinical safety or deployment readiness.

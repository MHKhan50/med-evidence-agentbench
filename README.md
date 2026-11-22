# MedEvidence-AgentBench v0.1.0

A lightweight, reproducible **Agentic NLP** benchmark for checking whether medication-safety claims are supported by evidence in official Structured Product Label (SPL) documents.

## Research objective
Evaluate a tool-using agent pipeline that plans a search, retrieves DailyMed label passages, verifies a claim against the retrieved text, and audits whether the final answer is properly grounded.

**This is a research prototype, not a medical device.** It does not diagnose, prescribe, recommend treatment, or make patient-specific decisions. A result of `insufficient_evidence` means the retrieved material did not establish the claim; it does not mean the claim is false.

## Agent architecture
1. **Planner agent**: decomposes a claim into medication and evidence-search terms.
2. **Retrieval agent**: calls the public DailyMed REST API and fetches an SPL record.
3. **Evidence verifier**: ranks label sections/passages with TF-IDF similarity and assigns a conservative research label.
4. **Citation auditor**: checks that each evidence item retains its label URL and source identity.
5. **Orchestrator**: runs the agents in sequence and emits a machine-readable trace.

The benchmark intentionally exposes the intermediate trace so tool use and evidence provenance can be evaluated.

## Quick start (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
medevidence --help
medevidence demo
```

## Live retrieval
```bash
medevidence check --drug "ibuprofen" --claim "The label contains a warning about gastrointestinal bleeding."
```
DailyMed access requires an internet connection. The API is read-only. API failures are reported rather than replaced with invented evidence.

## Output labels
- `supported`: retrieved label text has a strong lexical match to the claim.
- `contradicted`: **not automatically emitted in v0.1.0**; absence of a matching passage is not proof of contradiction.
- `insufficient_evidence`: evidence is missing or too weak.

The initial version favours precision and abstention over unsupported conclusions. A future study should add human-annotated claim/evidence pairs, semantic entailment models, calibration, and blinded expert review.

## Reproducibility
- Python 3.9+
- CPU-only; no GPU required
- TF-IDF retrieval baseline (scikit-learn)
- JSON trace includes agent steps, retrieved passages, source URL, and scores
- `pytest -q` runs offline tests

## Data and ethics
No patient data is collected. Live evidence is retrieved from public drug-label records. Product labels are manufacturer-submitted labeling available through the U.S. National Library of Medicine's DailyMed service. Labels are not a substitute for professional clinical judgement. The prototype must not be used to make care decisions.

## Primary references
- DailyMed Web Services (NLM): https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
- What is DailyMed? (NLM): https://support.nlm.nih.gov/knowledgebase/article/KA-04892/en-us
- RxNav / NLM API documentation: https://lhncbc.nlm.nih.gov/RxNav/APIs/

## Suggested paper title
*Evidence-Grounded Agentic NLP for Medication-Label Claim Verification: A Reproducible Tool-Use Benchmark*

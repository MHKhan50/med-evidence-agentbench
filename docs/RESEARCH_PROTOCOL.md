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

## Frozen benchmark corpus: DailyMed metformin

The initial v0.2 retrieval corpus contains five versioned DailyMed Structured
Product Label (SPL) XML snapshots for metformin.

- Source snapshots: 5
- Extracted narrative sections: 214
- Extracted passages: 2,438
- Unique passage identifiers: 2,438
- Repeated normalized-text groups: 672
- Cross-source repeated-text groups: 309
- Distinct normalized passage texts: 1,166
- Singleton passage rows: 494

### Snapshot and extraction integrity

Each source is identified by its DailyMed set ID and SPL version. Existing
snapshot files are reused rather than silently overwritten during regeneration.
New snapshots are downloaded only when the corresponding versioned file is
absent. SHA-256 hashes are recorded in the source manifest and checked during
corpus validation.

The benchmark extractor retains narrative sections, inherits the nearest
titled ancestor for eligible untitled nested sections, and excludes structured
product-composition metadata and recognized administrative/package text.
Patient Information leaflets are retained as narrative evidence. The original
XML snapshots remain the provenance source.

### Duplicate-aware evaluation

Passages with identical normalized text are grouped in
`data/benchmark/duplicate_groups.jsonl`. All passages belonging to the same
duplicate group must remain in the same evaluation partition. Random
passage-level splitting without duplicate grouping is prohibited because
repeated text across labels can cause train/test leakage and inflated scores.

The current duplicate-group integrity check verifies passage references,
occurrence counts, and cross-source flags. It does not establish semantic
equivalence between non-identical passages.

### Annotation and task boundaries

Retrieval annotations are passage-level labels: `relevant` and `not_relevant`.
Claim-level labels such as `supported`, `contradicted`, and `not_mentioned`
belong to a separate claim-verification dataset and must not be inferred from
retrieval similarity scores.

Human annotation, independent review, and adjudication remain pending.
Similarity scores from TF-IDF, embeddings, or hybrid retrieval are ranking
signals, not probabilities of entailment, factual correctness, or clinical
support.

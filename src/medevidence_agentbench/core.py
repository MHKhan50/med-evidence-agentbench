import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


@dataclass
class Evidence:
    text: str
    source_title: str
    source_url: str
    score: float
    section_title: str = ""

    def to_dict(self):
        return asdict(self)


class PlannerAgent:
    name = "planner"

    def plan(self, drug: str, claim: str) -> Dict[str, Any]:
        terms = [
            w.lower()
            for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", claim)
            if w.lower() not in {
                "the", "and", "for", "that", "with", "contains"
            }
        ]
        return {
            "drug": drug.strip(),
            "claim_terms": terms[:12],
            "search_strategy": "DailyMed SPL search followed by XML section extraction and passage ranking"
        }


class DailyMedTool:
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MedEvidence-AgentBench/0.2 research prototype"
        })

    def search(self, drug: str) -> List[Dict[str, Any]]:
        response = self.session.get(
            f"{BASE}/spls.json",
            params={"drug_name": drug, "pagesize": 5},
            timeout=self.timeout
        )
        response.raise_for_status()

        payload = response.json()
        records = payload.get("data", [])

        if isinstance(records, list):
            return records

        # Defensive handling for alternate response structures.
        if isinstance(records, dict):
            nested = records.get("spls", [])
            return nested if isinstance(nested, list) else []

        return []

    def fetch_label(self, setid: str) -> ET.Element:
        response = self.session.get(
            f"{BASE}/spls/{setid}.xml",
            timeout=self.timeout
        )
        response.raise_for_status()
        return ET.fromstring(response.content)


class RetrievalAgent:
    name = "retrieval"

    def __init__(self, tool=None):
        self.tool = tool or DailyMedTool()

    @staticmethod
    def _tag_endswith(element, name):
        return element.tag.split("}")[-1] == name

    def _extract_sections(self, root):
        sections = []

        for section in root.iter():
            if not self._tag_endswith(section, "section"):
                continue

            title = ""

            for child in section:
                if self._tag_endswith(child, "title"):
                    title = " ".join("".join(child.itertext()).split())
                    break

            # Collect this section's own text, excluding nested sections.
            parts = []

            for child in section:
                if self._tag_endswith(child, "section"):
                    continue
                parts.extend(child.itertext())

            content = " ".join(" ".join(parts).split())

            if len(content) >= 45:
                sections.append((title or "Untitled section", content))

        return sections

    def retrieve(self, drug: str, claim: str) -> List[Evidence]:
        records = self.tool.search(drug)
        passages = []

        for record in records[:5]:
            setid = record.get("setid") or record.get("setId")

            if not setid:
                continue

            root = self.tool.fetch_label(setid)

            title = (
                record.get("title")
                or record.get("drug_name")
                or drug
            )

            label_url = (
                "https://dailymed.nlm.nih.gov/dailymed/"
                f"drugInfo.cfm?setid={setid}"
            )

            for section_title, section_text in self._extract_sections(root):
                chunks = re.split(
                    r"(?<=[.!?])\s+",
                    section_text
                )

                for chunk in chunks:
                    chunk = re.sub(r"\s+", " ", chunk).strip()

                    if len(chunk) >= 45:
                        passages.append((
                            chunk[:1200],
                            title,
                            label_url,
                            section_title
                        ))

        if not passages:
            return []

        documents = [claim] + [item[0] for item in passages]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(documents)
        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        ).ravel()

        ranked = sorted(
            zip(scores, passages),
            key=lambda item: item[0],
            reverse=True
        )[:8]

        return [
            Evidence(
                text=item[0],
                source_title=item[1],
                source_url=item[2],
                score=float(score),
                section_title=item[3]
            )
            for score, item in ranked
            if score > 0
        ]


class EvidenceVerifierAgent:
    name = "evidence_verifier"

    def verify(self, claim: str, evidence: List[Evidence]) -> Dict[str, Any]:
        best = max((e.score for e in evidence), default=0.0)

        label = (
            "supported"
            if best >= 0.28
            else "insufficient_evidence"
        )

        return {
            "label": label,
            "confidence_proxy": round(best, 4),
            "method": "TF-IDF lexical similarity threshold; not a medical inference model",
            "warning": "Similarity is not entailment. Human validation is required."
        }


class CitationAuditorAgent:
    name = "citation_auditor"

    def audit(self, evidence: List[Evidence]) -> Dict[str, Any]:
        valid = [
            e for e in evidence
            if e.source_url.startswith(
                "https://dailymed.nlm.nih.gov/"
            )
        ]

        if not evidence:
            status = "no_citations"
        elif len(valid) == len(evidence):
            status = "verified_source_urls"
        else:
            status = "invalid_or_missing_sources"

        return {
            "citation_count": len(valid),
            "all_citations_have_source": (
                bool(evidence) and len(valid) == len(evidence)
            ),
            "status": status
        }


class AgentOrchestrator:
    def __init__(self, retrieval_agent=None):
        self.planner = PlannerAgent()
        self.retriever = retrieval_agent or RetrievalAgent()
        self.verifier = EvidenceVerifierAgent()
        self.auditor = CitationAuditorAgent()

    def run(self, drug: str, claim: str) -> Dict[str, Any]:
        plan = self.planner.plan(drug, claim)
        evidence = self.retriever.retrieve(drug, claim)
        verdict = self.verifier.verify(claim, evidence)
        audit = self.auditor.audit(evidence)

        return {
            "project": "MedEvidence-AgentBench",
            "version": "0.1.0",
            "input": {
                "drug": drug,
                "claim": claim
            },
            "plan": plan,
            "verdict": verdict,
            "evidence": [e.to_dict() for e in evidence],
            "citation_audit": audit,
            "agent_trace": [
                self.planner.name,
                self.retriever.name,
                self.verifier.name,
                self.auditor.name
            ],
            "research_only": True
        }

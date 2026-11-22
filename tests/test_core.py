import xml.etree.ElementTree as ET

from medevidence_agentbench.core import (
    PlannerAgent,
    EvidenceVerifierAgent,
    CitationAuditorAgent,
    RetrievalAgent,
    DailyMedTool,
    Evidence,
)


def test_planner_extracts_terms():
    plan = PlannerAgent().plan(
        "ibuprofen",
        "The label contains a warning about bleeding"
    )
    assert plan["drug"] == "ibuprofen"
    assert "warning" in plan["claim_terms"]


def test_verifier_abstains_without_evidence():
    result = EvidenceVerifierAgent().verify("claim", [])
    assert result["label"] == "insufficient_evidence"


def test_verifier_marks_lexical_match_as_supported_proxy():
    evidence = [
        Evidence(
            "warning about gastrointestinal bleeding",
            "label",
            "https://dailymed.nlm.nih.gov/x",
            0.5
        )
    ]
    assert EvidenceVerifierAgent().verify(
        "warning", evidence
    )["label"] == "supported"


def test_citation_auditor_checks_source():
    evidence = [
        Evidence(
            "text",
            "label",
            "https://dailymed.nlm.nih.gov/x",
            0.4
        )
    ]
    result = CitationAuditorAgent().audit(evidence)
    assert result["all_citations_have_source"] is True
    assert result["citation_count"] == 1


def test_empty_citation_audit_is_not_verified():
    result = CitationAuditorAgent().audit([])
    assert result["citation_count"] == 0
    assert result["all_citations_have_source"] is False
    assert result["status"] == "no_citations"


def test_dailymed_search_parses_list_response():
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {"setid": "test-id", "title": "Test label"}
                ]
            }

    class FakeSession:
        headers = {}

        def get(self, *args, **kwargs):
            return FakeResponse()

    tool = DailyMedTool()
    tool.session = FakeSession()

    records = tool.search("metformin")

    assert len(records) == 1
    assert records[0]["setid"] == "test-id"


def test_xml_section_extraction():
    xml = """
    <document xmlns="urn:hl7-org:v3">
        <section>
            <title>INDICATIONS &amp; USAGE</title>
            <text>
                Metformin is indicated to improve glycemic control
                in adults with type 2 diabetes mellitus.
            </text>
        </section>
    </document>
    """

    root = ET.fromstring(xml)
    sections = RetrievalAgent()._extract_sections(root)

    assert len(sections) == 1
    assert sections[0][0] == "INDICATIONS & USAGE"
    assert "improve glycemic control" in sections[0][1]


def test_xml_nested_sections_are_not_duplicated():
    xml = """
    <document xmlns="urn:hl7-org:v3">
        <section>
            <title>CLINICAL PHARMACOLOGY</title>
            <text>General pharmacology information for metformin.</text>
            <section>
                <title>Mechanism of Action</title>
                <text>Metformin improves glucose tolerance.</text>
            </section>
        </section>
    </document>
    """

    root = ET.fromstring(xml)
    sections = RetrievalAgent()._extract_sections(root)

    assert len(sections) == 2
    assert "Metformin improves glucose tolerance." not in sections[0][1]
    assert "Metformin improves glucose tolerance." in sections[1][1]


def test_retrieval_preserves_section_and_source():
    xml = """
    <document xmlns="urn:hl7-org:v3">
        <section>
            <title>INDICATIONS &amp; USAGE</title>
            <text>
                Metformin is indicated to improve glycemic control
                in adults with type 2 diabetes mellitus.
            </text>
        </section>
    </document>
    """

    class FakeTool:
        def search(self, drug):
            return [{
                "setid": "test-set-id",
                "title": "Metformin Test Label"
            }]

        def fetch_label(self, setid):
            return ET.fromstring(xml)

    evidence = RetrievalAgent(
        tool=FakeTool()
    ).retrieve(
        "metformin",
        "Metformin improves glycemic control in adults with type 2 diabetes."
    )

    assert evidence
    assert evidence[0].section_title == "INDICATIONS & USAGE"
    assert evidence[0].source_url.endswith("setid=test-set-id")
    assert "glycemic control" in evidence[0].text.lower()

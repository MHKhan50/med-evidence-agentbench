import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from medevidence_agentbench.core import DailyMedTool

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "benchmark"
SNAPSHOTS = BENCHMARK / "snapshots"
MANIFEST = BENCHMARK / "source_manifest.jsonl"
PASSAGES = BENCHMARK / "passages.jsonl"


def local_name(tag):
    return tag.split("}")[-1]


def xml_value(root, element_name, attribute):
    for element in root.iter():
        if local_name(element.tag) == element_name:
            return element.attrib.get(attribute)
    return None


def passage_id(source_key, section_index, passage_index, text):
    payload = (
        f"{source_key}|{section_index}|{passage_index}|{text}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]



def extract_benchmark_sections(root):
    """Extract narrative sections while excluding structured product metadata."""
    parents = {
        child: parent
        for parent in root.iter()
        for child in parent
    }

    sections = []

    for section in root.iter():
        if local_name(section.tag) != "section":
            continue

        # Structured product composition is not narrative label evidence.
        if any(
            local_name(element.tag) == "manufacturedProduct"
            for element in section.iter()
        ):
            continue

        title = ""
        for child in section:
            if local_name(child.tag) == "title":
                title = " ".join("".join(child.itertext()).split())
                break

        # Preserve this section's own content, excluding nested sections.
        parts = []
        for child in section:
            if local_name(child.tag) == "section":
                continue
            parts.extend(child.itertext())

        content = " ".join(" ".join(parts).split())

        if len(content) < 45:
            continue

        # Inherit the nearest titled ancestor for untitled narrative sections.
        if not title:
            ancestor = parents.get(section)
            while ancestor is not None:
                if local_name(ancestor.tag) == "section":
                    for child in ancestor:
                        if local_name(child.tag) == "title":
                            title = " ".join(
                                "".join(child.itertext()).split()
                            )
                            break
                    if title:
                        break
                ancestor = parents.get(ancestor)

        # Exclude administrative/manufacturing and package-identification text.
        if not title:
            if re.match(
                r"^(manufactured by:|manufactured for:|"
                r"repackaged and distributed by:|DRUG:|GENERIC:|"
                r"DOSAGE:|ADMINSTRATION:|ADMINISTRATION:|NDC:)",
                content,
                flags=re.IGNORECASE,
            ):
                continue

            if (
                "PATIENT INFORMATION" in content[:500].upper()
                and (
                    "READ THE PATIENT INFORMATION" in content[:900].upper()
                    or content.upper().startswith("PATIENT INFORMATION")
                )
            ):
                title = "Patient Information"
            else:
                title = "Unlabelled narrative section"

        sections.append((title, content))

    return sections


def main():
    parser = argparse.ArgumentParser(
        description="Collect versioned DailyMed label snapshots for research."
    )
    parser.add_argument("--drug", default="metformin")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    BENCHMARK.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and (MANIFEST.exists() or PASSAGES.exists()):
        raise SystemExit(
            "Benchmark outputs already exist. Use --overwrite to replace them."
        )

    tool = DailyMedTool()
    records = tool.search(args.drug)

    if not records:
        raise SystemExit(f"No DailyMed records found for: {args.drug}")

    captured_at = datetime.now(timezone.utc).isoformat()

    manifest_rows = []
    passage_rows = []
    seen_sources = set()

    for record in records:
        setid = record.get("setid") or record.get("setId")
        version = record.get("spl_version")

        if not setid or version is None:
            raise ValueError(f"Missing source identifier or version: {record}")

        source_key = f"{setid}_v{version}"
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        filename = f"{source_key}.xml"
        snapshot_path = SNAPSHOTS / filename

        if snapshot_path.exists():
            xml_bytes = snapshot_path.read_bytes()
            snapshot_status = "reused existing snapshot"
        else:
            response = tool.session.get(
                f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml",
                timeout=tool.timeout,
            )
            response.raise_for_status()
            xml_bytes = response.content
            snapshot_path.write_bytes(xml_bytes)
            snapshot_status = "downloaded new snapshot"

        xml_root = ET.fromstring(xml_bytes)

        xml_setid = xml_value(xml_root, "setId", "root")
        xml_version = xml_value(xml_root, "versionNumber", "value")

        if xml_setid != setid:
            raise ValueError(f"XML setId mismatch for {source_key}")

        if xml_version != str(version):
            raise ValueError(f"XML version mismatch for {source_key}")

        digest = hashlib.sha256(xml_bytes).hexdigest()
        print(f"Snapshot {source_key}: {snapshot_status}")

        title = record.get("title") or args.drug
        source_url = (
            "https://dailymed.nlm.nih.gov/dailymed/"
            f"drugInfo.cfm?setid={setid}"
        )

        manifest_rows.append({
            "source_key": source_key,
            "drug_name": args.drug,
            "setid": setid,
            "spl_version": str(version),
            "published_date": record.get("published_date"),
            "effective_date": xml_value(xml_root, "effectiveTime", "value"),
            "source_title": title,
            "source_url": source_url,
            "snapshot_file": f"snapshots/{filename}",
            "snapshot_sha256": digest,
            "captured_at_utc": captured_at,
        })

        sections = extract_benchmark_sections(xml_root)

        for section_index, (section_title, content) in enumerate(sections):
            chunks = re.split(r"(?<=[.!?])\s+", content)

            passage_index = 0
            for chunk in chunks:
                remaining = re.sub(r"\s+", " ", chunk).strip()
                if len(remaining) < 45:
                    continue

                subchunks = []
                while len(remaining) > 1200:
                    boundary = remaining.rfind(" ", 0, 1201)
                    if boundary < 45:
                        boundary = 1200
                    subchunks.append(remaining[:boundary].strip())
                    remaining = remaining[boundary:].strip()

                if remaining:
                    if (
                        len(remaining) < 45
                        and subchunks
                        and len(subchunks[-1]) + 1 + len(remaining) <= 1200
                    ):
                        subchunks[-1] += " " + remaining
                    else:
                        subchunks.append(remaining)

                for text in subchunks:
                    if len(text) < 45:
                        continue

                    passage_rows.append({
                        "passage_id": passage_id(
                            source_key, section_index, passage_index, text
                        ),
                        "source_key": source_key,
                        "setid": setid,
                        "spl_version": str(version),
                        "source_title": title,
                        "source_url": source_url,
                        "section_title": section_title,
                        "passage_text": text,
                    })
                    passage_index += 1

        print(
            f"Collected {source_key}: "
            f"{len(sections)} sections, "
            f"{sum(p['source_key'] == source_key for p in passage_rows)} passages"
        )

    MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in manifest_rows),
        encoding="utf-8",
    )
    PASSAGES.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in passage_rows),
        encoding="utf-8",
    )

    print("\nCollection complete")
    print("Unique sources:", len(manifest_rows))
    print("Total passages:", len(passage_rows))
    print("Manifest:", MANIFEST.relative_to(ROOT))
    print("Passages:", PASSAGES.relative_to(ROOT))


if __name__ == "__main__":
    main()

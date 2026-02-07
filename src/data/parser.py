from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Dict


def parse_medquad(data_dir: str) -> List[Dict]:
    """
    Parse MedQuAD XML files into QA records.
    Only includes QA pairs with non-empty answers.

    Returns:
        List of dicts:
        {
            doc_id,
            qa_id,
            question,
            answer,
            source,
            url
        }
    """
    records = []
    data_path = Path(data_dir)

    xml_files = list(data_path.rglob("*.xml"))

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            continue

        # Document-level metadata
        document_id = root.attrib.get("id", "").strip()
        source = root.attrib.get("source", "").strip()
        url = root.attrib.get("url", "").strip()

        if not document_id or not source:
            continue

        # Assignment requirement:
        # doc_id = source + "_" + document_id
        doc_id = f"{source}_{document_id}"

        qapairs = root.findall(".//QAPair")

        for qa in qapairs:
            pid = qa.attrib.get("pid", "").strip()
            question = qa.findtext("Question", default="").strip()
            answer = qa.findtext("Answer", default="").strip()

            # Skip empty answers
            if not question or not answer:
                continue

            qa_id = f"{doc_id}_{pid}"

            records.append(
                {
                    "doc_id": doc_id,
                    "qa_id": qa_id,
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "url": url,
                }
            )

    return records

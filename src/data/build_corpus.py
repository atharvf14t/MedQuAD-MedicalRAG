"""Build corpus from MedQuAD XML files.

Parses medical Q&A documents from MedQuAD dataset into JSON format.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_medquad(data_dir: str):
    """
    Parse MedQuAD XML files and return QA records
    with non-empty answers.
    """
    data_path = Path(data_dir)
    xml_files = list(data_path.rglob("*.xml"))

    corpus = []

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            continue

        doc_id = root.attrib.get("id", "").strip()
        source = root.attrib.get("source", "").strip()
        url = root.attrib.get("url", "").strip()

        if not doc_id or not source:
            continue

        full_doc_id = f"{source}_{doc_id}"

        qapairs = root.findall(".//QAPair")

        for qa in qapairs:
            question = qa.findtext("Question", default="").strip()
            answer = qa.findtext("Answer", default="").strip()

            # Skip empty answers
            if not answer:
                continue

            pid = qa.attrib.get("pid", "0")
            qa_id = f"{full_doc_id}_{pid}"

            corpus.append(
                {
                    "qa_id": qa_id,
                    "doc_id": full_doc_id,
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "url": url,
                }
            )

    return corpus


def build_corpus(data_dir: str, output_path: str):
    """Build and save corpus from MedQuAD dataset.
    
    Args:
        data_dir: Path to MedQuAD root directory
        output_path: Output path for corpus JSON file
    """
    corpus = parse_medquad(data_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print("\nCorpus build complete")
    print("---------------------")
    print(f"Total QA pairs stored: {len(corpus)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build corpus from MedQuAD")
    parser.add_argument("--data_dir", required=True, help="Path to MedQuAD root folder")
    parser.add_argument("--output_path", required=True, help="Path to corpus.json")

    args = parser.parse_args()

    build_corpus(args.data_dir, args.output_path)

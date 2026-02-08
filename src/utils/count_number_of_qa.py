"""Count QA statistics in MedQuAD dataset.

Scans XML files and counts QA pairs with/without answers.
"""
# python -m utils.count_number_of_qa  --data_dir MedQuAD-master1
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple


def scan_medquad(data_dir: str) -> Tuple[int, int, int]:
    """
    Scan MedQuAD XML files and compute QA statistics.

    Returns:
        total_qas
        qas_with_answers
        qas_empty_answers
    """
    data_path = Path(data_dir)
    xml_files = list(data_path.rglob("*.xml"))

    total_qas = 0
    qas_with_answers = 0
    qas_empty_answers = 0

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception:
            continue

        # Each document contains QAPairs
        qapairs = root.findall(".//QAPair")

        for qa in qapairs:
            total_qas += 1

            question = qa.findtext("Question", default="").strip()
            answer = qa.findtext("Answer", default="").strip()

            if answer:
                qas_with_answers += 1
            else:
                qas_empty_answers += 1

    return total_qas, qas_with_answers, qas_empty_answers


def main():
    """CLI to print MedQuAD QA statistics."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan MedQuAD QA statistics")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to MedQuAD root folder",
    )

    args = parser.parse_args()

    total, with_ans, empty_ans = scan_medquad(args.data_dir)

    print("\nMedQuAD QA Statistics")
    print("---------------------")
    print(f"Total QA pairs:            {total}")
    print(f"With non-empty answers:    {with_ans}")
    print(f"With empty answers:        {empty_ans}")


if __name__ == "__main__":
    main()

import argparse
import csv
import glob
import json
import os
from pathlib import Path


LABEL_VALUES = {"1", "2", "3"}
LABEL_MEANINGS = {
    "1": "positive",
    "2": "negative",
    "3": "neutral",
}


def find_latest_annotation_csv(root_dir):
    pattern = os.path.join(root_dir, "**", "*_annotation.csv")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f))

    if len(reader) < 5:
        raise ValueError("CSV format is not the expected annotation export.")

    meta_rows = reader[:3]
    header_row = None
    data_start = None
    for index, row in enumerate(reader):
        if row and row[0] == "Tweet ID":
            header_row = row
            data_start = index + 1
            break

    if header_row is None:
        raise ValueError("Could not find the tweet table header row.")

    rows = []
    for row in reader[data_start:]:
        if not row or all(not cell.strip() for cell in row):
            continue
        padded = row + [""] * max(0, len(header_row) - len(row))
        item = {header_row[i]: padded[i] if i < len(padded) else "" for i in range(len(header_row))}
        rows.append(item)

    return meta_rows, rows


def normalize_row(row):
    row.setdefault("Relevant", "")
    row.setdefault("Gold Label", "")
    row.setdefault("Notes", "")
    return row


def prompt_relevant(row, index, total):
    print()
    print(f"[{index + 1}/{total}] Tweet ID: {row.get('Tweet ID', '')}")
    print(f"Time: {row.get('Tweet Date', '')}")
    print(f"User: {row.get('Display Name', '')} {row.get('User Name', '')}")
    print(f"URL: {row.get('Tweet URL', '')}")
    print("Content:")
    print(row.get("Tweet Content", "").strip())
    print("-" * 80)

    while True:
        value = input("Relevant? [1/0, y/n]: ").strip().lower()
        if value in {"1", "y", "yes"}:
            return "1"
        if value in {"0", "n", "no"}:
            return "0"
        print("Please enter 1 or 0.")


def prompt_label():
    while True:
        print("Label map: 1=positive, 2=negative, 3=neutral")
        value = input("Gold Label [1=positive, 2=negative, 3=neutral]: ").strip().lower()
        if value in LABEL_VALUES:
            return value
        print("Please enter 1, 2, or 3.")


# def prompt_notes():
#     value = input("Notes (optional, press Enter to skip): ").strip()
#     return value


def write_output(output_path, meta_rows, rows):
    output_fields = [
        "Tweet ID",
        "Topic",
        "Tweet Date",
        "Display Name",
        "User Name",
        "Tweet URL",
        "Tweet Content",
        "Favorite Count",
        "Retweet Count",
        "Reply Count",
        "Relevant",
        "Gold Label",
        "Notes",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        for row in meta_rows:
            writer.writerow(row)
        writer.writerow(output_fields)
        for row in rows:
            writer.writerow([row.get(field, "") for field in output_fields])


def load_state(state_path):
    if not os.path.exists(state_path):
        return {"index": 0}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path, state):
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Interactive tweet annotation helper.")
    parser.add_argument("--input", help="Input *_annotation.csv path")
    parser.add_argument("--output", help="Output annotated CSV path")
    parser.add_argument("--root", default="result", help="Search root for latest annotation CSV")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        input_path = find_latest_annotation_csv(args.root)
    if not input_path:
        raise FileNotFoundError("No *_annotation.csv file found.")

    input_path = os.path.abspath(input_path)
    meta_rows, rows = load_rows(input_path)
    rows = [normalize_row(row) for row in rows]

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base = Path(input_path)
        output_path = str(base.with_name(f"{base.stem}_labeled{base.suffix}"))

    state_path = str(Path(output_path).with_suffix(".state.json"))
    state = load_state(state_path) if args.resume else {"index": 0}
    start_index = int(state.get("index", 0))
    if start_index < 0 or start_index > len(rows):
        start_index = 0

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Start from row {start_index + 1 if rows else 0}")

    for index in range(start_index, len(rows)):
        row = rows[index]
        relevant = prompt_relevant(row, index, len(rows))
        row["Relevant"] = relevant

        if relevant == "1":
            row["Gold Label"] = prompt_label()
            # row["Notes"] = prompt_notes()
        else:
            row["Gold Label"] = ""
            # row["Notes"] = prompt_notes()

        save_state(state_path, {"index": index + 1, "input_path": input_path, "output_path": output_path})
        write_output(output_path, meta_rows, rows)

        print(f"Saved progress: {index + 1}/{len(rows)}")

    if os.path.exists(state_path):
        os.remove(state_path)

    print("Annotation finished.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

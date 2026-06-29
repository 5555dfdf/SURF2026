import argparse
import csv
import glob
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib import request
from urllib.error import URLError

OLLAMA_API_BASE = "http://127.0.0.1:11434/api"
OLLAMA_GENERATE_URL = f"{OLLAMA_API_BASE}/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_API_BASE}/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_API_BASE}/tags"
OLLAMA_PULL_URL = f"{OLLAMA_API_BASE}/pull"

REASONING_MODELS = {"qwen3", "qwq", "deepseek-r1", "granite"}

DEFAULT_MODELS = ["qwen3:4b"]

# 标签
RELEVANT_LABELS = {"0", "1"}
SENTIMENT_LABELS = {"1", "2", "3"}

SENTIMENT_TO_NAME = {"0": "irrelevant", "1": "positive", "2": "negative", "3": "neutral"}


def find_latest_annotated_csv(root_dir: str) -> str | None:
    pattern = os.path.join(root_dir, "**", "*.csv")
    candidates = []
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            if len(rows) < 5:
                continue
            header = None
            for row in rows:
                if row and row[0] == "Tweet ID":
                    header = row
                    break
            if not header:
                continue
            # 评估 Relevant 和 Gold Label
            if "Relevant" not in header or "Gold Label" not in header:
                continue
            candidates.append(path)
        except Exception:
            continue

    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_annotation_csv(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    meta_rows = []
    header_row = None
    data_start = None
    for index, row in enumerate(rows):
        if row and row[0] == "Tweet ID":
            header_row = row
            data_start = index + 1
            break
        meta_rows.append(row)

    if header_row is None or data_start is None:
        raise ValueError("Could not find tweet table header in CSV.")

    tweets = []
    for row in rows[data_start:]:
        if not row or all(not cell.strip() for cell in row):
            continue
        padded = row + [""] * max(0, len(header_row) - len(row))
        item = {header_row[i]: padded[i] if i < len(padded) else "" for i in range(len(header_row))}
        tweets.append(item)

    return meta_rows, header_row, tweets


def normalize_relevant(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"1", "y", "yes", "true", "relevant"}:
        return "1"
    if v in {"0", "n", "no", "false", "irrelevant"}:
        return "0"
    return "0"  # 默认不相关


def normalize_label(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"1", "positive", "pos", "pro"}:
        return "1"
    if v in {"2", "negative", "neg", "anti"}:
        return "2"
    if v in {"3", "neutral", "neu"}:
        return "3"
    return "3"  # 默认中立


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", model)


def build_prompt(tweet: str) -> str:
    return f"""You are an expert data annotator. Your task is to classify tweets about digital restrictions, regulation, or protection for minors.

=== TASK PIPELINE ===

Step 1: Relevance Check

Relevant = 1 ONLY if BOTH conditions are met:

(A) Mentions minors (children, kids, teens, youth, minors, adolescents, young people, underage, kid, child, teenage, youngsters)

(B) Mentions digital context (social media, TikTok, Instagram, X, Twitter, Facebook, gaming, video games, screen time, internet, online, apps, parental control, online safety, age restriction, platform regulation, digital, cyber, web)

Content types that are RELEVANT:
- Government regulations or age-restriction policies for minors online
- Social media or gaming restrictions/curfews for children
- Parental control tools or children online safety acts

Content types that are IRRELEVANT (mark as 0):
- Mentions children but NO digital context (e.g., education, moral/cultural discussion, real-world child protection issues)
- Mentions digital content but NO minors (e.g., adult social media use, general gaming discussion)
- Broad social/political/cultural topics unrelated to digital restrictions (e.g., nudity/moral debates, AI/economics/international politics, general social discussion)

RULE: BOTH "minors" AND "digital context" must be present. If either is missing → relevant = 0.

Step 2: Sentiment Classification (only if relevant = 1)

Sentiment = 1 (Supports restrictions):
- Supports bans or limits on minors' digital use
- Emphasizes protecting children online
- Supports government or platform regulation
Example: "Children should not have free access to social media."
Example: "Screen time limits are necessary for kids."

Sentiment = 2 (Opposes restrictions):
- Opposes bans on minors' digital use
- Emphasizes freedom or autonomy for young users
- Opposes parental/government control of online access
Example: "Banning teens from social media is wrong."
Example: "Kids should have the freedom to use the internet."

Sentiment = 3 (Neutral / Factual):
- News reporting style
- Describes policies without taking a stance
- States facts objectively or presents multiple viewpoints
Example: "The government is considering screen time limits for minors."
Example: "Experts are discussing youth social media regulation."

=== LABELING EXAMPLES ===

Tweet: "Children should not be allowed on social media unsupervised."
Step 1: Mentions minors (children) + digital context (social media) → relevant = 1
Step 2: Supports restrictions → sentiment = 1
Result: {{"relevant": 1, "sentiment": 1}}

Tweet: "Banning teens from TikTok is an overreach. Kids should have internet freedom."
Step 1: Mentions minors (teens, kids) + digital context (TikTok, internet) → relevant = 1
Step 2: Opposes restrictions → sentiment = 2
Result: {{"relevant": 1, "sentiment": 2}}

Tweet: "New regulations propose limiting screen time for underage users. The policy is under review."
Step 1: Mentions minors (underage) + digital context (screen time, regulations) → relevant = 1
Step 2: Neutral factual statement → sentiment = 3
Result: {{"relevant": 1, "sentiment": 3}}

Tweet: "The education reform bill will change how schools teach math."
Step 1: Mentions children implicitly but NO digital context → relevant = 0
Result: {{"relevant": 0, "sentiment": 0}}

Tweet: "AI will transform the global economy in the next decade."
Step 1: No mention of minors AND no digital restriction context → relevant = 0
Result: {{"relevant": 0, "sentiment": 0}}

=== NOW CLASSIFY THIS TWEET ===

Follow the pipeline: (1) check both conditions for relevance, (2) if relevant, classify sentiment.

Tweet: "{tweet}"

Return ONLY a JSON object with keys "relevant" and "sentiment". Do NOT include any other text, explanation, or markdown formatting."""

def get_json_from_text(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    json_str = None

    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if match:
        json_str = match.group(1)

    if not json_str:
        match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text)
        if match:
            json_str = match.group(1)

    if not json_str:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            json_str = match.group(0)

    if not json_str:
        lines = text.strip().split("\n")
        json_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("{") or stripped.startswith("\"") or stripped.startswith("}") or json_lines:
                json_lines.append(stripped)
        if json_lines:
            candidate = "\n".join(json_lines)
            match = re.search(r"\{[\s\S]*\}", candidate)
            if match:
                json_str = match.group(0)

    if not json_str:
        raise ValueError(f"No JSON object found: {text[:200]}")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        fixed = json_str.replace("'", '"')
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*\]", "]", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON: {exc}. Raw: {json_str[:200]}")


def parse_confidence(value) -> float:
    try:
        confidence = float(value)
    except Exception:
        return 0.0
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


def list_installed_models() -> set[str]:
    with request.urlopen(OLLAMA_TAGS_URL, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {model.get("name", "") for model in data.get("models", []) if model.get("name")}


def pull_model(model: str):
    payload = json.dumps({"model": model, "stream": True}).encode("utf-8")
    req = request.Request(
        OLLAMA_PULL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"Pulling model: {model}")
    with request.urlopen(req, timeout=3600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(line)
                continue
            status = data.get("status")
            if status:
                print(f"  {status}")
            if data.get("error"):
                raise RuntimeError(data["error"])


def ensure_models(models: list[str], pull_missing: bool) -> bool:
    installed = set()
    try:
        installed = list_installed_models()
    except Exception as exc:
        print(f"Warning: could not query installed Ollama models: {exc}")

    missing = [model for model in models if model not in installed]
    if missing and not pull_missing:
        print("Missing models:")
        for model in missing:
            print(f"  - {model}")
        print("Run with --pull-missing to download them.")
        return False

    for model in missing:
        pull_model(model)
    return True


def classify_with_ollama(model: str, tweet: str, timeout: int) -> dict:
    model_family = model.split(":")[0]
    use_chat = model_family in REASONING_MODELS
    prompt_text = build_prompt(tweet)

    if use_chat:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
            },
        }
        url = OLLAMA_CHAT_URL
    else:
        payload = {
            "model": model,
            "prompt": prompt_text,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
            },
        }
        url = OLLAMA_GENERATE_URL

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        response = json.loads(resp.read().decode("utf-8"))

    raw_content = (
        response.get("response")
        or response.get("message", {}).get("content", "")
        or response.get("thinking", "")
    )
    parsed = get_json_from_text(raw_content)

    pred_relevant = normalize_relevant(parsed.get("relevant"))
    pred_sentiment = normalize_label(parsed.get("sentiment"))

    if pred_relevant == "0":
        pred_sentiment = "0"

    return {
        "pred_relevant": pred_relevant,
        "pred_sentiment": pred_sentiment,
        "pred_name": SENTIMENT_TO_NAME.get(pred_sentiment, "irrelevant"),
        "confidence": parse_confidence(parsed.get("confidence")),
        "reason": str(parsed.get("reason") or "").strip(),
        "raw_response": raw_content,
    }


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(records: list[dict]) -> dict:
    gold_rel = [row["gold_relevant"] for row in records]
    pred_rel = [row["pred_relevant"] for row in records]

    gold_sent = [row["gold_sentiment"] for row in records]
    pred_sent = [row["pred_sentiment"] for row in records]

    latencies = [float(row.get("latency_sec", 0.0)) for row in records]
    confidences = [float(row.get("confidence", 0.0)) for row in records]

    total = len(records)

    # 相关性指标
    rel_correct = sum(1 for g, p in zip(gold_rel, pred_rel) if g == p)
    rel_accuracy = rel_correct / total if total else 0.0

    # 仅针对 gold_relevant == "1" 的样本计算情感准确率和 F1
    rel_indices = [i for i, g in enumerate(gold_rel) if g == "1"]
    sub_gold_sent = [gold_sent[i] for i in rel_indices]
    sub_pred_sent = [pred_sent[i] for i in rel_indices]
    rel_total = len(rel_indices)

    sent_correct = sum(1 for g, p in zip(sub_gold_sent, sub_pred_sent) if g == p)
    sent_accuracy = sent_correct / rel_total if rel_total else 0.0

    f1_scores = []
    for label in ["1", "2", "3"]:
        tp = sum(1 for g, p in zip(sub_gold_sent, sub_pred_sent) if g == label and p == label)
        fp = sum(1 for g, p in zip(sub_gold_sent, sub_pred_sent) if g != label and p == label)
        fn = sum(1 for g, p in zip(sub_gold_sent, sub_pred_sent) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_scores.append(f1)

    macro_f1_relevant = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "samples": total,
        "rel_accuracy": rel_accuracy,
        "sent_accuracy": sent_accuracy,
        "sent_samples": rel_total,
        "macro_f1_on_relevant": macro_f1_relevant,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "mean_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
        "median_latency_sec": statistics.median(latencies) if latencies else 0.0,
        "throughput_tps": (total / sum(latencies)) if latencies and sum(latencies) > 0 else 0.0,
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple Ollama models on dual-stage text classification.")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--root", type=str, default="result")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N tweets.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--pull-missing", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    input_path = args.input or find_latest_annotated_csv(args.root)
    if not input_path:
        raise FileNotFoundError("No annotated CSV file found.")
    input_path = os.path.abspath(input_path)

    meta_rows, header_row, tweets = load_annotation_csv(input_path)

    all_tweets = []
    for row in tweets:
        g_rel = normalize_relevant(row.get("Relevant", ""))
        g_sent = normalize_label(row.get("Gold Label", "")) if g_rel == "1" else "0"

        all_tweets.append({
            "Tweet ID": row.get("Tweet ID", ""),
            "Tweet Date": row.get("Tweet Date", ""),
            "Display Name": row.get("Display Name", ""),
            "User Name": row.get("User Name", ""),
            "Tweet URL": row.get("Tweet URL", ""),
            "Tweet Content": row.get("Tweet Content", ""),
            "gold_relevant": g_rel,
            "gold_sentiment": g_sent,
            "gold_name": SENTIMENT_TO_NAME[g_sent],
        })

    if args.limit:
        all_tweets = all_tweets[: args.limit]

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise ValueError("No models specified.")

    if not ensure_models(models, args.pull_missing):
        return

    output_dir = args.output_dir or os.path.join(os.path.dirname(input_path), "model_benchmark")
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_path).stem

    summary_rows = []
    summary_fieldnames = [
        "model", "samples", "rel_accuracy", "sent_accuracy", "macro_f1_on_relevant",
        "avg_confidence", "mean_latency_sec", "median_latency_sec", "throughput_tps"
    ]
    detail_fieldnames = [
        "model", "Tweet ID", "Tweet Content", "gold_relevant", "pred_relevant",
        "gold_sentiment", "pred_sentiment", "gold_name", "pred_name",
        "confidence", "latency_sec", "reason", "correct", "error"
    ]

    for model in models:
        print(f"\n=== Model: {model} ===")
        safe_model = sanitize_model_name(model)
        model_output_path = os.path.join(output_dir, f"{base_name}_{safe_model}.csv")

        existing_rows = {}
        if os.path.exists(model_output_path):
            with open(model_output_path, "r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("Tweet ID"):
                        existing_rows[row["Tweet ID"]] = row

        model_detail_rows = list(existing_rows.values())
        already_done_ids = set(existing_rows.keys())
        start_time = time.perf_counter()

        for index, row in enumerate(all_tweets, start=1):
            tweet_id = row["Tweet ID"]
            if tweet_id in already_done_ids:
                continue

            print(f"[{index}/{len(all_tweets)}] Processing {tweet_id}", flush=True)
            t0 = time.perf_counter()
            error = ""
            try:
                result = classify_with_ollama(model, row["Tweet Content"], args.timeout)
            except (URLError, TimeoutError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                result = {
                    "pred_relevant": "0",
                    "pred_sentiment": "0",
                    "pred_name": "error",
                    "confidence": 0.0,
                    "reason": "",
                    "raw_response": "",
                }
                error = str(exc)

            latency = time.perf_counter() - t0

            # 综合正确性判断（相关性和情感都对才算对）
            is_correct = "1" if (result["pred_relevant"] == row["gold_relevant"] and result["pred_sentiment"] == row[
                "gold_sentiment"]) else "0"

            detail_row = {
                "model": model,
                "Tweet ID": row["Tweet ID"],
                "Tweet Content": row["Tweet Content"],
                "gold_relevant": row["gold_relevant"],
                "pred_relevant": result["pred_relevant"],
                "gold_sentiment": row["gold_sentiment"],
                "pred_sentiment": result["pred_sentiment"],
                "gold_name": row["gold_name"],
                "pred_name": result["pred_name"],
                "confidence": format_float(float(result["confidence"])),
                "latency_sec": format_float(latency),
                "reason": result["reason"],
                "correct": is_correct,
                "error": error,
            }
            model_detail_rows.append(detail_row)
            already_done_ids.add(tweet_id)
            write_csv(model_output_path, model_detail_rows, detail_fieldnames)

            if args.sleep:
                time.sleep(args.sleep)

        metrics = compute_metrics(model_detail_rows)
        summary_rows.append({"model": model, **{k: metrics.get(k, "") for k in summary_fieldnames if k != "model"}})

        # 导出 Markdown 报告
        report_path = os.path.join(output_dir, f"{base_name}_{safe_model}_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Model Evaluation Report: {model}\n\n")
            f.write(f"- Total Samples: {metrics['samples']}\n")
            f.write(f"- Relevance Stage Accuracy: {metrics['rel_accuracy']:.4f}\n")
            f.write(f"- Sentiment Accuracy (Only on Relevant Samples, n={metrics['sent_samples']}): {metrics['sent_accuracy']:.4f}\n")
            f.write(f"- Macro F1 (Only on Relevant Samples): {metrics['macro_f1_on_relevant']:.4f}\n")
            f.write(f"- Mean Latency: {metrics['mean_latency_sec']:.4f}s\n")
            f.write(f"- Throughput: {metrics['throughput_tps']:.4f} tweets/sec\n")

    # 导出对比汇总
    markdown_path = os.path.join(output_dir, f"{base_name}_comparison_summary.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Summary (Two-Stage Pipeline)\n\n")
        f.write("| Model | Samples | Rel Acc | Sent Acc (Rel) | Macro F1 (Rel) | Mean Latency (s) | Throughput |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['model']} | {row['samples']} | {float(row['rel_accuracy']):.4f} | "
                f"{float(row['sent_accuracy']):.4f} | {float(row['macro_f1_on_relevant']):.4f} | "
                f"{float(row['mean_latency_sec']):.4f} | {float(row['throughput_tps']):.4f} |\n"
            )

    print(f"\nSaved summary Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
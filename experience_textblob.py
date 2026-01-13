import pandas as pd
import json
import re

# ---------- helpers ----------

def clean_text(text):
    """Remove HTML tags and normalize whitespace"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_faqs(faqs_raw):
    """Convert faqs JSON into readable text"""
    if not faqs_raw or pd.isna(faqs_raw):
        return ""

    try:
        faqs = json.loads(faqs_raw)
    except Exception:
        return ""

    blocks = []
    for item in faqs:
        q = item.get("title", "").strip()
        a = item.get("content", "").strip()
        if q or a:
            blocks.append(f"Q: {q}\nA: {a}")

    return "\n\n".join(blocks)


def parse_overview(overview_raw):
    """Convert overview JSON into readable text"""
    if not overview_raw or pd.isna(overview_raw):
        return ""

    try:
        overview = json.loads(overview_raw)
    except Exception:
        return ""

    text_parts = []

    # desc can be list of paragraphs
    for para in overview.get("desc", []):
        text_parts.append(para.strip())

    # highlights
    for h in overview.get("highlights", []):
        term = h.get("term", "").strip()
        desc = h.get("desc", "").strip()
        if term or desc:
            text_parts.append(f"{term} {desc}".strip())

    return "\n".join(text_parts)


# ---------- main ----------

INPUT_FILE = "experiences_rows.csv"
OUTPUT_FILE = "experiences_with_textblob.csv"

df = pd.read_csv(INPUT_FILE)

textblobs = []

for _, row in df.iterrows():
    parts = []

    title = clean_text(row.get("title"))
    content = clean_text(row.get("content"))
    address = clean_text(row.get("address"))
    faqs = parse_faqs(row.get("faqs"))
    overview = parse_overview(row.get("overview"))

    if title:
        parts.append(f"Title:\n{title}")
    if content:
        parts.append(f"\nDescription:\n{content}")
    if address:
        parts.append(f"\nAddress:\n{address}")
    if faqs:
        parts.append(f"\nFAQs:\n{faqs}")
    if overview:
        parts.append(f"\nOverview:\n{overview}")

    textblob = "\n\n".join(parts)
    textblobs.append(textblob)

df["textblob"] = textblobs

df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Textblob generation complete → {OUTPUT_FILE}")

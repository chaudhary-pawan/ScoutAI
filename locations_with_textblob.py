import pandas as pd
import json
import re

# ---------- helpers ----------

def clean_text(text):
    """Remove HTML, brackets, and normalize whitespace"""
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove brackets {}, []
    text = re.sub(r"[\{\}\[\]]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_trip_ideas(trip_raw):
    """Robust parser for trip_ideas (list | dict | string)"""
    if not trip_raw or pd.isna(trip_raw):
        return ""

    # If already a list or dict
    if isinstance(trip_raw, (list, dict)):
        data = trip_raw
    else:
        # Try parsing JSON string
        try:
            data = json.loads(trip_raw)
        except Exception:
            # Plain string fallback
            return clean_text(str(trip_raw))

    blocks = []

    # If list of items
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                title = clean_text(item.get("title", ""))
                content = clean_text(item.get("content", ""))
                if title or content:
                    blocks.append(
                        f"{title}:\n{content}" if title and content else title or content
                    )
            else:
                blocks.append(clean_text(str(item)))

    # If single dict
    elif isinstance(data, dict):
        for key, value in data.items():
            value_text = clean_text(str(value))
            if value_text:
                blocks.append(f"{key}:\n{value_text}")

    return "\n\n".join(blocks)



def parse_general_info(info_raw):
    if not info_raw or pd.isna(info_raw):
        return ""

    if isinstance(info_raw, dict):
        data = info_raw
    else:
        try:
            data = json.loads(info_raw)
        except Exception:
            return clean_text(str(info_raw))

    blocks = []

    if isinstance(data, dict):
        for key, value in data.items():
            value_text = clean_text(str(value))
            if value_text:
                blocks.append(f"{key}:\n{value_text}")
    elif isinstance(data, list):
        for item in data:
            blocks.append(clean_text(str(item)))

    return "\n\n".join(blocks)



# ---------- main ----------

INPUT_FILE = "locations_rows.csv"
OUTPUT_FILE = "locations_with_textblob.csv"

df = pd.read_csv(INPUT_FILE)

textblobs = []

for _, row in df.iterrows():
    parts = []

    name = clean_text(row.get("name"))
    content = clean_text(row.get("content"))
    trip_ideas = parse_trip_ideas(row.get("trip_ideas"))
    general_info = parse_general_info(row.get("general_info"))

    if name:
        parts.append(f"Title:\n{name}")

    if content:
        parts.append(f"Description:\n{content}")

    if trip_ideas:
        parts.append(f"Trip Ideas:\n{trip_ideas}")

    if general_info:
        parts.append(f"General Info:\n{general_info}")

    textblob = "\n\n".join(parts)
    textblobs.append(textblob)

df["textblob"] = textblobs

df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Textblob generation complete → {OUTPUT_FILE}")

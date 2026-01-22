import csv
import re
import html

INPUT_FILE = "treks_with_textblob.csv"
OUTPUT_FILE = "treks_rows_cleaned.csv"
TEXTBLOB_COLUMN = "textblob"

# Regex patterns
HTML_TAG_RE = re.compile(r"<[^>]+>")
IMAGE_ID_RE = re.compile(r"\bimage_id\s*:\s*\d+\b", re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Decode HTML entities (&nbsp; etc.)
    text = html.unescape(text)

    # Remove HTML tags
    text = HTML_TAG_RE.sub(" ", text)

    # Remove image_id lines
    text = IMAGE_ID_RE.sub("", text)

    # Remove stray quotes
    text = text.replace('"', "").replace("'", "")

    # Normalize spaces
    text = MULTI_SPACE_RE.sub(" ", text)

    return text.strip()


with open(INPUT_FILE, newline="", encoding="utf-8") as infile, \
    open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    if TEXTBLOB_COLUMN not in fieldnames:
        raise ValueError(f"Column '{TEXTBLOB_COLUMN}' not found in CSV")

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        row[TEXTBLOB_COLUMN] = clean_text(row[TEXTBLOB_COLUMN])
        writer.writerow(row)

print("✅ Cleaning complete.")
print(f"📄 Output written to: {OUTPUT_FILE}")

import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# ======================================================
# CONFIGURATION
# ======================================================
input_file = 'experiences_rows.csv'
output_file = 'experiences_with_textblob_and_embeddings.csv'

# Columns to keep in final file
COLUMNS_TO_KEEP = [
    'title', 'content', 'address', 'faqs', 'overview', 'duration',
    'textblob', 'embedding'
]

print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def clean_html(text):
    if pd.isna(text) or text == "":
        return ""
    try:
        soup = BeautifulSoup(str(text), "html.parser")
        return soup.get_text(separator=" ").strip()
    except:
        return str(text)

def parse_faqs(json_str):
    if pd.isna(json_str) or json_str == "":
        return ""
    try:
        faqs = json.loads(json_str)
        if isinstance(faqs, list):
            return "\n".join([f"Q: {item.get('title','')} A: {item.get('content','')}" for item in faqs])
    except:
        pass
    return ""

def parse_overview(json_str):
    if pd.isna(json_str) or json_str == "":
        return ""
    try:
        data = json.loads(json_str)
        parts = []
        
        # Description
        if 'desc' in data:
            if isinstance(data['desc'], list): parts.append(" ".join(data['desc']))
            else: parts.append(str(data['desc']))
            
        # Highlights
        if 'highlights' in data and isinstance(data['highlights'], list):
            highlights = [f"{h.get('term', '')} {h.get('desc', '')}" for h in data['highlights']]
            parts.append("Highlights: " + "; ".join(highlights))
            
        return "\n".join(parts)
    except:
        pass
    return ""

# ======================================================
# MAIN LOGIC: CREATING "TEXTBLOB"
# ======================================================

def create_textblob(row):
    parts = []
    
    # 1. Title & Address
    if pd.notna(row['title']):
        parts.append(f"Title: {row['title']}")
        
    if pd.notna(row['address']):
        parts.append(f"Location: {row['address']}")
        
    # 2. Duration (With Unit if available)
    if pd.notna(row.get('duration')):
        duration_text = f"Duration: {row['duration']}"
        # Agar 'duration_unit' column exist karta hai (e.g. Hours, Days)
        if 'duration_unit' in row and pd.notna(row['duration_unit']):
            duration_text += f" {row['duration_unit']}"
        parts.append(duration_text)

    # 3. Rich Content
    content = clean_html(row['content'])
    if content:
        parts.append(f"Description:\n{content}")
        
    overview = parse_overview(row['overview'])
    if overview:
        parts.append(f"Overview:\n{overview}")
        
    faqs = parse_faqs(row['faqs'])
    if faqs:
        parts.append(f"FAQs:\n{faqs}")
        
    return "\n\n".join(parts)

print("Creating TextBlob column...")
df['textblob'] = df.apply(create_textblob, axis=1)

# Sample check
print("\n--- Sample TextBlob ---")
print(df['textblob'].iloc[0][:300])
print("-----------------------\n")

# ======================================================
# GENERATING EMBEDDINGS
# ======================================================
print("Loading Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating Embeddings...")
df['embedding'] = model.encode(df['textblob'].tolist(), show_progress_bar=True).tolist()

# ======================================================
# SAVING
# ======================================================
print("Saving final CSV...")
available_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
df_final = df[available_cols]

df_final.to_csv(output_file, index=False)
print(f"Done! New file saved as: {output_file}")
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# ======================================================
# CONFIGURATION
# ======================================================
input_file = 'locations_rows.csv'
output_file = 'locations_with_textblob_and_embeddings.csv'

# Columns to keep in final file
COLUMNS_TO_KEEP = [
    'name', 'content', 'trip_ideas', 'general_info', 
    'map_lat', 'map_lng', 
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

def parse_content(json_str):
    """Main content (desc, highlights) ko extract karta hai"""
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
            highlights = [f"{h.get('term', '')} {h.get('desc', '')}" for h in data['highlights'] if 'desc' in h]
            if highlights:
                parts.append("Highlights: " + "; ".join(highlights))
                
        # Backdrop (Optional, add kar sakte hain agar chahiye)
        if 'backdrop' in data and isinstance(data['backdrop'], list):
             parts.append("Backdrop: " + " ".join(data['backdrop']))
             
        return "\n".join(parts)
    except:
        return ""

def parse_trip_ideas(json_str):
    """Trip Ideas se Title aur Content nikaalta hai"""
    if pd.isna(json_str) or json_str == "":
        return ""
    try:
        ideas = json.loads(json_str)
        if isinstance(ideas, list):
            idea_texts = []
            for idea in ideas:
                # Sirf Title aur Content
                t = idea.get('title', '').strip()
                c = idea.get('content', '').strip()
                if t or c:
                    idea_texts.append(f"{t}: {c}")
            return "\n".join(idea_texts)
    except:
        pass
    return ""

def parse_general_info(json_str):
    """General Info se Title aur Content nikaalta hai"""
    if pd.isna(json_str) or json_str == "":
        return ""
    try:
        infos = json.loads(json_str)
        if isinstance(infos, list):
            info_texts = []
            for info in infos:
                t = info.get('title', '').strip()
                c = info.get('content', '').strip()
                if t or c:
                    info_texts.append(f"{t}: {c}")
            return "\n".join(info_texts)
    except:
        pass
    return ""

# ======================================================
# MAIN LOGIC: CREATING "TEXTBLOB"
# ======================================================

def create_textblob(row):
    parts = []
    
    # 1. Name & Location Info
    if pd.notna(row['name']):
        parts.append(f"Location Name: {row['name']}")
        
    # 2. Numeric Fields (Latitude & Longitude)
    if pd.notna(row.get('map_lat')):
        parts.append(f"Latitude: {row['map_lat']}")
        
    if pd.notna(row.get('map_lng')):
        parts.append(f"Longitude: {row['map_lng']}")

    # 3. Rich Content
    # Main Description
    desc_text = parse_content(row['content'])
    if desc_text:
        parts.append(f"Description:\n{desc_text}")
        
    # Trip Ideas
    ideas_text = parse_trip_ideas(row['trip_ideas'])
    if ideas_text:
        parts.append(f"Trip Ideas:\n{ideas_text}")
        
    # General Info
    info_text = parse_general_info(row['general_info'])
    if info_text:
        parts.append(f"General Info:\n{info_text}")

    # Combine with double newlines
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
# Ensure columns exist
available_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
df_final = df[available_cols]

df_final.to_csv(output_file, index=False)
print(f"Done! New file saved as: {output_file}")
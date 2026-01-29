import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# ======================================================
# CONFIGURATION
# ======================================================
input_file = 'treks_rows.csv'
output_file = 'treks_with_textblob_and_embeddings.csv'

# Ye wo columns hain jo output file mein rahenge
COLUMNS_TO_KEEP = [
    'title', 'content', 'address', 'faqs', 'overview', 'itinerary',
    'map_lat', 'total_distance', 'suitable_age', 'duration',
    'textblob', 'embedding' # textblob aur embedding dono rahenge
]

print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

# ======================================================
# HELPER FUNCTIONS (Cleaning & Parsing)
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
            # Format: "Q: ... A: ..."
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
        if 'desc' in data:
            desc = data['desc']
            if isinstance(desc, list): parts.append(" ".join(desc))
            else: parts.append(str(desc))
            
        if 'highlights' in data and isinstance(data['highlights'], list):
            highlights = [f"{h.get('term', '')} {h.get('desc', '')}" for h in data['highlights']]
            parts.append("Highlights: " + "; ".join(highlights))
        return "\n".join(parts)
    except:
        pass
    return ""

def parse_itinerary(json_str):
    if pd.isna(json_str) or json_str == "":
        return ""
    try:
        days = json.loads(json_str)
        if isinstance(days, list):
            itinerary_text = []
            for day in days:
                day_parts = []
                # Specific fields requested: Title, Desc, Content
                if 'title' in day and day['title']:
                    day_parts.append(f"Day {day['title']}")
                
                if 'desc' in day and day['desc']:
                    day_parts.append(str(day['desc']))
                
                if 'content' in day and day['content']:
                    day_parts.append(clean_html(day['content']))
                
                if day_parts:
                    itinerary_text.append(" | ".join(day_parts))
            return "\n".join(itinerary_text)
    except:
        pass
    return ""

# ======================================================
# MAIN LOGIC: CREATING "TEXTBLOB"
# ======================================================

def create_textblob(row):
    parts = []
    
    # 1. Header Info
    if pd.notna(row['title']):
        parts.append(f"Title: {row['title']}")
    
    if pd.notna(row['address']):
        parts.append(f"Address: {row['address']}")

    # 2. Numeric Fields (Formatted for LLM)
    if pd.notna(row.get('map_lat')):
        parts.append(f"Latitude: {row['map_lat']}")
        
    if pd.notna(row.get('total_distance')):
        parts.append(f"Total Distance: {row['total_distance']} km")
        
    if pd.notna(row.get('duration')):
        parts.append(f"Duration: {row['duration']} Days")
        
    if pd.notna(row.get('suitable_age')):
        parts.append(f"Suitable Age: {row['suitable_age']}")

    # 3. Rich Content Blocks
    # Description
    content = clean_html(row['content'])
    if content:
        parts.append(f"Description:\n{content}")
        
    # Overview
    overview = parse_overview(row['overview'])
    if overview:
        parts.append(f"Overview:\n{overview}")
        
    # Itinerary (Most important for LLM context)
    itinerary = parse_itinerary(row['itinerary'])
    if itinerary:
        parts.append(f"Itinerary:\n{itinerary}")
        
    # FAQs
    faqs = parse_faqs(row['faqs'])
    if faqs:
        parts.append(f"FAQs:\n{faqs}")
        
    # Combine with double newlines for clear separation
    return "\n\n".join(parts)

print("Creating TextBlob column...")
df['textblob'] = df.apply(create_textblob, axis=1)

# Check sample
print("\n--- Sample TextBlob ---")
print(df['textblob'].iloc[0][:300])
print("-----------------------\n")

# ======================================================
# GENERATING EMBEDDINGS
# ======================================================
print("Loading Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating Embeddings from TextBlob...")
# Note: Hum embedding bhi usi 'textblob' se bana rahe hain taaki search accurate ho
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
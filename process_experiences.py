import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup # HTML clean karne ke liye

# ======================================================
# CONFIGURATION
# ======================================================
input_file = 'experiences_rows.csv'
output_file = 'experiences_with_embeddings.csv'

# Columns to keep in final file
COLUMNS_TO_KEEP = ['title', 'content', 'address', 'faqs', 'overview']

print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

# ======================================================
# HELPER FUNCTIONS
# ======================================================

# 1. HTML Cleaner (Content column ke liye)
def clean_html(text):
    if pd.isna(text) or text == "":
        return ""
    try:
        soup = BeautifulSoup(str(text), "html.parser")
        return soup.get_text(separator=" ")
    except:
        return str(text)

# 2. FAQ Parser (JSON se Q&A nikaalne ke liye)
def parse_faqs(json_str):
    if pd.isna(json_str) or json_str == "":
        return ""
    try:
        faqs = json.loads(json_str)
        if isinstance(faqs, list):
            # Format: "Q: What is X? A: It is Y."
            return " ".join([f"Q: {item.get('title','')} A: {item.get('content','')}" for item in faqs])
    except:
        return ""
    return ""

# 3. Overview Parser (JSON se Desc aur Highlights ke liye)
def parse_overview(json_str):
    if pd.isna(json_str) or json_str == "":
        return ""
    try:
        data = json.loads(json_str)
        parts = []
        
        # Description
        if 'desc' in data and isinstance(data['desc'], list):
            parts.append(" ".join(data['desc']))
            
        # Highlights
        if 'highlights' in data and isinstance(data['highlights'], list):
            highlights = [f"{h.get('term', '')} {h.get('desc', '')}" for h in data['highlights']]
            parts.append("Highlights: " + "; ".join(highlights))
            
        return " ".join(parts)
    except:
        return ""
    return ""

# ======================================================
# MAIN LOGIC
# ======================================================

def create_search_text(row):
    text_stack = []
    
    # 1. Title (High Priority)
    if pd.notna(row['title']):
        text_stack.append(f"Experience: {row['title']}")
    
    # 2. Address
    if pd.notna(row['address']):
        text_stack.append(f"Location: {row['address']}")
        
    # 3. Content (Cleaned HTML)
    text_stack.append(clean_html(row['content']))
    
    # 4. Overview (Parsed JSON)
    text_stack.append(parse_overview(row['overview']))
    
    # 5. FAQs (Parsed JSON)
    text_stack.append(parse_faqs(row['faqs']))
    
    # Join everything
    return " ".join(text_stack)

print("Processing text (Cleaning HTML & Parsing JSON)...")
df['search_text'] = df.apply(create_search_text, axis=1)

# Generate Embeddings
print("Loading Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating Embeddings...")
df['embedding'] = model.encode(df['search_text'].tolist(), show_progress_bar=True).tolist()

# ======================================================
# SAVING
# ======================================================
print("Saving final file...")

# Columns filter karna
final_columns = COLUMNS_TO_KEEP + ['embedding']
available_cols = [c for c in final_columns if c in df.columns]

df_final = df[available_cols]
df_final.to_csv(output_file, index=False)

print(f"Done! File saved as: {output_file}")
import psycopg2
import json
from sentence_transformers import SentenceTransformer

# ============================
# DATABASE CONFIG
# ============================
DB_CONFIG = {
    "host": "aws-1-ap-south-1.pooler.supabase.com",
    "port": 6543,
    "dbname": "postgres",
    "user": "postgres.qwozjwjqhgqrgvjadrgm",
    "password": "Scoutripper193",
    "sslmode": "require"
}

# ============================
# MAIN PIPELINE
# ============================

def main():
    print("🔌 Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("📥 Fetching experiences...")
    cur.execute("""
        SELECT
            id,
            title,
            slug,
            location_id,
            address,
            duration,
            price,
            sale_price,
            textblob
        FROM experiences
        WHERE textblob IS NOT NULL
    """)

    rows = cur.fetchall()
    print(f"📊 Found {len(rows)} experiences")

    for row in rows:
        (
            experience_id,
            title,
            slug,
            location_id,
            address,
            duration,
            price,
            sale_price,
            textblob
        ) = row

        # ---------------- METADATA ----------------
        metadata = {
            "experience_id": experience_id,
            "title": title,
            "slug": slug,
            "location_id": location_id,
            "address": address,
            "duration": duration,
            "price": price,
            "sale_price": sale_price
        }

        # ---------------- EMBEDDING ----------------
        embedding = model.encode(
            textblob,
            normalize_embeddings=True
        )

        cur.execute("""
            INSERT INTO experience_embeddings
            (experience_id, chunk_index, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """, (
                experience_id,
                0,                  # ✅ chunk_index fixed
                textblob,
                embedding.tolist(),
                json.dumps(metadata)
                ))


        print(f"✅ Embedded experience {experience_id}")

    conn.commit()
    cur.close()
    conn.close()
    print("🎉 ALL EXPERIENCES EMBEDDED SUCCESSFULLY")


if __name__ == "__main__":
    main()

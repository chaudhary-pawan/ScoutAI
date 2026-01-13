import psycopg2
from sentence_transformers import SentenceTransformer
import json

# ---------------- CONFIG ----------------
DB_CONFIG = {
    "host": "aws-1-ap-south-1.pooler.supabase.com",
    "port": 6543,
    "dbname": "postgres",
    "user": "postgres.qwozjwjqhgqrgvjadrgm",
    "password": "Scoutripper193",
    "sslmode": "require"
}

MODEL_NAME = "all-MiniLM-L6-v2"
# --------------------------------------


def main():
    print("🔌 Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("📥 Fetching locations...")
    cur.execute("""
        SELECT
            id,
            textblob,
            name,
            slug,
            map_lat,
            map_lng
        FROM locations
        WHERE textblob IS NOT NULL
    """)
    rows = cur.fetchall()

    print(f"📊 Found {len(rows)} locations")

    for row in rows:
        (
            location_id,
            textblob,
            name,
            slug,
            map_lat,
            map_lng
        ) = row

        if not textblob.strip():
            continue

        # --- create embedding ---
        embedding = model.encode(
            textblob,
            normalize_embeddings=True
        )

        # --- metadata ---
        metadata = {
            "name": name,
            "slug": slug,
            "map_lat": map_lat,
            "map_lng": map_lng
        }

        # --- insert ---
        cur.execute("""
            INSERT INTO locations_embeddings
            (locations_id, chunk_index, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            location_id,
            0,                  # SINGLE chunk
            textblob,
            embedding.tolist(),
            json.dumps(metadata)
        ))

        print(f"✅ Embedded location {location_id}")

    conn.commit()
    cur.close()
    conn.close()

    print("🎉 ALL LOCATION EMBEDDINGS STORED SUCCESSFULLY")


if __name__ == "__main__":
    main()

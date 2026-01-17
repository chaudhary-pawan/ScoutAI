import json
import psycopg2
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
DB_CONFIG = {
    "host": "aws-1-ap-south-1.pooler.supabase.com",
    "port": 6543,
    "dbname": "postgres",
    "user": "postgres.qwozjwjqhgqrgvjadrgm",
    "password": "Scoutripper193",  # 🔐
    "sslmode": "require"
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# --------------------------------------


def main():
    print("🔌 Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("📥 Fetching treks...")
    cur.execute("""
        SELECT
            id,
            textblob,
            title,
            slug,
            address,
            map_lat,
            map_lng,
            price,
            sale_price,
            duration,
            min_people,
            max_people,
            total_distance,
            altitude,
            suitable_age,
            include,
            exclude,
            min_day_before_booking
        FROM treks
        WHERE textblob IS NOT NULL
    """)

    rows = cur.fetchall()
    print(f"📊 Found {len(rows)} treks")

    for row in rows:
        (
            treks_id,
            textblob,
            title,
            slug,
            address,
            map_lat,
            map_lng,
            price,
            sale_price,
            duration,
            min_people,
            max_people,
            total_distance,
            altitude,
            suitable_age,
            include,
            exclude,
            min_day_before_booking
        ) = row

        # --------- Build metadata ----------
        metadata = {
            "trek_id": treks_id,
            "title": title,
            "slug": slug,
            "address": address,
            "map_lat": map_lat,
            "map_lng": map_lng,
            "price": price,
            "sale_price": sale_price,
            "duration": duration,
            "min_people": min_people,
            "max_people": max_people,
            "total_distance": total_distance,
            "altitude": altitude,
            "suitable_age": suitable_age,
            "include": include,
            "exclude": exclude,
            "min_day_before_booking": min_day_before_booking
        }

        # --------- Create embedding ----------
        embedding = model.encode(
            textblob,
            normalize_embeddings=True
        )

        # --------- Insert ----------
        cur.execute("""
            INSERT INTO treks_embeddings
            (treks_id, chunk_index, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            treks_id,
            0,                 # ✅ single-document embedding
            textblob,
            embedding.tolist(),
            json.dumps(metadata)
        ))

        print(f"✅ Embedded trek {treks_id}")

    conn.commit()
    cur.close()
    conn.close()
    print("🎉 ALL TREKS EMBEDDED SUCCESSFULLY")


if __name__ == "__main__":
    main()

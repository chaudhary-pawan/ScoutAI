import pandas as pd
import psycopg2

# ---------- CONFIG ----------
CSV_FILE = "locations_with_textblob.csv"

DB_CONFIG = {
    "host": "aws-1-ap-south-1.pooler.supabase.com",
    "port": 6543,
    "dbname": "postgres",
    "user": "postgres.qwozjwjqhgqrgvjadrgm",
    "password": "Scoutripper193",
    "sslmode": "require"
}
# ----------------------------


def main():
    print("📥 Loading CSV...")
    df = pd.read_csv(CSV_FILE)

    print(f"📊 Rows found: {len(df)}")

    print("🔌 Connecting to Supabase...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    updated = 0

    for _, row in df.iterrows():
        location_id = int(row["id"])
        textblob = str(row["textblob"]).strip()

        if not textblob or textblob == "nan":
            continue

        cur.execute(
            """
            UPDATE locations
            SET textblob = %s
            WHERE id = %s
            """,
            (textblob, location_id)
        )

        if cur.rowcount > 0:
            updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Updated {updated} location rows with textblob")


if __name__ == "__main__":
    main()

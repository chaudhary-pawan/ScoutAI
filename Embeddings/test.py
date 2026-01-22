import psycopg2

print("Trying connection...")

conn = psycopg2.connect(
    host="aws-1-ap-south-1.pooler.supabase.com",
    port=6543,
    dbname="postgres",
    user="postgres.qwozjwjqhgqrgvjadrgm",
    password="Scoutripper193",
    sslmode="require"
)

print("CONNECTED SUCCESSFULLY")
conn.close()

import nltk
nltk.download("punkt")

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()  

def get_connection():
    return psycopg.connect(os.environ["DATABASE_URL"])

if __name__ == "__main__":
    with get_connection() as conn:
        print(conn.execute("SELECT version()").fetchone())

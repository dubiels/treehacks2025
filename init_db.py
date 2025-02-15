import sqlite3

DATABASE = "captcha.db"

def initialize_db():
    """Creates the database and table for storing Imgur URLs."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    initialize_db()

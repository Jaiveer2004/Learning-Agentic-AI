import sqlite3

DATABASE_NAME = "bank_data.db"

def get_db():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    
    try:
        yield conn
    finally:
        conn.close()
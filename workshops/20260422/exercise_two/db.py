import sqlite3

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_username(username):
    conn = get_db_connection()
    
    query = f"SELECT * FROM users WHERE username = '{username}'"
    
    user = conn.execute(query).fetchone()
    conn.close()
    return user
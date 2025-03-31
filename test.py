import sqlite3

conn = sqlite3.connect("ptt_data.db")
cur = conn.cursor()
cur.execute("SELECT id, title, title_star_label, content, content_star_label FROM sentiments LIMIT 10")
rows = cur.fetchall()
for row in rows:
    print(row)
conn.close()
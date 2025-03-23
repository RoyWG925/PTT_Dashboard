import pandas as pd
from sqlalchemy import create_engine
import sqlite3

# 定義 PostgreSQL 連線參數
PG_HOST = "localhost"
PG_PORT = 5432
PG_DBNAME = "ptt_db"
PG_USER = "ptt_user"
PG_PASSWORD = "ptt_password"

# 從 PostgreSQL 讀取
pg_engine = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}")
df_sentiments = pd.read_sql_query("SELECT * FROM sentiments", pg_engine)
df_push = pd.read_sql_query("SELECT * FROM push_comments", pg_engine)

# 寫入 SQLite
sqlite_conn = sqlite3.connect("ptt_data.db")
df_sentiments.to_sql("sentiments", sqlite_conn, if_exists="replace", index=False)
df_push.to_sql("push_comments", sqlite_conn, if_exists="replace", index=False)
sqlite_conn.close()

print("資料已成功匯出到 ptt_data.db")
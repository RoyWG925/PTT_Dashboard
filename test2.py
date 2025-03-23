import sqlite3
import pandas as pd

# 連接到原始資料庫
source_conn = sqlite3.connect("ptt_data.db")

# 讀取並抽樣 sentiments 表
df_sentiments = pd.read_sql_query("SELECT * FROM sentiments", source_conn)
total_rows = len(df_sentiments)
target_rows = total_rows // 3  # 取 1/3 的記錄數
df_sentiments_sampled = df_sentiments.sample(n=target_rows, random_state=42)  # 隨機抽樣

# 縮減 content 欄位（可選）
df_sentiments_sampled['content'] = df_sentiments_sampled['content'].apply(lambda x: x[:100] if x else x)  # 保留前 100 字元

# 讀取並抽樣 push_comments 表（只保留相關的 article_id）
sampled_ids = df_sentiments_sampled['id'].tolist()
df_push = pd.read_sql_query("SELECT * FROM push_comments WHERE article_id IN (" + ",".join(map(str, sampled_ids)) + ")", source_conn)
total_push_rows = len(df_push)
target_push_rows = total_push_rows // 3
df_push_sampled = df_push.sample(n=target_push_rows, random_state=42) if total_push_rows > 0 else df_push

# 縮減 push_content 欄位（可選）
df_push_sampled['push_content'] = df_push_sampled['push_content'].apply(lambda x: x[:50] if x else x)  # 保留前 50 字元

# 關閉原始連線
source_conn.close()

# 建立新的資料庫
target_conn = sqlite3.connect("ptt_data_reduced.db")

# 寫入抽樣後的資料
df_sentiments_sampled.to_sql("sentiments", target_conn, if_exists="replace", index=False)
df_push_sampled.to_sql("push_comments", target_conn, if_exists="replace", index=False)

# 執行 VACUUM 壓縮資料庫
target_conn.execute("VACUUM")
target_conn.close()

print(f"已生成減少後的資料庫 ptt_data_reduced.db，sentiments 記錄數: {len(df_sentiments_sampled)}, push_comments 記錄數: {len(df_push_sampled)}")
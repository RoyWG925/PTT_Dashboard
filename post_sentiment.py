from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import time
import logging
import psycopg2
import sys

# ----------------------------
# Logging 設定
# ----------------------------
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filename="post_sentiment.log",
        filemode='a',  # 使用追加模式，保留之前的 log
        encoding='utf-8'
    )
    logging.info("Logging initialized (post-sentiment, batch mode)")
except Exception as e:
    print(f"Logging init error: {e}")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ----------------------------
# PostgreSQL 連線參數
# ----------------------------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DBNAME = "ptt_db"
PG_USER = "ptt_user"
PG_PASSWORD = "ptt_password"

# ----------------------------
# 初始化情緒分析模型 (使用 GPU, 可改 device=-1 用 CPU)
# ----------------------------
from transformers import pipeline
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0,
        truncation=True,
        max_length=512
    )
    logging.info("Sentiment analyzer initialized (GPU, batch mode)")
except Exception as e:
    logging.error(f"Model initialization failed: {e}")
    sys.exit(1)

def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD
    )

# ----------------------------
# star_label 轉換為情緒
# ----------------------------
def star_label_to_sentiment(star_label: str) -> str:
    star = int(star_label[0])
    if star <= 2:
        return "NEGATIVE"
    elif star == 3:
        return "NEUTRAL"
    else:
        return "POSITIVE"

# ----------------------------
# 確保需要的欄位已存在
# ----------------------------
def ensure_db_columns():
    conn = get_pg_connection()
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE sentiments ADD COLUMN title_star_label TEXT")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    try:
        cur.execute("ALTER TABLE sentiments ADD COLUMN title_sentiment TEXT")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    try:
        cur.execute("ALTER TABLE sentiments ADD COLUMN title_score DOUBLE PRECISION")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    try:
        cur.execute("ALTER TABLE sentiments ADD COLUMN content_star_label TEXT")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    try:
        cur.execute("ALTER TABLE sentiments ADD COLUMN content_sentiment TEXT")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    try:
        cur.execute("ALTER TABLE sentiments ADD COLUMN content_score DOUBLE PRECISION")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    try:
        cur.execute("ALTER TABLE push_comments ADD COLUMN push_star_label TEXT")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    try:
        cur.execute("ALTER TABLE push_comments ADD COLUMN push_sentiment TEXT")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    try:
        cur.execute("ALTER TABLE push_comments ADD COLUMN push_score DOUBLE PRECISION")
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()
    conn.commit()
    cur.close()
    conn.close()

# ----------------------------
# 批次推論
# ----------------------------
def batch_inference(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_out = sentiment_analyzer(
            batch,
            truncation=True,
            max_length=512
        )
        for out in batch_out:
            star_label = out["label"]
            confidence = out["score"]
            sentiment_label = star_label_to_sentiment(star_label)
            results.append((star_label, sentiment_label, confidence))
    return results

# ----------------------------
# 分析 sentiments (title, content) in batch
# ----------------------------
def analyze_sentiments_main():
    conn = get_pg_connection()
    cur = conn.cursor()
    # 只取尚未更新情緒的文章，避免重複分析
    cur.execute("SELECT id, title, content FROM sentiments WHERE title_star_label IS NULL OR content_star_label IS NULL ORDER BY id ASC")
    rows = cur.fetchall()
    total = len(rows)
    logging.info(f"Found {total} articles to analyze (title & content) that haven't been updated.")
    if total == 0:
        logging.info("All articles already analyzed.")
        cur.close()
        conn.close()
        return

    title_texts = []
    content_texts = []
    article_ids = []

    for idx, r in enumerate(rows):
        article_id, title, content = r
        article_ids.append(article_id)
        title_texts.append(title if title else "")
        content_texts.append(content if content else "")
        # 每處理 100 筆 log 一次進度
        if (idx+1) % 100 == 0:
            logging.info(f"Prepared {idx+1}/{total} articles for analysis.")

    logging.info("Start batch inference for titles...")
    title_results = batch_inference(title_texts, batch_size=16)
    logging.info("Start batch inference for contents...")
    content_results = batch_inference(content_texts, batch_size=16)

    for i, article_id in enumerate(article_ids):
        title_star, title_sent, title_score = title_results[i]
        cont_star, cont_sent, cont_score = content_results[i]
        try:
            update_sql = """
            UPDATE sentiments
            SET title_star_label=%s,
                title_sentiment=%s,
                title_score=%s,
                content_star_label=%s,
                content_sentiment=%s,
                content_score=%s
            WHERE id=%s
            """
            cur.execute(update_sql, (
                title_star, title_sent, title_score,
                cont_star, cont_sent, cont_score,
                article_id
            ))
        except Exception as e:
            logging.error(f"Update sentiments failed for id={article_id}: {e}")
        if (i+1) % 100 == 0:
            logging.info(f"Updated {i+1}/{total} articles.")
    conn.commit()
    cur.close()
    conn.close()
    logging.info("Done updating sentiments (title & content).")

# ----------------------------
# 分析 push_comments in batch
# ----------------------------
def analyze_push_comments():
    conn = get_pg_connection()
    cur = conn.cursor()
    # 只選擇尚未更新推文情緒的資料
    cur.execute("SELECT id, push_content FROM push_comments WHERE push_star_label IS NULL ORDER BY id ASC")
    rows = cur.fetchall()
    total = len(rows)
    logging.info(f"Found {total} push comments to analyze.")
    if total == 0:
        logging.info("All push comments already analyzed.")
        cur.close()
        conn.close()
        return

    push_ids = []
    push_texts = []
    for idx, (push_id, push_content) in enumerate(rows):
        push_ids.append(push_id)
        push_texts.append(push_content if push_content else "")
        if (idx+1) % 100 == 0:
            logging.info(f"Prepared {idx+1}/{total} push comments for analysis.")

    logging.info("Start batch inference for push_comments...")
    push_results = batch_inference(push_texts, batch_size=16)

    for i, push_id in enumerate(push_ids):
        star_label, sentiment_label, score = push_results[i]
        try:
            update_sql = """
            UPDATE push_comments
            SET push_star_label=%s,
                push_sentiment=%s,
                push_score=%s
            WHERE id=%s
            """
            cur.execute(update_sql, (star_label, sentiment_label, score, push_id))
        except Exception as e:
            logging.error(f"Update push_comments failed for id={push_id}: {e}")
        if (i+1) % 100 == 0:
            logging.info(f"Updated {i+1}/{total} push comments.")
    conn.commit()
    cur.close()
    conn.close()
    logging.info("Done updating push_comments.")

def main():
    ensure_db_columns()
    analyze_sentiments_main()
    analyze_push_comments()
    logging.info("Post-sentiment batch analysis done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Main error: {e}")
        sys.exit(1)

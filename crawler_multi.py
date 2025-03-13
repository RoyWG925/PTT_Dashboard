from datetime import datetime
import requests
import re
import time
import logging
import psycopg2
import sys
from bs4 import BeautifulSoup
from sqlalchemy import create_engine

# ----------------------------
# 看板與頁碼參數
# 這裡示範:
# NBA 從 index6503 → 6400
# Stock 從 index8442 → 8350
# ----------------------------
BOARD_CONFIG = [
    {
        "board": "NBA",
        "start_page": 6503,
        "end_page": 6400,
    },
    {
        "board": "Stock",
        "start_page": 8442,
        "end_page": 8350,
    }
]

SLEEP_SEC = 1
LOG_FILE = "multi_crawler.log"

# PostgreSQL 連線參數
PG_HOST = "localhost"
PG_PORT = 5432
PG_DBNAME = "ptt_db"
PG_USER = "ptt_user"
PG_PASSWORD = "ptt_password"

# ----------------------------
# Logging 設定
# ----------------------------
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filename=LOG_FILE,
        filemode='w',
        encoding='utf-8'
    )
    logging.info("Logging initialized (multi-board crawler)")
except Exception as e:
    print(f"Logging init error: {e}")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ----------------------------
# PostgreSQL 連線函式
# ----------------------------
def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DBNAME,
        user=PG_USER,
        password=PG_PASSWORD
    )

# ----------------------------
# 資料庫初始化（同原結構）
# ----------------------------
def init_db():
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentiments (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP,
        board TEXT,
        title TEXT,
        content TEXT,
        link TEXT UNIQUE
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS push_comments (
        id SERIAL PRIMARY KEY,
        article_id INT,
        push_tag TEXT,
        push_userid TEXT,
        push_content TEXT,
        push_time TEXT,
        CONSTRAINT fk_article FOREIGN KEY(article_id)
           REFERENCES sentiments(id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

# ----------------------------
# 解析文章時間
# ----------------------------
def fetch_article_time(soup):
    post_time_str = None
    metalines = soup.find_all("div", class_="article-metaline")
    metalines_right = soup.find_all("div", class_="article-metaline-right")
    all_metalines = metalines + metalines_right
    for meta in all_metalines:
        tag_span = meta.find("span", class_="article-meta-tag")
        value_span = meta.find("span", class_="article-meta-value")
        if tag_span and value_span and "時間" in tag_span.text.strip():
            post_time_str = value_span.text.strip()
            break

    # 若沒在標準位置找到，就用全文正則
    if not post_time_str:
        full_text = soup.get_text()
        pattern = r"[A-Z][a-z]{2}\s[A-Z][a-z]{2}\s{1,2}\d{1,2}\s\d{2}:\d{2}:\d{2}\s\d{4}"
        match = re.search(pattern, full_text)
        if match:
            post_time_str = match.group(0).strip()

    # 嘗試解析
    if post_time_str:
        try:
            dt = datetime.strptime(post_time_str, '%a %b %d %H:%M:%S %Y')
            return dt
        except ValueError as e:
            logging.warning(f"Time parsing failed: {post_time_str}, {e}")
            # 若解析失敗，fallback 用現在時間
            return datetime.now().replace(microsecond=0)
    else:
        # 若沒有任何時間資訊，就用現在時間
        return datetime.now().replace(microsecond=0)

# ----------------------------
# 抓取文章內容 + 推文
# ----------------------------
def fetch_content_and_push(link_url: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    cookies = {"over18": "1"}
    try:
        resp = requests.get(link_url, headers=headers, cookies=cookies, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        post_time = fetch_article_time(soup)

        # 內文
        main_content = soup.find(id="main-content")
        content_text = ""
        if main_content:
            # 移除 article-metaline
            for meta in main_content.find_all("div", class_=re.compile("article-metaline")):
                meta.decompose()
            for meta_r in main_content.find_all("div", class_=re.compile("article-metaline-right")):
                meta_r.decompose()
            content_text = main_content.get_text().strip()

        # 推文
        push_list = []
        pushes = soup.find_all("div", class_="push")
        for p in pushes:
            tag_span = p.find("span", class_="push-tag")
            userid_span = p.find("span", class_="push-userid")
            content_span = p.find("span", class_="push-content")
            time_span = p.find("span", class_="push-ipdatetime")
            push_tag = tag_span.get_text(strip=True) if tag_span else ""
            push_userid = userid_span.get_text(strip=True) if userid_span else ""
            push_content = content_span.get_text(strip=True).lstrip(":") if content_span else ""
            push_time = time_span.get_text(strip=True) if time_span else ""
            push_list.append({
                "tag": push_tag,
                "userid": push_userid,
                "content": push_content,
                "time": push_time
            })

        return post_time, content_text, push_list
    except Exception as e:
        logging.error(f"Fetching content failed: {e}, URL: {link_url}")
        # 無法抓取則回傳預設
        return datetime.now().replace(microsecond=0), "", []

# ----------------------------
# 寫入資料庫
# ----------------------------
def save_article_and_push(timestamp, board, title, content, link, push_list):
    conn = get_pg_connection()
    cur = conn.cursor()
    article_id = None
    try:
        insert_sql = """
        INSERT INTO sentiments(timestamp, board, title, content, link)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        cur.execute(insert_sql, (timestamp, board, title, content, link))
        article_id = cur.fetchone()[0]
        conn.commit()
    except psycopg2.IntegrityError:
        logging.info(f"Duplicate article, skipping: {link}")
        conn.rollback()
        cur.close()
        conn.close()
        return
    except Exception as e:
        logging.error(f"Inserting main article failed: {e}")
        conn.rollback()
        cur.close()
        conn.close()
        return

    # 寫入推文
    try:
        for p in push_list:
            cur.execute("""
            INSERT INTO push_comments(article_id, push_tag, push_userid, push_content, push_time)
            VALUES (%s, %s, %s, %s, %s)
            """, (article_id, p["tag"], p["userid"], p["content"], p["time"]))
        conn.commit()
    except Exception as e:
        logging.error(f"Inserting push_comments failed: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

# ----------------------------
# 爬取單一看板
# ----------------------------
def crawl_board(board, start_page, end_page):
    total_pages = abs(start_page - end_page) + 1
    pages_processed = 0
    step = -1 if start_page > end_page else 1
    page = start_page

    logging.info(f"Start crawling board={board}, from index{start_page} to index{end_page}")

    while (step < 0 and page >= end_page) or (step > 0 and page <= end_page):
        pages_processed += 1
        progress = (pages_processed / total_pages) * 100
        logging.info(f"[{board}] Processing page {page}, progress: {pages_processed}/{total_pages} ({progress:.1f}%)")

        url = f"https://www.ptt.cc/bbs/{board}/index{page}.html"
        headers = {"User-Agent":"Mozilla/5.0"}
        cookies = {"over18":"1"}

        try:
            resp = requests.get(url, headers=headers, cookies=cookies, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            articles = soup.select(".r-ent")
        except Exception as e:
            logging.error(f"[{board}] Error reading {url}: {e}")
            break

        for art in articles:
            title_tag = art.select_one(".title a")
            if not title_tag:
                continue
            title = title_tag.text.strip()
            link = "https://www.ptt.cc" + title_tag["href"]

            post_time, content_text, push_list = fetch_content_and_push(link)
            save_article_and_push(post_time, board, title, content_text, link, push_list)

        page += step
        time.sleep(SLEEP_SEC)

    logging.info(f"Crawling {board} finished. (no sentiment analysis)")

def main():
    # 初始化資料庫
    init_db()

    # 一次執行多看板爬蟲
    for conf in BOARD_CONFIG:
        board = conf["board"]
        start_p = conf["start_page"]
        end_p = conf["end_page"]
        crawl_board(board, start_p, end_p)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Main error: {e}")
        sys.exit(1)

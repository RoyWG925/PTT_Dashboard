import streamlit as st
import pandas as pd
import math
import plotly.express as px
from sqlalchemy import create_engine
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols

#############################
# 建立 SQLAlchemy engine (SQLite)
#############################
def get_engine():
    # 假設 ptt_data.db 在與程式碼相同的目錄下
    db_uri = "sqlite:///ptt_data.db"
    engine = create_engine(db_uri)
    return engine

#############################
# star_label -> 數字
#############################
def star_label_to_int(star_label: str):
    if not star_label:
        return None
    try:
        return int(star_label[0])  # "1 star" -> 1, "5 stars" -> 5
    except:
        return None

#############################
# 顏色映射 + HTML 呈現
#############################
def sentiment_color(star_int):
    if star_int == 1:
        return "#d73027"
    elif star_int == 2:
        return "#fc8d59"
    elif star_int == 3:
        return "#fee08b"
    elif star_int == 4:
        return "#d9ef8b"
    elif star_int == 5:
        return "#1a9850"
    return "#cccccc"

def color_star_label(star_label):
    if not star_label:
        return "未知"
    try:
        star_int = int(star_label[0])
        col = sentiment_color(star_int)
        return f"<span style='color:{col}; font-weight:bold;'>{star_label}</span>"
    except:
        return star_label

#############################
# 讀取文章 (支援看板篩選)
#############################
def fetch_articles(board_filter=None):
    engine = get_engine()
    if board_filter and board_filter != "All":
        sql = f"""
        SELECT id, timestamp, board, title, content,
               title_star_label, content_star_label
        FROM sentiments
        WHERE board = '{board_filter}'
        ORDER BY timestamp DESC
        """
    else:
        sql = """
        SELECT id, timestamp, board, title, content,
               title_star_label, content_star_label
        FROM sentiments
        ORDER BY timestamp DESC
        """
    df = pd.read_sql_query(sql, engine)
    engine.dispose()
    return df

#############################
# 星等分佈 (1~5)
#############################
def fetch_star_distribution(board_filter=None):
    engine = get_engine()
    if board_filter and board_filter != "All":
        sql_title = f"""
        SELECT title_star_label AS star_label, COUNT(*) AS cnt
        FROM sentiments
        WHERE title_star_label IS NOT NULL AND board='{board_filter}'
        GROUP BY title_star_label
        """
        sql_content = f"""
        SELECT content_star_label AS star_label, COUNT(*) AS cnt
        FROM sentiments
        WHERE content_star_label IS NOT NULL AND board='{board_filter}'
        GROUP BY content_star_label
        """
        sql_push = f"""
        SELECT push_star_label AS star_label, COUNT(*) AS cnt
        FROM push_comments
        WHERE push_star_label IS NOT NULL
          AND article_id IN (
             SELECT id FROM sentiments WHERE board='{board_filter}'
          )
        GROUP BY push_star_label
        """
    else:
        sql_title = """
        SELECT title_star_label AS star_label, COUNT(*) AS cnt
        FROM sentiments
        WHERE title_star_label IS NOT NULL
        GROUP BY title_star_label
        """
        sql_content = """
        SELECT content_star_label AS star_label, COUNT(*) AS cnt
        FROM sentiments
        WHERE content_star_label IS NOT NULL
        GROUP BY content_star_label
        """
        sql_push = """
        SELECT push_star_label AS star_label, COUNT(*) AS cnt
        FROM push_comments
        WHERE push_star_label IS NOT NULL
        GROUP BY push_star_label
        """

    df_title = pd.read_sql_query(sql_title, engine)
    df_content = pd.read_sql_query(sql_content, engine)
    df_push = pd.read_sql_query(sql_push, engine)
    engine.dispose()

    df_title["star_int"] = df_title["star_label"].apply(star_label_to_int)
    df_content["star_int"] = df_content["star_label"].apply(star_label_to_int)
    df_push["star_int"] = df_push["star_label"].apply(star_label_to_int)

    return df_title, df_content, df_push

#############################
# 時間序列 (timestamp vs star_int)
#############################
def fetch_time_series(board_filter=None):
    engine = get_engine()
    if board_filter and board_filter != "All":
        sql_sent = f"""
        SELECT id, timestamp, board, title_star_label, content_star_label
        FROM sentiments
        WHERE board='{board_filter}'
        ORDER BY timestamp ASC
        """
    else:
        sql_sent = """
        SELECT id, timestamp, board, title_star_label, content_star_label
        FROM sentiments
        ORDER BY timestamp ASC
        """
    df_sent = pd.read_sql_query(sql_sent, engine)

    sql_push = """
    SELECT article_id, push_star_label
    FROM push_comments
    """
    df_push = pd.read_sql_query(sql_push, engine)
    engine.dispose()

    df_push["push_int"] = df_push["push_star_label"].apply(star_label_to_int)
    df_push_mean = df_push.groupby("article_id", as_index=False)["push_int"].mean().rename(columns={"push_int":"push_mean"})

    df_merged = pd.merge(df_sent, df_push_mean, left_on="id", right_on="article_id", how="left")
    df_merged["title_int"] = df_merged["title_star_label"].apply(star_label_to_int)
    df_merged["content_int"] = df_merged["content_star_label"].apply(star_label_to_int)

    df_merged = df_merged.dropna(subset=["title_int", "content_int", "push_mean"])
    return df_merged

#############################
# 統計分析: 取 sentiments & push 平均
#############################
def get_data_for_analysis(board_filter=None):
    engine = get_engine()
    if board_filter and board_filter != "All":
        sql_sent = f"""
        SELECT id, title_star_label, content_star_label, board
        FROM sentiments
        WHERE board = '{board_filter}'
        """
    else:
        sql_sent = """
        SELECT id, title_star_label, content_star_label, board
        FROM sentiments
        """
    df_sent = pd.read_sql_query(sql_sent, engine)

    sql_push = """
    SELECT article_id, push_star_label
    FROM push_comments
    """
    df_push = pd.read_sql_query(sql_push, engine)
    engine.dispose()

    df_sent['title_int'] = df_sent['title_star_label'].apply(star_label_to_int)
    df_sent['content_int'] = df_sent['content_star_label'].apply(star_label_to_int)

    df_push['push_int'] = df_push['push_star_label'].apply(star_label_to_int)
    df_push_mean = df_push.groupby('article_id', as_index=False)['push_int'].mean().rename(columns={'push_int':'push_mean'})

    df_all = pd.merge(df_sent, df_push_mean, left_on='id', right_on='article_id', how='left')
    df_all = df_all.dropna(subset=['title_int','content_int','push_mean'])
    return df_all

#############################
# Streamlit 主程式
#############################
st.set_page_config(page_title="PTT Dashboard (SQLite)", layout="wide")

# 移除 "文字雲" 選項
menu = st.sidebar.radio("功能選單", ["文章列表", "資料視覺化", "時間序列", "統計分析"], index=0)

if st.sidebar.button("刷新資料"):
    st.experimental_rerun()

if menu == "文章列表":
    st.title("PTT 文章列表（含情緒）")

    board_selection = st.sidebar.selectbox("篩選看板", ["All", "Gossiping", "NBA", "Stock"])
    engine = get_engine()
    if board_selection != "All":
        sql_count = f"SELECT COUNT(*) as cnt FROM sentiments WHERE board='{board_selection}'"
    else:
        sql_count = "SELECT COUNT(*) as cnt FROM sentiments"
    df_count = pd.read_sql_query(sql_count, engine)
    engine.dispose()
    total_articles = df_count["cnt"].iloc[0]

    page_size = 10
    total_pages = math.ceil(total_articles / page_size)

    current_page = st.sidebar.number_input("當前頁碼", min_value=1, max_value=max(1,total_pages), value=1, step=1)
    st.write(f"當前看板: {board_selection} | 共 {total_articles} 篇文章，每頁 {page_size} 篇，總頁數: {total_pages}。目前顯示第 {current_page} 頁。")

    offset = (current_page - 1) * page_size
    engine = get_engine()
    if board_selection != "All":
        query = f"""
        SELECT id, timestamp, board, title, content,
               title_star_label, content_star_label
        FROM sentiments
        WHERE board='{board_selection}'
        ORDER BY timestamp DESC
        LIMIT {page_size} OFFSET {offset}
        """
    else:
        query = f"""
        SELECT id, timestamp, board, title, content,
               title_star_label, content_star_label
        FROM sentiments
        ORDER BY timestamp DESC
        LIMIT {page_size} OFFSET {offset}
        """
    df_articles = pd.read_sql_query(query, engine)
    engine.dispose()

    article_ids = df_articles["id"].tolist()
    if article_ids:
        engine = get_engine()
        if len(article_ids) == 1:
            id_tuple = f"({article_ids[0]})"
        else:
            id_tuple = tuple(article_ids)
        q_push = f"""
        SELECT article_id, push_tag, push_userid, push_content,
               push_time, push_star_label
        FROM push_comments
        WHERE article_id IN {id_tuple}
        """
        df_push = pd.read_sql_query(q_push, engine)
        engine.dispose()
    else:
        df_push = pd.DataFrame()

    if df_articles.empty:
        st.write("無文章資料。")
    else:
        for idx, row in df_articles.iterrows():
            st.subheader(f"[{row['board']}] {row['title']}")
            st.write(f"文章ID: {row['id']} | 發文時間: {row['timestamp']}")

            if row.get("title_star_label"):
                star_html = color_star_label(row["title_star_label"])
                st.markdown(f"【標題星等】 {star_html}", unsafe_allow_html=True)
            if row.get("content_star_label"):
                star_html2 = color_star_label(row["content_star_label"])
                st.markdown(f"【內文星等】 {star_html2}", unsafe_allow_html=True)

            c = row["content"] if row["content"] else ""
            content_show = c[:1000] + ("..." if len(c) > 1000 else "")
            st.write("內文：")
            st.write(content_show)

            art_push = df_push[df_push["article_id"] == row["id"]]
            if art_push.empty:
                st.write("無推文")
            else:
                with st.expander(f"展開 {len(art_push)} 筆推文"):
                    for _, pushrow in art_push.iterrows():
                        push_text = f"{pushrow['push_tag']} {pushrow['push_userid']}: {pushrow['push_content']} ({pushrow['push_time']})"
                        if pushrow.get("push_star_label"):
                            star_html_p = color_star_label(pushrow["push_star_label"])
                            push_text += f" | {star_html_p}"
                        st.markdown(push_text, unsafe_allow_html=True)
            st.markdown("---")

elif menu == "資料視覺化":
    st.title("資料視覺化：星等分佈")
    st.write("改用星等(1~5) 來顯示分佈")

    board_selection = st.sidebar.selectbox("篩選看板", ["All", "Gossiping", "NBA", "Stock"])
    df_title, df_content, df_push = fetch_star_distribution(board_filter=board_selection)

    if df_title.empty:
        st.write("無標題星等資料")
    else:
        fig_title = px.bar(df_title, x="star_int", y="cnt", title="標題星等分佈", labels={"star_int":"星等", "cnt":"數量"})
        st.plotly_chart(fig_title, use_container_width=True)

    if df_content.empty:
        st.write("無內文星等資料")
    else:
        fig_content = px.bar(df_content, x="star_int", y="cnt", title="內文星等分佈", labels={"star_int":"星等", "cnt":"數量"})
        st.plotly_chart(fig_content, use_container_width=True)

    if df_push.empty:
        st.write("無推文星等資料")
    else:
        fig_push = px.bar(df_push, x="star_int", y="cnt", title="推文星等分佈", labels={"star_int":"星等", "cnt":"數量"})
        st.plotly_chart(fig_push, use_container_width=True)

elif menu == "時間序列":
    st.title("時間序列：情緒星等 (1~5) vs 時間")
    board_selection = st.sidebar.selectbox("篩選看板", ["All", "Gossiping", "NBA", "Stock"])
    df_time = fetch_time_series(board_filter=board_selection)
    if df_time.empty:
        st.write("無資料")
    else:
        df_long = pd.melt(
            df_time,
            id_vars=["id","timestamp","board","article_id"],
            value_vars=["title_int","content_int","push_mean"],
            var_name="type",
            value_name="star_value"
        )
        fig_ts = px.line(
            df_long,
            x="timestamp",
            y="star_value",
            color="type",
            title=f"{board_selection} - 時間序列情緒星等",
            labels={"timestamp":"時間", "star_value":"星等(1~5)", "type":"種類"}
        )
        st.plotly_chart(fig_ts, use_container_width=True)

else:  # "統計分析"
    st.title("統計分析 (星等)")
    st.write("此處示範：看板篩選、描述性統計、Pearson & Spearman 相關、多元迴歸，以及多看板差異檢定 (ANOVA)")

    board_options = ["All", "Gossiping", "NBA", "Stock"]
    board_choice = st.selectbox("選擇看板做分析", board_options, index=0)
    df_all = get_data_for_analysis(board_filter=board_choice)
    st.write(f"看板: {board_choice}, 有效文章數: {len(df_all)} (同時有標題星等、內文星等、推文平均星等)")

    if len(df_all) < 2:
        st.write("資料不足，無法進行統計分析。")
    else:
        st.subheader("描述性統計")
        desc_title = df_all['title_int'].describe()
        desc_content = df_all['content_int'].describe()
        desc_push = df_all['push_mean'].describe()

        st.write("標題星等：", desc_title.to_dict())
        st.write("內文星等：", desc_content.to_dict())
        st.write("推文平均星等：", desc_push.to_dict())

        st.subheader("相關分析 (Pearson, Spearman)")
        r_tc, p_tc = pearsonr(df_all['title_int'], df_all['content_int'])
        r_tp, p_tp = pearsonr(df_all['title_int'], df_all['push_mean'])
        r_cp, p_cp = pearsonr(df_all['content_int'], df_all['push_mean'])

        st.write(f"皮爾森 - 標題 vs 內文: r={r_tc:.3f}, p={p_tc:.4g}")
        st.write(f"皮爾森 - 標題 vs 推文: r={r_tp:.3f}, p={p_tp:.4g}")
        st.write(f"皮爾森 - 內文 vs 推文: r={r_cp:.3f}, p={p_cp:.4g}")

        r_tc_sp, p_tc_sp = spearmanr(df_all['title_int'], df_all['content_int'])
        r_tp_sp, p_tp_sp = spearmanr(df_all['title_int'], df_all['push_mean'])
        r_cp_sp, p_cp_sp = spearmanr(df_all['content_int'], df_all['push_mean'])

        st.write(f"斯皮爾曼 - 標題 vs 內文: r={r_tc_sp:.3f}, p={p_tc_sp:.4g}")
        st.write(f"斯皮爾曼 - 標題 vs 推文: r={r_tp_sp:.3f}, p={p_tp_sp:.4g}")
        st.write(f"斯皮爾曼 - 內文 vs 推文: r={r_cp_sp:.3f}, p={p_cp_sp:.4g}")

        st.write("若 p < 0.05，代表在統計上顯著 (樣本量大亦可使非常小的 r 也達顯著)")

        st.subheader("多元迴歸 (OLS)")
        X = df_all[['title_int', 'content_int']]
        X = sm.add_constant(X)
        y = df_all['push_mean']
        model = sm.OLS(y, X).fit()
        st.text(model.summary())

    st.subheader("多看板差異檢定 (ANOVA)")
    df_allboards = get_data_for_analysis(board_filter=None)
    if len(df_allboards) < 2 or df_allboards['board'].nunique() < 2:
        st.write("全看板資料不足或只有單一看板，無法做 ANOVA。")
    else:
        df_allboards = df_allboards.dropna(subset=["board"])
        formula = 'push_mean ~ C(board)'
        model_anova = ols(formula, data=df_allboards).fit()
        anova_table = sm.stats.anova_lm(model_anova, typ=2)
        st.write("以下比較各看板之推文平均星等是否有顯著差異 (ANOVA):")
        st.dataframe(anova_table)
        st.write("若 p < 0.05，表示至少有一個看板的平均分數與其他看板不同，建議進一步進行事後檢定。")
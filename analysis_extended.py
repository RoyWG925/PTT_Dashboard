import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.stats import pearsonr, spearmanr
import plotly.express as px
import statsmodels.api as sm

# ----------------------------
# PostgreSQL 連線參數 (請根據實際情況調整)
# ----------------------------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DBNAME = "ptt_db"
PG_USER = "ptt_user"
PG_PASSWORD = "ptt_password"

def get_engine():
    db_uri = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"
    engine = create_engine(db_uri)
    return engine

def star_label_to_int(star_label: str):
    """
    將 "1 star" ~ "5 stars" 轉為整數 1~5
    若無效或空值，回傳 None
    """
    if not star_label:
        return None
    try:
        return int(star_label[0])
    except:
        return None

def load_data():
    engine = get_engine()
    # 讀取 sentiments 表中需要的欄位
    sql_sent = """
    SELECT
      id,
      title_star_label,
      content_star_label
    FROM sentiments
    """
    df_sent = pd.read_sql_query(sql_sent, engine)
    
    # 讀取 push_comments 表
    sql_push = """
    SELECT
      article_id,
      push_star_label
    FROM push_comments
    """
    df_push = pd.read_sql_query(sql_push, engine)
    return df_sent, df_push

def prepare_data():
    """
    轉換 star_label 到數字，並合併推文 (取平均)
    """
    df_sent, df_push = load_data()

    # 將 sentiments 的欄位轉為整數
    df_sent['title_int'] = df_sent['title_star_label'].apply(star_label_to_int)
    df_sent['content_int'] = df_sent['content_star_label'].apply(star_label_to_int)
    
    # 將 push_comments 的推文欄位轉數字
    df_push['push_int'] = df_push['push_star_label'].apply(star_label_to_int)
    # 依 article_id 計算平均推文星等
    df_push_mean = df_push.groupby('article_id', as_index=False)['push_int'].mean().rename(columns={'push_int': 'push_mean'})
    
    # 合併 sentiments 與 push_mean（左連接）
    df_all = pd.merge(df_sent, df_push_mean, left_on='id', right_on='article_id', how='left')
    
    # 去除缺值（至少需要三個數值）
    df_all = df_all.dropna(subset=['title_int', 'content_int', 'push_mean'])
    return df_all

def descriptive_statistics(df):
    """
    計算標題、內文與推文平均星等的描述性統計數據
    """
    stats = {}
    stats['title'] = df['title_int'].describe()
    stats['content'] = df['content_int'].describe()
    stats['push'] = df['push_mean'].describe()
    return stats

def correlation_analysis(df):
    """
    分別計算皮爾森與斯皮爾曼相關
    """
    results = {}
    # 皮爾森相關
    r_tc, p_tc = pearsonr(df['title_int'], df['content_int'])
    r_tp, p_tp = pearsonr(df['title_int'], df['push_mean'])
    r_cp, p_cp = pearsonr(df['content_int'], df['push_mean'])
    
    results['pearson'] = {
        'title_vs_content': (r_tc, p_tc),
        'title_vs_push': (r_tp, p_tp),
        'content_vs_push': (r_cp, p_cp)
    }
    
    # 斯皮爾曼相關
    r_tc_sp, p_tc_sp = spearmanr(df['title_int'], df['content_int'])
    r_tp_sp, p_tp_sp = spearmanr(df['title_int'], df['push_mean'])
    r_cp_sp, p_cp_sp = spearmanr(df['content_int'], df['push_mean'])
    
    results['spearman'] = {
        'title_vs_content': (r_tc_sp, p_tc_sp),
        'title_vs_push': (r_tp_sp, p_tp_sp),
        'content_vs_push': (r_cp_sp, p_cp_sp)
    }
    
    return results

def regression_analysis(df):
    """
    以標題與內文星等作為自變數，推文平均星等作為因變數，
    建立多元線性迴歸模型，並回傳模型 summary
    """
    # 自變數：title_int 與 content_int
    X = df[['title_int', 'content_int']]
    # 加上常數項
    X = sm.add_constant(X)
    # 因變數：push_mean
    y = df['push_mean']
    model = sm.OLS(y, X).fit()
    return model.summary()

def scatter_plots(df):
    """
    利用 Plotly Express 畫散佈圖與回歸趨勢線 (trendline="ols")
    """
    figs = {}
    # 標題 vs 內文
    fig_tc = px.scatter(df, x='title_int', y='content_int', trendline="ols",
                        labels={'title_int':"標題星等", 'content_int':"內文星等"},
                        title="標題 vs 內文 星等散佈圖")
    figs['title_vs_content'] = fig_tc

    # 標題 vs 推文
    fig_tp = px.scatter(df, x='title_int', y='push_mean', trendline="ols",
                        labels={'title_int':"標題星等", 'push_mean':"推文平均星等"},
                        title="標題 vs 推文 星等散佈圖")
    figs['title_vs_push'] = fig_tp

    # 內文 vs 推文
    fig_cp = px.scatter(df, x='content_int', y='push_mean', trendline="ols",
                        labels={'content_int':"內文星等", 'push_mean':"推文平均星等"},
                        title="內文 vs 推文 星等散佈圖")
    figs['content_vs_push'] = fig_cp

    return figs

def main():
    # 準備資料
    df_all = prepare_data()
    print(f"有效文章數量: {len(df_all)}")
    
    # 描述性統計
    stats = descriptive_statistics(df_all)
    print("描述性統計：")
    print("標題星等：\n", stats['title'])
    print("內文星等：\n", stats['content'])
    print("推文平均星等：\n", stats['push'])
    
    # 相關分析
    corr_results = correlation_analysis(df_all)
    print("\n皮爾森相關:")
    print("標題 vs 內文: r = {:.3f}, p = {:.4g}".format(*corr_results['pearson']['title_vs_content']))
    print("標題 vs 推文: r = {:.3f}, p = {:.4g}".format(*corr_results['pearson']['title_vs_push']))
    print("內文 vs 推文: r = {:.3f}, p = {:.4g}".format(*corr_results['pearson']['content_vs_push']))
    
    print("\n斯皮爾曼相關:")
    print("標題 vs 內文: r = {:.3f}, p = {:.4g}".format(*corr_results['spearman']['title_vs_content']))
    print("標題 vs 推文: r = {:.3f}, p = {:.4g}".format(*corr_results['spearman']['title_vs_push']))
    print("內文 vs 推文: r = {:.3f}, p = {:.4g}".format(*corr_results['spearman']['content_vs_push']))
    
    # 多元迴歸分析
    reg_summary = regression_analysis(df_all)
    print("\n多元迴歸分析 (以標題與內文預測推文平均星等)：")
    print(reg_summary)
    
    # 利用 Plotly 畫散佈圖與趨勢線
    figs = scatter_plots(df_all)
    
    # 使用 Streamlit 顯示結果
    st.title("PTT 文章與推文情緒數值統計分析")
    st.write(f"有效文章數量: {len(df_all)}")
    
    st.subheader("描述性統計")
    st.write("標題星等描述：", stats['title'].to_dict())
    st.write("內文星等描述：", stats['content'].to_dict())
    st.write("推文平均星等描述：", stats['push'].to_dict())
    
    st.subheader("相關分析")
    st.write("皮爾森相關：")
    st.write(f"標題 vs 內文: r = {corr_results['pearson']['title_vs_content'][0]:.3f}, p = {corr_results['pearson']['title_vs_content'][1]:.4g}")
    st.write(f"標題 vs 推文: r = {corr_results['pearson']['title_vs_push'][0]:.3f}, p = {corr_results['pearson']['title_vs_push'][1]:.4g}")
    st.write(f"內文 vs 推文: r = {corr_results['pearson']['content_vs_push'][0]:.3f}, p = {corr_results['pearson']['content_vs_push'][1]:.4g}")
    
    st.write("斯皮爾曼相關：")
    st.write(f"標題 vs 內文: r = {corr_results['spearman']['title_vs_content'][0]:.3f}, p = {corr_results['spearman']['title_vs_content'][1]:.4g}")
    st.write(f"標題 vs 推文: r = {corr_results['spearman']['title_vs_push'][0]:.3f}, p = {corr_results['spearman']['title_vs_push'][1]:.4g}")
    st.write(f"內文 vs 推文: r = {corr_results['spearman']['content_vs_push'][0]:.3f}, p = {corr_results['spearman']['content_vs_push'][1]:.4g}")
    
    st.subheader("多元迴歸分析")
    st.text(reg_summary.as_text())
    
    st.subheader("散佈圖與趨勢線")
    st.plotly_chart(figs['title_vs_content'], use_container_width=True)
    st.plotly_chart(figs['title_vs_push'], use_container_width=True)
    st.plotly_chart(figs['content_vs_push'], use_container_width=True)

if __name__ == "__main__":
    main()

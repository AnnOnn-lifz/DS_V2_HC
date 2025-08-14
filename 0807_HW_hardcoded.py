
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm, font_manager

# ---------------------------
# 기본 설정 & 한글 폰트 처리
# ---------------------------
st.set_page_config(page_title="2023년 매출 대시보드 (하드코딩 버전)", layout="wide")

KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"]
def set_korean_font():
    available = set(f.name for f in font_manager.fontManager.ttflist)
    for f in KOREAN_FONTS:
        if f in available:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

TITLE = "2023년 매출 관련 세부 내역 (하드코딩 데이터 사용)"

# ---------------------------
# 유틸 함수
# ---------------------------
def to_datetime_series(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return s

def safe_col(df, name_candidates):
    for c in name_candidates:
        if c in df.columns:
            return c
    return None

def annotate_bar_values(ax, bars, values, fontsize=9):
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{int(v)}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )

def percent_labels_outside_pie(ax, values, labels, colors):
    wedges, _ = ax.pie(values, startangle=90, colors=colors, labels=[None]*len(values))
    total = float(np.sum(values))
    for w, val, lab in zip(wedges, values, labels):
        theta = (w.theta2 + w.theta1) / 2.0
        ang = np.deg2rad(theta)
        x, y = np.cos(ang), np.sin(ang)
        r_label = 1.25
        lx, ly = r_label * np.cos(ang), r_label * np.sin(ang)

        ha = "left" if lx >= 0 else "right"
        pct = 100.0 * val / total
        txt = f"{pct:.1f}%"

        ax.annotate(
            txt,
            xy=(x, y), xycoords="data",
            xytext=(lx, ly), textcoords="data",
            fontsize=10, fontweight="bold",
            ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", shrinkA=0, shrinkB=0, lw=1),
        )
        voffset_pt = 4 if ly >= 0 else -4
        ax.text(lx, ly + (voffset_pt/72), lab, fontsize=9, ha=ha,
                va="bottom" if voffset_pt > 0 else "top")

def pareto_chart(ax, ax2, df, cat_col, value_col):
    d = df.sort_values(by=value_col, ascending=False).reset_index(drop=True)
    d["누적비율"] = d[value_col].cumsum() / d[value_col].sum() * 100.0

    ax.bar(d[cat_col], d[value_col], width=0.5, color="#C8D7C4")
    ax2.plot(d[cat_col], d["누적비율"], marker="o", linewidth=2.5, color="#333f4d")

    for x, y in zip(d[cat_col], d["누적비율"]):
        ax2.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    ax.set_xlabel(cat_col)
    ax.set_ylabel(value_col)
    ax2.set_ylabel("누적 기여도(%)")
    ax.set_title("부서별 매출 및 누적 기여도", fontsize=14)

def bubble_sizes(series, min_size=80, max_size=780):
    s = series.astype(float)
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s_norm * (max_size - min_size) + min_size

# ---------------------------
# 데이터: 하드코딩 모듈에서 로드
# ---------------------------
try:
    from data_hardcoded import (
        BARCHART_HISTOGRAM as _BAR,
        TIME_SERIES as _TIME,
        PIE_CHART as _PIE,
        SCATTER as _SCAT,
        PARETO as _PAR,
        BUBBLE as _BUB,
    )
except Exception as e:
    st.error("data_hardcoded.py 를 찾을 수 없습니다. 같은 디렉토리에 위치시켜 주세요.")
    st.stop()

# 리스트[dict] -> DataFrame
df_bar = pd.DataFrame(_BAR)
df_time = pd.DataFrame(_TIME)
df_pie = pd.DataFrame(_PIE)
df_scatter = pd.DataFrame(_SCAT)
df_pareto = pd.DataFrame(_PAR)
df_bubble = pd.DataFrame(_BUB)

# ---------------------------
# 본문
# ---------------------------
st.title(TITLE)

with st.expander("원본 데이터 미리보기", expanded=False):
    st.write("바차트_히스토그램", df_bar.head(10))
    st.write("시계열차트", df_time.head(10))
    st.write("파이차트", df_pie.head(10))
    st.write("산점도", df_scatter.head(10))
    st.write("파레토차트", df_pareto.head(10))
    st.write("버블차트", df_bubble.head(10))

# ---------------------------
# 상단: 매출 개요 (막대 + 파이)
# ---------------------------
top_left, top_mid, top_right = st.columns([1.2, 0.05, 0.8])

with top_left:
    st.subheader("월별 총 매출")
    col_date = safe_col(df_bar, ["월", "날짜", "일자"])
    col_value = safe_col(df_bar, ["총 매출", "매출", "값", "Value"])
    if (col_date is None) or (col_value is None):
        st.error("막대 차트 데이터에 '월'/'총 매출' 컬럼이 필요합니다.")
    else:
        x = to_datetime_series(df_bar[col_date])
        y = df_bar[col_value].astype(float)

        fig, ax = plt.subplots(figsize=(8, 4))
        if pd.api.types.is_datetime64_any_dtype(x):
            bars = ax.bar(x, y, width=np.timedelta64(20, "D"), color="#BBCBD2")
        else:
            bars = ax.bar(x, y, color="#BBCBD2")

        ax.set_xlabel(col_date)
        ax.set_ylabel(col_value)
        ax.tick_params(axis="x", rotation=45)
        annotate_bar_values(ax, bars, y)

        st.pyplot(fig, clear_figure=True)

with top_right:
    st.subheader("1분기 제품별 매출 비중")
    col_label = df_pie.columns[0]
    col_val = safe_col(df_pie, ["1분기 매출", "매출", "값", "Value"]) or df_pie.columns[1]
    values = df_pie[col_val].astype(float).values
    labels = df_pie[col_label].astype(str).values

    fig, ax = plt.subplots(figsize=(6, 4))
    pastel_colors = [
        (0.6, 0.7, 0.8),
        (0.7, 0.8, 0.7),
        (0.8, 0.7, 0.7),
        (0.8, 0.8, 0.6),
        (0.7, 0.7, 0.8),
    ][: len(values)]
    percent_labels_outside_pie(ax, values, labels, pastel_colors)
    ax.set_title("1분기 제품별 매출 비중", fontsize=14)
    st.pyplot(fig, clear_figure=True)

# ---------------------------
# 중단: 제품별 분석 (다중 시계열 + 산점도)
# ---------------------------
mid_left, mid_right = st.columns([1.6, 1.0])

with mid_left:
    st.subheader("제품별 월별 매출 추이 (저채도)")
    time_col = safe_col(df_time, ["월", "날짜", "일자"])
    series_cols = [c for c in df_time.columns if c != time_col]
    if (time_col is None) or (len(series_cols) == 0):
        st.error("시계열 데이터에 날짜 컬럼과 1개 이상의 시계열 컬럼이 필요합니다.")
    else:
        tx = to_datetime_series(df_time[time_col])
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = cm.Set2.colors
        for i, col in enumerate(series_cols):
            y = df_time[col].astype(float)
            c = colors[i % len(colors)]
            ax.plot(tx, y, marker="o", linewidth=2.5, label=col, color=c)
            for xx, yy in zip(tx, y):
                ax.text(xx, yy, f"{yy}", fontsize=9, fontweight="bold", ha="center", va="bottom")
        ax.set_xlabel(time_col)
        ax.set_ylabel("매출")
        ax.legend(loc="upper left", ncol=3, frameon=False)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, clear_figure=True)

with mid_right:
    st.subheader("제품 A 매출 vs 비용")
    xcol = safe_col(df_scatter, ["제품 A 매출", "제품A매출", "X", "x"])
    ycol = safe_col(df_scatter, ["비용", "Y", "y"])
    if (xcol is None) or (ycol is None):
        st.error("산점도 데이터에 '제품 A 매출'과 '비용' 컬럼이 필요합니다.")
    else:
        x = df_scatter[xcol].astype(float)
        y = df_scatter[ycol].astype(float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, marker="D", facecolors="#333f4d", edgecolors="#333f4d")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        st.pyplot(fig, clear_figure=True)

# ---------------------------
# 하단: 부서/고객 분석 (파레토 + 버블)
# ---------------------------
bot_left, bot_right = st.columns([1.6, 1.0])

with bot_left:
    st.subheader("파레토 분석")
    cat_col = safe_col(df_pareto, ["부서", "카테고리", "구분"])
    val_col = safe_col(df_pareto, ["매출", "값", "Value"])
    if (cat_col is None) or (val_col is None):
        st.error("파레토 데이터에 '부서'와 '매출' 컬럼이 필요합니다.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax2 = ax.twinx()
        pareto_chart(ax, ax2, df_pareto, cat_col, val_col)
        st.pyplot(fig, clear_figure=True)

with bot_right:
    st.subheader("버블 차트")
    p_col = safe_col(df_bubble, ["제품", "항목", "label", "Label"]) or df_bubble.columns[0]
    x_col = safe_col(df_bubble, ["제품별 비용", "비용", "X"]) or df_bubble.columns[1]
    y_col = safe_col(df_bubble, ["마진", "Y"]) or df_bubble.columns[2]
    s_col = safe_col(df_bubble, ["고객 수", "고객수", "Size", "size"]) or df_bubble.columns[3]
    try:
        X = df_bubble[x_col].astype(float)
        Y = df_bubble[y_col].astype(float)
        S = bubble_sizes(df_bubble[s_col])
        labels = df_bubble[p_col].astype(str).tolist()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            X, Y,
            s=S,
            facecolors=(0, 0, 0, 0),
            edgecolors="#333f4d",
            marker="o"
        )
        for i, txt in enumerate(labels):
            ax.annotate(txt, (X.iloc[i], Y.iloc[i]), fontsize=9, ha="center", va="bottom")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"버블 차트 데이터 형식 오류: {e}")

st.caption("© 2025. 스트림릿 대시보드 데모 — 하드코딩 데이터 버전.")

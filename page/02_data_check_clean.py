import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

DATA_PATH = "ES_Pre.csv"

NUM_COLS = ["age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]

st.set_page_config(page_title="02 데이터 정리", layout="wide")
st.title("02) 데이터 정리(결측치/이상치/형식 통일)")

@st.cache_data(show_spinner=False)
def load_raw():
    return pd.read_csv(DATA_PATH)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 숫자형 변환
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 문자열 정리(소문자, 공백 제거)
    for c in df.columns:
        if c not in NUM_COLS:
            df[c] = df[c].astype(str).str.strip()

    # 인터넷 yes/no 정리
    if "internet_access" in df.columns:
        df["internet_access"] = df["internet_access"].str.lower().replace({
            "y": "yes", "n": "no", "true": "yes", "false": "no"
        })

    # 수면질 정리
    if "sleep_quality" in df.columns:
        df["sleep_quality"] = df["sleep_quality"].str.lower()

    # 점수 범위 이상치: 0~100 밖은 NaN 처리(원하면 제거 가능)
    if "exam_score" in df.columns:
        df.loc[(df["exam_score"] < 0) | (df["exam_score"] > 100), "exam_score"] = np.nan

    # 공부시간/수면시간 음수 이상치 처리
    for c in ["study_hours", "sleep_hours"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # 출석률 범위
    if "class_attendance" in df.columns:
        df.loc[(df["class_attendance"] < 0) | (df["class_attendance"] > 100), "class_attendance"] = np.nan

    return df

raw = load_raw()
df = clean_df(raw)

st.subheader("결측치 현황")
na = df.isna().sum().sort_values(ascending=False).reset_index()
na.columns = ["열", "결측치 수"]
st.dataframe(na, use_container_width=True)

st.subheader("이상치/범위 점검(주요 수치형)")
for c in [x for x in NUM_COLS if x in df.columns]:
    st.write(f"**{c}**")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(df[c].describe())
    with col2:
        fig = px.box(df, y=c, points="outliers", title=f"{c} 분포(이상치 포함)")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("정리된 데이터 미리보기")
st.dataframe(df.head(15), use_container_width=True)

st.info("이 페이지의 정리 로직은 분석/예측 페이지에서도 동일하게 적용됩니다.")

import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = "ES_Pre.csv"
NUM_COLS = ["age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]

st.set_page_config(page_title="03 분석(상관관계)", layout="wide")
st.title("03) 분석: 전체 + 과목별 상관관계")

@st.cache_data(show_spinner=False)
def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 범위 정리
    if "exam_score" in df.columns:
        df.loc[(df["exam_score"] < 0) | (df["exam_score"] > 100), "exam_score"] = np.nan
    if "class_attendance" in df.columns:
        df.loc[(df["class_attendance"] < 0) | (df["class_attendance"] > 100), "class_attendance"] = np.nan
    for c in ["study_hours", "sleep_hours"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan
    # 문자열
    for c in df.columns:
        if c not in NUM_COLS:
            df[c] = df[c].astype(str).str.strip()
    return df

df = load_and_clean()

# --- 1) 전체 상관(공부시간-점수)
st.subheader("1) 전체: 공부시간 ↔ 시험점수 상관")
d = df[["study_hours", "exam_score"]].dropna()
if len(d) < 3:
    st.warning("분석에 필요한 데이터가 부족합니다.")
else:
    r, p = stats.pearsonr(d["study_hours"], d["exam_score"])
    reg = stats.linregress(d["study_hours"], d["exam_score"])

    c1, c2, c3 = st.columns(3)
    c1.metric("상관계수(r)", f"{r:.3f}")
    c2.metric("p값", f"{p:.3f}")
    c3.metric("예상 증가폭", f"+{reg.slope:.2f}점 / 1시간")

    fig = px.scatter(d, x="study_hours", y="exam_score", opacity=0.35, title="전체: 공부시간 vs 시험점수")
    xs = np.linspace(d["study_hours"].min(), d["study_hours"].max(), 200)
    ys = reg.intercept + reg.slope * xs
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="추세선"))
    fig.update_layout(xaxis_title="공부시간(시간)", yaxis_title="시험점수")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 2) 과목별 상관(공부시간-점수)
st.subheader("2) 과목별: 공부시간 ↔ 시험점수 상관(테이블)")
if "course" not in df.columns:
    st.warning("course 열이 없습니다.")
else:
    rows = []
    for course, g in df.groupby("course"):
        gg = g[["study_hours", "exam_score"]].dropna()
        n = len(gg)
        if n >= 10:  # 표본 너무 적으면 r가 불안정해서 최소 10
            r, p = stats.pearsonr(gg["study_hours"], gg["exam_score"])
            rows.append({"과목": course, "표본수(n)": n, "상관계수(r)": r, "p값": p})
        else:
            rows.append({"과목": course, "표본수(n)": n, "상관계수(r)": np.nan, "p값": np.nan})

    corr_table = pd.DataFrame(rows).sort_values("상관계수(r)", ascending=False)
    st.dataframe(corr_table, use_container_width=True)

    st.caption("※ 표본수가 너무 적은 과목은 상관계수를 표시하지 않았습니다(불안정).")

st.divider()

# --- 3) 과목 선택 후 상세 그래프(공부시간-점수 + 회귀선)
st.subheader("3) 과목을 선택해서 더 자세히 보기")
if "course" in df.columns:
    course_list = sorted(df["course"].dropna().unique().tolist())
    picked = st.selectbox("과목 선택", course_list, index=0)

    g = df[df["course"] == picked][["study_hours", "exam_score"]].dropna()
    if len(g) < 3:
        st.warning("이 과목은 데이터가 부족합니다.")
    else:
        r, p = stats.pearsonr(g["study_hours"], g["exam_score"])
        reg = stats.linregress(g["study_hours"], g["exam_score"])
        st.write(f"- 표본수: **{len(g)}**, r: **{r:.3f}**, p: **{p:.3f}**, 증가폭: **+{reg.slope:.2f}점/시간**")

        fig = px.scatter(g, x="study_hours", y="exam_score", opacity=0.45, title=f"{picked}: 공부시간 vs 시험점수")
        xs = np.linspace(g["study_hours"].min(), g["study_hours"].max(), 200)
        ys = reg.intercept + reg.slope * xs
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="추세선"))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 4) (보너스) 수면시간-점수, 출석-점수도 과목별로 볼 수 있게
st.subheader("4) (선택) 다른 변수도 과목별로 확인")
var = st.selectbox("점수와 비교할 변수 선택", ["sleep_hours", "class_attendance", "age"], index=0)
if ("course" in df.columns) and (var in df.columns):
    rows = []
    for course, g in df.groupby("course"):
        gg = g[[var, "exam_score"]].dropna()
        n = len(gg)
        if n >= 10:
            r, p = stats.pearsonr(gg[var], gg["exam_score"])
            rows.append({"과목": course, "표본수(n)": n, f"{var}-점수 r": r, "p값": p})
        else:
            rows.append({"과목": course, "표본수(n)": n, f"{var}-점수 r": np.nan, "p값": np.nan})
    t = pd.DataFrame(rows).sort_values(f"{var}-점수 r", ascending=False)
    st.dataframe(t, use_container_width=True)

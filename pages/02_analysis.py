import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

DATA_PATH = "ES_Pre.csv"
GENDER_COLOR = {"male": "#1f77b4", "female": "#ff7f0e", "other": "#2ca02c"}

st.set_page_config(page_title="데이터 분석", layout="wide")
st.title("02) 데이터 분석")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    # 수치형 캐스팅(안전)
    for c in ["study_hours", "exam_score", "sleep_hours", "class_attendance", "age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

# 1) 공부시간 vs 시험점수
st.subheader("1) 공부시간이 늘면 시험점수가 올라가나요? (상관관계)")
d = df[["study_hours", "exam_score"]].dropna()
if len(d) < 3:
    st.warning("분석에 필요한 데이터가 충분하지 않습니다.")
else:
    x = d["study_hours"].values
    y = d["exam_score"].values
    r, p = stats.pearsonr(x, y)
    reg = stats.linregress(x, y)

    c1, c2, c3 = st.columns(3)
    c1.metric("상관계수(r)", f"{r:.3f}")
    c2.metric("p값", f"{p:.3f}")
    c3.metric("대략 증가폭", f"+{reg.slope:.2f}점 / 1시간")

    fig = px.scatter(d, x="study_hours", y="exam_score", opacity=0.35, title="공부시간 vs 시험점수 (추세선 포함)")
    xs = np.linspace(x.min(), x.max(), 200)
    ys = reg.intercept + reg.slope * xs
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="추세선"))
    fig.update_layout(xaxis_title="공부시간(시간)", yaxis_title="시험점수")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 2) 성별 평균 차이
st.subheader("2) 성별에 따라 시험점수 평균이 다를까요?")
if "gender" not in df.columns or "exam_score" not in df.columns:
    st.warning("gender 또는 exam_score 열이 없습니다.")
else:
    d = df[df["gender"].isin(["male", "female"])][["gender", "exam_score"]].dropna()
    if len(d) < 3:
        st.warning("분석에 필요한 데이터가 충분하지 않습니다.")
    else:
        summary = d.groupby("gender")["exam_score"].agg(["count", "mean", "std"]).reset_index()
        summary["SE"] = summary["std"] / np.sqrt(summary["count"])
        st.dataframe(summary.rename(columns={"gender": "성별", "count": "사람수", "mean": "평균", "std": "표준편차"}), use_container_width=True)

        a = d[d["gender"] == "female"]["exam_score"]
        b = d[d["gender"] == "male"]["exam_score"]
        t = stats.ttest_ind(a, b, equal_var=False)
        st.write(f"- 평균 차이(여-남): **{(a.mean()-b.mean()):.2f}점**, p값: **{t.pvalue:.3f}**")

        fig1 = px.box(d, x="gender", y="exam_score", color="gender", points="outliers",
                      color_discrete_map=GENDER_COLOR, title="성별 점수 분포(박스플롯)")
        fig1.update_layout(xaxis_title="성별", yaxis_title="시험점수")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(summary, x="gender", y="mean", error_y="SE", color="gender",
                      color_discrete_map=GENDER_COLOR, title="성별 평균(±표준오차)")
        fig2.update_layout(xaxis_title="성별", yaxis_title="평균 시험점수")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# 3) 인터넷 이용 vs 수면의 질(좋음/나쁨 비율)
st.subheader("3) 인터넷 이용 여부에 따라 수면의 질(좋음/나쁨) 비율이 다를까요?")
need = {"internet_access", "sleep_quality"}
if not need.issubset(set(df.columns)):
    st.warning("internet_access 또는 sleep_quality 열이 없습니다.")
else:
    d = df[list(need)].dropna()
    d = d[d["sleep_quality"].isin(["good", "poor"])].copy()
    if len(d) < 3:
        st.warning("good/poor 데이터가 충분하지 않습니다.")
    else:
        ct = pd.crosstab(d["internet_access"], d["sleep_quality"])
        ct = ct.rename(index={"yes": "인터넷 사용", "no": "인터넷 미사용"},
                       columns={"good": "좋음", "poor": "나쁨"})
        st.write("**사람 수(빈도)**")
        st.dataframe(ct, use_container_width=True)

        pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
        st.write("**비율(%)**")
        st.dataframe(pct, use_container_width=True)

        if ct.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
            st.write(f"- 카이제곱 검정 p값: **{p:.3f}**")

        long = pct.reset_index().melt(id_vars="internet_access", var_name="수면의 질", value_name="비율(%)")
        fig = px.bar(long, x="internet_access", y="비율(%)", color="수면의 질", barmode="stack",
                     title="인터넷 이용 여부에 따른 수면의 질 비율")
        fig.update_layout(xaxis_title="", yaxis_title="비율(%)")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 4) 공부 방식별 평균 비교(요청 사항)
st.subheader("4) 공부 방식에 따라 시험점수 평균이 다를까요?")
need = {"study_method", "exam_score"}
if not need.issubset(set(df.columns)):
    st.warning("study_method 또는 exam_score 열이 없습니다.")
else:
    d = df[list(need)].dropna()
    if len(d) < 3:
        st.warning("분석에 필요한 데이터가 충분하지 않습니다.")
    else:
        summary = (
            d.groupby("study_method")["exam_score"]
             .agg(["count", "mean", "std"])
             .reset_index()
             .rename(columns={"study_method": "공부 방식", "count": "사람수", "mean": "평균점수", "std": "표준편차"})
        )
        summary["표준오차(SE)"] = summary["표준편차"] / np.sqrt(summary["사람수"])
        summary = summary.sort_values("평균점수", ascending=False)

        st.write("**공부 방식별 평균 표**")
        st.dataframe(summary, use_container_width=True)

        fig = px.bar(summary, x="공부 방식", y="평균점수", error_y="표준오차(SE)",
                     title="공부 방식별 평균(±표준오차)")
        fig.update_layout(xaxis_title="공부 방식", yaxis_title="평균 시험점수")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(d, x="study_method", y="exam_score", points="outliers",
                      title="공부 방식별 점수 분포(박스플롯)")
        fig2.update_layout(xaxis_title="공부 방식", yaxis_title="시험점수")
        st.plotly_chart(fig2, use_container_width=True)

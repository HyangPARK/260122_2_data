import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Exam Score Analyzer", layout="wide")

EXPECTED_COLUMNS = [
    "student_id", "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty", "exam_score"
]

# ✅ 과목별 추천 동영상(고정 링크: 런타임 API 불필요)
# (YouTube는 streamlit에서 st.video로 바로 임베드 가능)
COURSE_VIDEOS = {
    "b.tech": [
        ("Active recall 기반 실전 공부법", "https://www.youtube.com/watch?v=5gP-RjHIsWM"),  # :contentReference[oaicite:0]{index=0}
        ("시험 대비 evidence-based masterclass", "https://www.youtube.com/watch?v=Lt54CX9DmS4"),  # :contentReference[oaicite:1]{index=1}
    ],
    "bca": [
        ("BCA C 프로그래밍 End-Sem 전략", "https://www.youtube.com/watch?v=eyn9Ye7Eeh0"),  # :contentReference[oaicite:2]{index=2}
        ("BCA study routine/시험 준비 루틴", "https://www.youtube.com/watch?v=t5wT55s2ogs"),  # :contentReference[oaicite:3]{index=3}
    ],
    "b.com": [
        ("Accounting 시험 공부법", "https://www.youtube.com/watch?v=c5lbguWGPzg"),  # :contentReference[oaicite:4]{index=4}
        ("Accountancy pass 전략", "https://www.youtube.com/watch?v=MTUHxz3XRus"),  # :contentReference[oaicite:5]{index=5}
    ],
    "ba": [
        ("Evidence-based 시험 공부법", "https://www.youtube.com/watch?v=ukLnPbIffxE"),  # :contentReference[oaicite:6]{index=6}
        ("에세이/서술형 답안 연습법(essay technique)", "https://www.youtube.com/watch?v=H2BY1_hSBG4"),  # :contentReference[oaicite:7]{index=7}
    ],
    "bba": [
        ("BBA 30일 플랜/전략", "https://www.youtube.com/watch?v=mEsGqrIP2L8"),  # :contentReference[oaicite:8]{index=8}
        ("B.com/BBA 공통 시험 전략", "https://www.youtube.com/watch?v=2zs3tpx11Sg"),  # :contentReference[oaicite:9]{index=9}
    ],
    "b.sc": [
        ("BSc exam preparation playlist", "https://www.youtube.com/playlist?list=PLvstRegfprDDGOwE1d0nV0Us8SvEQR26i"),  # :contentReference[oaicite:10]{index=10}
        ("Active recall(과학적 공부법) 설명", "https://www.youtube.com/watch?v=dUs9Tv3YG4A"),  # :contentReference[oaicite:11]{index=11}
    ],
    "diploma": [
        ("Diploma 시험 시간표/전략", "https://www.youtube.com/watch?v=JDyHOIHcb-8"),  # :contentReference[oaicite:12]{index=12}
        ("30일 플랜으로 diploma exams", "https://www.youtube.com/watch?v=KNBdubQUwuw"),  # :contentReference[oaicite:13]{index=13}
    ],
}

@st.cache_data(show_spinner=False)
def load_default_data():
    # ✅ 리포지토리에 Exam_Score_Prediction.csv 포함 시 기본 탑재 데이터로 로드
    return pd.read_csv("ES_Pre.csv")

def read_uploaded_csv(file) -> pd.DataFrame:
    content = file.read()
    return pd.read_csv(io.BytesIO(content))

def validate_schema(df: pd.DataFrame):
    cols = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in EXPECTED_COLUMNS]
    ok = (len(missing) == 0) and (len(extra) == 0)
    return ok, missing, extra

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["student_id", "age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["gender", "course", "internet_access", "sleep_quality",
              "study_method", "facility_rating", "exam_difficulty"]:
        df[c] = df[c].astype(str)
    return df

def data_quality(df: pd.DataFrame):
    st.subheader("데이터 점검 (결측치/기초 통계)")
    c1, c2 = st.columns(2)
    with c1:
        st.write({"rows": int(df.shape[0]), "cols": int(df.shape[1])})
        st.write("**결측치(열별)**")
        st.dataframe(df.isna().sum().sort_values(ascending=False))
    with c2:
        st.write("**수치형 요약**")
        st.dataframe(df[["age","study_hours","class_attendance","sleep_hours","exam_score"]].describe().T)

    st.write("**논리 범위 체크(위반 건수)**")
    checks = []
    def add_check(name, cond):
        checks.append({"check": name, "violations": int((~cond).sum())})
    add_check("class_attendance 0~100", df["class_attendance"].between(0, 100))
    add_check("exam_score 0~100", df["exam_score"].between(0, 100))
    add_check("study_hours > 0", df["study_hours"] > 0)
    add_check("sleep_hours 0~24", df["sleep_hours"].between(0, 24))
    st.dataframe(pd.DataFrame(checks))

def analysis_1_study_hours(df: pd.DataFrame):
    st.subheader("1) 공부시간(study_hours) ↔ 시험점수(exam_score)")
    d = df[["study_hours","exam_score"]].dropna()

    r, p = stats.pearsonr(d["study_hours"], d["exam_score"])
    reg = stats.linregress(d["study_hours"], d["exam_score"])
    slope, intercept, se = reg.slope, reg.intercept, reg.stderr
    ci_low, ci_high = slope - 1.96*se, slope + 1.96*se

    m1, m2, m3 = st.columns(3)
    m1.metric("상관계수 r", f"{r:.3f}")
    m2.metric("기울기(점/시간)", f"{slope:.2f}")
    m3.metric("p-value", f"{p:.3e}" if p > 0 else "< 1e-300")
    st.caption(f"단순회귀: exam_score = {intercept:.2f} + {slope:.2f} * study_hours (slope 95% CI: [{ci_low:.2f}, {ci_high:.2f}])")

    # ✅ 더 간단한 상관관계 시각화: ① scatter + 추세선, ② corr heatmap
    fig_scatter = px.scatter(
        d, x="study_hours", y="exam_score", opacity=0.35,
        title="Study Hours vs Exam Score (Scatter + Regression Line)"
    )
    xs = np.linspace(d["study_hours"].min(), d["study_hours"].max(), 200)
    ys = intercept + slope * xs
    fig_scatter.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Regression"))
    st.plotly_chart(fig_scatter, use_container_width=True)

    corr_df = df[["study_hours","class_attendance","sleep_hours","exam_score"]].corr(numeric_only=True)
    fig_heat = px.imshow(corr_df, text_auto=True, title="Correlation Heatmap (numeric variables)")
    st.plotly_chart(fig_heat, use_container_width=True)

def analysis_2_gender(df: pd.DataFrame):
    st.subheader("2) 성별(gender)별 시험점수 평균 차이")
    d = df[df["gender"].isin(["male","female"])][["gender","exam_score"]].dropna()

    summary = d.groupby("gender")["exam_score"].agg(["count","mean","std"]).reset_index()
    st.dataframe(summary)

    a = d[d["gender"]=="female"]["exam_score"]
    b = d[d["gender"]=="male"]["exam_score"]
    t = stats.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())

    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    d_cohen = diff / sp if sp > 0 else np.nan

    m1, m2, m3 = st.columns(3)
    m1.metric("평균차(여-남)", f"{diff:.3f}")
    m2.metric("p-value(Welch t)", f"{t.pvalue:.3f}")
    m3.metric("Cohen's d", f"{d_cohen:.3f}")

    fig_box = px.box(d, x="gender", y="exam_score", points="outliers",
                     title="Exam Score by Gender (Boxplot)")
    st.plotly_chart(fig_box, use_container_width=True)

def analysis_3_internet_sleep_quality(df: pd.DataFrame):
    st.subheader("3) 인터넷 이용(internet_access) → 수면의 질(sleep_quality) 영향/차이")
    d = df[["internet_access","sleep_quality"]].dropna()

    ct = pd.crosstab(d["internet_access"], d["sleep_quality"])
    st.write("**교차표(Counts)**")
    st.dataframe(ct)

    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    phi2 = chi2 / n
    r_dim, k_dim = ct.shape
    cramers_v = np.sqrt(phi2 / min(k_dim-1, r_dim-1)) if min(k_dim-1, r_dim-1) > 0 else np.nan

    m1, m2, m3 = st.columns(3)
    m1.metric("Chi-square", f"{chi2:.3f}")
    m2.metric("p-value", f"{p:.3f}")
    m3.metric("Cramér's V", f"{cramers_v:.3f}")

    # row% stacked bar
    row_pct = ct.div(ct.sum(axis=1), axis=0).reset_index()
    long = row_pct.melt(id_vars="internet_access", var_name="sleep_quality", value_name="proportion")
    long["proportion"] = long["proportion"] * 100

    fig = px.bar(long, x="internet_access", y="proportion", color="sleep_quality",
                 barmode="stack",
                 title="Sleep Quality Distribution by Internet Access (Row %)",
                 labels={"proportion":"Percent (%)"})
    st.plotly_chart(fig, use_container_width=True)

def analysis_4_course_strategy(df: pd.DataFrame):
    st.subheader("4) 과목별(score 향상 전략 + 추천 영상)")
    courses = sorted(df["course"].dropna().unique().tolist())
    course = st.selectbox("과목 선택(course)", courses)

    sub = df[df["course"] == course].dropna(subset=["exam_score","study_hours","class_attendance","sleep_hours"])

    # ✅ 과목별 '데이터 기반' 영향 신호(상관): study_hours/attendance/sleep_hours
    corrs = {}
    for v in ["study_hours","class_attendance","sleep_hours"]:
        corrs[v] = stats.pearsonr(sub[v], sub["exam_score"])[0]

    corr_table = (pd.DataFrame({"variable": list(corrs.keys()), "pearson_r": list(corrs.values())})
                  .sort_values("pearson_r", ascending=False))
    st.write("**이 과목에서 점수와 더 같이 움직이는 변수(상관 r)**")
    st.dataframe(corr_table)

    # ✅ 간단 전략(데이터 기반: 상관 큰 순서로 제시)
    top = corr_table.iloc[0]["variable"]
    st.markdown("**권장 전략(이 데이터셋 기준, 상관이 큰 요인 중심)**")
    bullets = []
    if top == "study_hours":
        bullets.append("- 공부시간을 ‘늘리는 것’ 자체가 점수와 가장 강하게 함께 움직입니다. (예: 주간 계획 + 반복/회상 기반 학습)")
    if "class_attendance" in corr_table["variable"].values:
        bullets.append("- 출석률도 중간 정도의 양의 상관이 있어, 결석/지각을 줄이는 쪽이 유리합니다.")
    if "sleep_hours" in corr_table["variable"].values:
        bullets.append("- 수면시간은 약한~중간의 양의 상관이어서, 과도한 수면 부족이 누적되지 않도록 관리가 필요합니다.")
    st.write("\n".join(bullets))

    # ✅ 추천 영상 임베드
    st.markdown("**추천 영상(과목/전략 관련)**")
    vids = COURSE_VIDEOS.get(course, [])
    if not vids:
        st.info("이 과목에 대해 등록된 추천 영상이 없습니다.")
    else:
        for title, url in vids:
            st.write(f"- {title}")
            st.video(url)

def main():
    st.title("Exam Score Analyzer (기본 탑재 + 업로드 자동 분석)")

    st.sidebar.header("데이터 선택")
    uploaded_files = st.sidebar.file_uploader(
        "같은 형식의 CSV 업로드 시 업로드 데이터로 자동 분석(복수 업로드 가능)",
        type=["csv"], accept_multiple_files=True
    )

    # 기본: 탑재 데이터
    df = load_default_data()

    # 업로드 있으면 스키마 확인 후 concat
    if uploaded_files and len(uploaded_files) > 0:
        dfs = []
        errors = []
        for f in uploaded_files:
            try:
                tmp = read_uploaded_csv(f)
                ok, missing, extra = validate_schema(tmp)
                if not ok:
                    errors.append((f.name, missing, extra))
                else:
                    dfs.append(tmp)
            except Exception as e:
                errors.append((f.name, ["(read error)"], [str(e)]))

        if errors:
            st.sidebar.error("형식이 맞지 않거나 읽기 실패한 파일이 있습니다.")
            for name, missing, extra in errors:
                st.sidebar.write(f"- {name}")
                st.sidebar.write(f"  missing: {missing}")
                st.sidebar.write(f"  extra/error: {extra}")

        if len(dfs) > 0:
            df = pd.concat(dfs, ignore_index=True)
            st.sidebar.success(f"업로드 데이터 사용: {len(dfs)}개 파일, {df.shape[0]}행")
        else:
            st.sidebar.info("유효 업로드가 없어 기본 탑재 데이터를 사용합니다.")
    else:
        st.sidebar.info("기본 탑재 데이터 사용 중 (Exam_Score_Prediction.csv)")

    # 열 순서가 다르면 재정렬
    if set(df.columns) == set(EXPECTED_COLUMNS):
        df = df[EXPECTED_COLUMNS]
    df = coerce_types(df)

    tabs = st.tabs(["데이터 점검", "1) 공부시간↔점수", "2) 성별 차이", "3) 인터넷↔수면의질", "4) 과목별 전략/영상"])

    with tabs[0]:
        data_quality(df)

    with tabs[1]:
        analysis_1_study_hours(df)

    with tabs[2]:
        analysis_2_gender(df)

    with tabs[3]:
        analysis_3_internet_sleep_quality(df)

    with tabs[4]:
        analysis_4_course_strategy(df)

if __name__ == "__main__":
    main()

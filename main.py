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

@st.cache_data(show_spinner=False)
def load_default_data():
    # ✅ 리포지토리에 Exam_Score_Prediction.csv 를 함께 올려두면 기본 탑재 데이터로 사용
    return pd.read_csv("Exam_Score_Prediction.csv")

def read_uploaded_csv(file) -> pd.DataFrame:
    # Streamlit UploadedFile -> bytes -> pandas
    content = file.read()
    return pd.read_csv(io.BytesIO(content))

def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
    cols = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in EXPECTED_COLUMNS]
    ok = (len(missing) == 0) and (len(extra) == 0)
    return ok, missing, extra

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # numeric
    for c in ["student_id", "age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # categorical/string
    for c in ["gender", "course", "internet_access", "sleep_quality",
              "study_method", "facility_rating", "exam_difficulty"]:
        df[c] = df[c].astype(str)

    return df

def basic_quality_checks(df: pd.DataFrame):
    st.subheader("데이터 품질 점검 (결측치/기초 통계)")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**행/열 크기**")
        st.write({"rows": int(df.shape[0]), "cols": int(df.shape[1])})

        st.write("**결측치 개수(열별)**")
        na = df.isna().sum().sort_values(ascending=False)
        st.dataframe(na)

    with col2:
        st.write("**수치형 변수 요약 통계**")
        num_cols = ["age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]
        st.dataframe(df[num_cols].describe().T)

    # 간단한 범위 체크(논리적 범위)
    st.write("**논리적 범위 체크(참고)**")
    checks = []
    def add_check(name, cond):
        checks.append({"check": name, "violations": int((~cond).sum())})

    add_check("class_attendance between 0 and 100", df["class_attendance"].between(0, 100))
    add_check("exam_score between 0 and 100", df["exam_score"].between(0, 100))
    add_check("study_hours > 0", df["study_hours"] > 0)
    add_check("sleep_hours between 0 and 24", df["sleep_hours"].between(0, 24))
    st.dataframe(pd.DataFrame(checks))

def analysis_study_hours_vs_score(df: pd.DataFrame):
    st.subheader("1) 공부시간(study_hours)과 시험점수(exam_score) 관계")

    clean = df[["study_hours", "exam_score"]].dropna()
    x = clean["study_hours"].astype(float).values
    y = clean["exam_score"].astype(float).values

    r, p = stats.pearsonr(x, y)
    reg = stats.linregress(x, y)
    slope, intercept, stderr = reg.slope, reg.intercept, reg.stderr
    ci_low, ci_high = slope - 1.96 * stderr, slope + 1.96 * stderr

    c1, c2, c3 = st.columns(3)
    c1.metric("Pearson r", f"{r:.3f}")
    c2.metric("회귀 기울기(점/시간)", f"{slope:.3f}")
    c3.metric("p-value", f"{p:.3e}" if p > 0 else "< 1e-300")

    st.caption(f"단순회귀: exam_score = {intercept:.3f} + {slope:.3f} * study_hours (slope 95% CI: [{ci_low:.3f}, {ci_high:.3f}])")

    # Plotly scatter + regression line
    fig = px.scatter(clean, x="study_hours", y="exam_score", opacity=0.35,
                     title="Study Hours vs Exam Score (Scatter + Regression Line)",
                     labels={"study_hours": "Study Hours", "exam_score": "Exam Score"})

    xs = np.linspace(x.min(), x.max(), 200)
    ys = intercept + slope * xs
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Regression line"))

    st.plotly_chart(fig, use_container_width=True)

def analysis_gender_mean_diff(df: pd.DataFrame):
    st.subheader("2) 성별(gender)에 따른 시험점수 평균 차이")

    option = st.radio("표시할 성별 범위", ["male vs female(비교 중심)", "전체 포함(male/female/other)"], horizontal=True)

    if option.startswith("male vs female"):
        d = df[df["gender"].isin(["male", "female"])][["gender", "exam_score"]].dropna()
    else:
        d = df[["gender", "exam_score"]].dropna()

    # Summary table
    summary = d.groupby("gender")["exam_score"].agg(["count", "mean", "std"]).reset_index()
    st.dataframe(summary)

    # Statistical test only for male vs female
    if set(d["gender"].unique()) >= {"male", "female"} and (option.startswith("male vs female")):
        a = d[d["gender"] == "female"]["exam_score"].astype(float)
        b = d[d["gender"] == "male"]["exam_score"].astype(float)
        t = stats.ttest_ind(a, b, equal_var=False)

        diff = float(a.mean() - b.mean())

        n1, n2 = len(a), len(b)
        s1, s2 = a.std(ddof=1), b.std(ddof=1)
        sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        d_cohen = diff / sp if sp > 0 else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("평균차(여-남)", f"{diff:.3f}")
        c2.metric("p-value", f"{t.pvalue:.3f}")
        c3.metric("Cohen's d", f"{d_cohen:.3f}")

    # Plotly box + mean bar
    fig1 = px.box(d, x="gender", y="exam_score", points="outliers",
                  title="Exam Score by Gender (Boxplot)",
                  labels={"gender": "Gender", "exam_score": "Exam Score"})
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(summary, x="gender", y="mean", error_y=summary["std"]/np.sqrt(summary["count"]),
                  title="Mean Exam Score by Gender (Mean ± SE)",
                  labels={"gender": "Gender", "mean": "Mean Exam Score"})
    st.plotly_chart(fig2, use_container_width=True)

def analysis_internet_vs_sleep_quality(df: pd.DataFrame):
    st.subheader("3) 인터넷 이용(internet_access)에 따른 수면의 질(sleep_quality) 비교")

    d = df[["internet_access", "sleep_quality"]].dropna()
    ct = pd.crosstab(d["internet_access"], d["sleep_quality"])
    st.write("**교차표(Counts)**")
    st.dataframe(ct)

    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    cramers_v = np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Chi-square", f"{chi2:.3f}")
    c2.metric("p-value", f"{p:.3f}")
    c3.metric("Cramer's V", f"{cramers_v:.3f}")

    # Row % stacked bar
    row_pct = ct.div(ct.sum(axis=1), axis=0).reset_index()
    long = row_pct.melt(id_vars="internet_access", var_name="sleep_quality", value_name="proportion")
    long["proportion"] = long["proportion"] * 100

    fig = px.bar(long, x="internet_access", y="proportion", color="sleep_quality",
                 barmode="stack",
                 title="Sleep Quality Distribution by Internet Access (Row %)",
                 labels={"internet_access": "Internet Access", "proportion": "Percent (%)", "sleep_quality": "Sleep Quality"})
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Exam Score Prediction 데이터 분석 웹앱")

    st.sidebar.header("데이터 선택")
    uploaded_files = st.sidebar.file_uploader(
        "같은 형식의 CSV를 업로드하면 해당 데이터로 자동 분석됩니다 (복수 업로드 가능)",
        type=["csv"],
        accept_multiple_files=True
    )

    use_uploaded = uploaded_files is not None and len(uploaded_files) > 0

    if use_uploaded:
        dfs = []
        schema_errors = []
        for f in uploaded_files:
            try:
                temp = read_uploaded_csv(f)
                ok, missing, extra = validate_schema(temp)
                if not ok:
                    schema_errors.append((f.name, missing, extra))
                else:
                    dfs.append(temp)
            except Exception as e:
                schema_errors.append((f.name, ["(read error)"], [str(e)]))

        if schema_errors:
            st.sidebar.error("업로드 파일 중 형식이 맞지 않는 파일이 있습니다.")
            for name, missing, extra in schema_errors:
                st.sidebar.write(f"- {name}")
                st.sidebar.write(f"  - missing: {missing}")
                st.sidebar.write(f"  - extra: {extra}")

        if len(dfs) == 0:
            st.warning("정상적으로 읽힌 업로드 데이터가 없어 기본 탑재 데이터를 사용합니다.")
            df = load_default_data()
        else:
            df = pd.concat(dfs, ignore_index=True)
            st.sidebar.success(f"업로드 데이터 사용 중: {len(dfs)}개 파일, 합계 {df.shape[0]}행")
    else:
        df = load_default_data()
        st.sidebar.info("기본 탑재 데이터 사용 중 (Exam_Score_Prediction.csv)")

    # type & quality
    if list(df.columns) != EXPECTED_COLUMNS:
        # 열 순서가 다를 수 있으니 재정렬
        df = df[[c for c in EXPECTED_COLUMNS if c in df.columns]]

    df = coerce_types(df)

    tabs = st.tabs(["데이터 점검", "분석 1", "분석 2", "분석 3"])

    with tabs[0]:
        basic_quality_checks(df)

    with tabs[1]:
        analysis_study_hours_vs_score(df)

    with tabs[2]:
        analysis_gender_mean_diff(df)

    with tabs[3]:
        analysis_internet_vs_sleep_quality(df)

if __name__ == "__main__":
    main()

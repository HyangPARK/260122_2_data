import pandas as pd
import streamlit as st
import plotly.express as px

DATA_PATH = "ES_Pre.csv"
DATA_SOURCE_URL = "https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset"

EXPECTED_COLS = [
    "student_id", "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty", "exam_score"
]

VAR_DESC = [
    ("student_id", "학생 고유 번호(식별자)", "10001"),
    ("age", "나이", "19"),
    ("gender", "성별(male/female/other)", "female"),
    ("course", "과목", "Mathematics"),
    ("study_hours", "공부시간(시간)", "2.5"),
    ("class_attendance", "출석률(%)", "92.0"),
    ("internet_access", "인터넷 이용(yes/no)", "yes"),
    ("sleep_hours", "수면시간(시간)", "7.0"),
    ("sleep_quality", "수면의 질(good/average/poor)", "good"),
    ("study_method", "공부 방식", "Self Study"),
    ("facility_rating", "시설 만족도(범주형)", "High"),
    ("exam_difficulty", "시험 난이도(범주형)", "Medium"),
    ("exam_score", "시험 점수(0~100)", "78.0"),
]

st.set_page_config(page_title="01 데이터 소개", layout="wide")
st.title("01) 데이터 소개")

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.subheader("데이터 출처")
st.markdown(f"- Kaggle: [{DATA_SOURCE_URL}]({DATA_SOURCE_URL})")

st.subheader("기본 정보")
c1, c2, c3 = st.columns(3)
c1.metric("행(학생 수)", f"{df.shape[0]:,}")
c2.metric("열(변수 수)", f"{df.shape[1]:,}")
c3.metric("결측치 수(전체)", f"{int(df.isna().sum().sum()):,}")

st.subheader("데이터 미리보기")
st.dataframe(df.head(15), use_container_width=True)

st.subheader("변수 설명")
st.dataframe(pd.DataFrame(VAR_DESC, columns=["변수", "설명", "예시"]), use_container_width=True)

st.subheader("열 구성 점검")
cols = df.columns.tolist()
missing = [c for c in EXPECTED_COLS if c not in cols]
extra = [c for c in cols if c not in EXPECTED_COLS]
if missing:
    st.warning(f"없는 열: {missing}")
if extra:
    st.info(f"추가 열: {extra}")
if (not missing) and (not extra):
    st.success("열 구성이 기대 형식과 동일합니다.")

st.subheader("간단한 분포")
if "course" in df.columns:
    top = df["course"].value_counts().head(12).reset_index()
    top.columns = ["course", "count"]
    fig = px.bar(top, x="course", y="count", title="과목별 학생 수(상위 12)")
    st.plotly_chart(fig, use_container_width=True)

if "exam_score" in df.columns:
    fig2 = px.histogram(df.dropna(subset=["exam_score"]), x="exam_score", nbins=30, title="시험점수 분포")
    st.plotly_chart(fig2, use_container_width=True)

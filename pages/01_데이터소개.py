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
    ("student_id", "학생 고유 번호(식별자). 보통 분석에서는 제외", "10001"),
    ("age", "나이", "19"),
    ("gender", "성별 (male/female/other)", "female"),
    ("course", "과목(수강 과목)", "Mathematics"),
    ("study_hours", "공부시간(시간)", "2.5"),
    ("class_attendance", "출석률(%)", "92.0"),
    ("internet_access", "인터넷 이용 가능 여부 (yes/no)", "yes"),
    ("sleep_hours", "수면시간(시간)", "7.0"),
    ("sleep_quality", "수면의 질 (good/average/poor)", "good"),
    ("study_method", "공부 방식", "Self Study"),
    ("facility_rating", "시설 만족도(범주형)", "High"),
    ("exam_difficulty", "시험 난이도(범주형)", "Medium"),
    ("exam_score", "시험 점수(0~100점)", "78.0"),
]

st.set_page_config(page_title="데이터 소개", layout="wide")
st.title("01) 데이터 소개")

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.subheader("데이터 출처")
st.markdown(f"- Kaggle: [{DATA_SOURCE_URL}]({DATA_SOURCE_URL})")
st.caption("※ 교육/연구용 데모 분석에 적합한 예제 데이터셋입니다.")

st.subheader("데이터 기본 정보")
c1, c2, c3 = st.columns(3)
c1.metric("행(학생 수)", f"{df.shape[0]:,}")
c2.metric("열(변수 수)", f"{df.shape[1]:,}")
c3.metric("결측치(비어있는 값)", f"{int(df.isna().sum().sum()):,}")

st.subheader("데이터 예시")
st.dataframe(df.head(12), use_container_width=True)

st.subheader("변수(열) 설명")
desc_df = pd.DataFrame(VAR_DESC, columns=["변수명", "설명", "예시"])
st.dataframe(desc_df, use_container_width=True)

st.subheader("열(변수) 형식 확인")
cols_in_df = df.columns.tolist()
missing = [c for c in EXPECTED_COLS if c not in cols_in_df]
extra = [c for c in cols_in_df if c not in EXPECTED_COLS]
if missing:
    st.warning(f"데이터에 없는 열: {missing}")
if extra:
    st.info(f"추가로 들어있는 열: {extra}")
if not missing and not extra:
    st.success("열 구성이 기대 형식과 동일합니다.")

st.subheader("간단한 분포 보기")
if "course" in df.columns:
    top_course = df["course"].value_counts().head(10).reset_index()
    top_course.columns = ["course", "count"]
    fig1 = px.bar(top_course, x="course", y="count", title="과목별 학생 수(상위 10개)")
    st.plotly_chart(fig1, use_container_width=True)

if "exam_score" in df.columns:
    fig2 = px.histogram(df.dropna(subset=["exam_score"]), x="exam_score", nbins=30, title="시험점수 분포")
    st.plotly_chart(fig2, use_container_width=True)

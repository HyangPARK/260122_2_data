import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "ES_Pre.csv"

FEATURE_COLS = [
    "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty"
]
TARGET = "exam_score"
NUM_COLS = ["age", "study_hours", "class_attendance", "sleep_hours"]
MIN_N_PER_COURSE_MODEL = 120

st.set_page_config(page_title="05 AI 학습코치", layout="wide")
st.title("05) AI 학습코치 (ML + LLM)")

@st.cache_data(show_spinner=False)
def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    for c in NUM_COLS + [TARGET]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.loc[(df[TARGET] < 0) | (df[TARGET] > 100), TARGET] = np.nan
    if "class_attendance" in df.columns:
        df.loc[(df["class_attendance"] < 0) | (df["class_attendance"] > 100), "class_attendance"] = np.nan
    for c in ["study_hours", "sleep_hours"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    for c in df.columns:
        if c not in NUM_COLS + [TARGET]:
            df[c] = df[c].astype(str).str.strip()

    cols = [c for c in FEATURE_COLS + [TARGET] if c in df.columns]
    return df[cols].copy()

def build_pipeline(feature_cols, num_cols, cat_cols):
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline([("prep", preprocess), ("model", model)])

@st.cache_resource(show_spinner=False)
def train_models(d, feature_cols, num_cols, cat_cols, target):
    # 전체 모델
    global_pipe = build_pipeline(feature_cols, num_cols, cat_cols)
    global_pipe.fit(d[feature_cols], d[target].astype(float))

    # 과목별 모델(가능한 과목만)
    course_models = {}
    if "course" in d.columns:
        for course, g in d.groupby("course"):
            if len(g) >= MIN_N_PER_COURSE_MODEL:
                pipe = build_pipeline(feature_cols, num_cols, cat_cols)
                pipe.fit(g[feature_cols], g[target].astype(float))
                course_models[course] = pipe

    # 기본값(중앙값/최빈값)
    defaults = {}
    for c in num_cols:
        defaults[c] = float(d[c].median())
    for c in cat_cols:
        defaults[c] = d[c].mode(dropna=True)[0]

    return global_pipe, course_models, defaults

def llm_advice(inputs: dict, pred_global: float, pred_course: float | None):
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 secrets에 없습니다.")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    course = inputs.get("course", "")
    pred_line = f"전체 모델 예측: {pred_global:.1f}점"
    if pred_course is not None:
        pred_line += f" / 과목별 모델 예측({course}): {pred_course:.1f}점"

    prompt = f"""
당신은 학생을 돕는 학습코치입니다.
아래 학생의 정보와 ML 예측을 바탕으로, 한국어로 쉽고 실천 가능한 조언을 작성하세요.

학생 정보:
{inputs}

ML 예측:
{pred_line}

요구사항:
- 어려운 용어는 피하고, 짧고 명확하게
- 가능한 한 '오늘 바로 할 수 있는 행동' 중심

출력 형식:
1) 한 줄 진단(1문장)
2) 우선순위 TOP 3 행동(각 1~2문장)
3) 이번 주 체크리스트(3~6개)
4) 주의할 점(2~4개)
5) 과목이 있으면({course}) 그 과목에 맞는 팁 2개를 추가
"""
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text

df = load_and_clean()
d = df.dropna(subset=[TARGET]).copy()

feature_cols = [c for c in FEATURE_COLS if c in d.columns]
num_cols = [c for c in NUM_COLS if c in feature_cols]
cat_cols = [c for c in feature_cols if c not in num_cols]

global_pipe, course_models, defaults = train_models(d, feature_cols, num_cols, cat_cols, TARGET)

st.subheader("1) 내 조건 입력")
inputs = {}

colA, colB = st.columns(2)
with colA:
    if "study_hours" in feature_cols:
        inputs["study_hours"] = st.slider("공부시간(시간)", 0.0, 12.0, float(defaults.get("study_hours", 2.0)), 0.5)
    if "sleep_hours" in feature_cols:
        inputs["sleep_hours"] = st.slider("수면시간(시간)", 0.0, 12.0, float(defaults.get("sleep_hours", 7.0)), 0.5)
    if "class_attendance" in feature_cols:
        inputs["class_attendance"] = st.slider("출석률(%)", 0.0, 100.0, float(defaults.get("class_attendance", 90.0)), 1.0)
    if "age" in feature_cols:
        inputs["age"] = st.slider("나이", 10, 40, int(round(defaults.get("age", 20))), 1)

with colB:
    for c in cat_cols:
        options = sorted(d[c].dropna().unique().tolist())
        if not options:
            continue
        default = defaults.get(c, options[0])
        if default not in options:
            default = options[0]
        inputs[c] = st.selectbox(c, options, index=options.index(default))

row = pd.DataFrame([inputs])[feature_cols]
pred_global = float(global_pipe.predict(row)[0])

pred_course = None
if "course" in inputs and inputs["course"] in course_models:
    pred_course = float(course_models[inputs["course"]].predict(row)[0])

c1, c2 = st.columns(2)
c1.metric("예상 점수(전체)", f"{pred_global:.1f}점")
c2.metric("예상 점수(과목별)", f"{pred_course:.1f}점" if pred_course is not None else "해당 과목 표본 부족")

st.divider()
st.subheader("2) AI 조언 생성(LLM)")

if st.button("AI 조언 생성"):
    try:
        with st.spinner("AI가 공부전략을 만드는 중..."):
            advice = llm_advice(inputs, pred_global, pred_course)
        st.write(advice)
    except Exception as e:
        st.error(f"오류: {e}")
        st.info("Secrets에 OPENAI_API_KEY가 있는지 확인하세요.")

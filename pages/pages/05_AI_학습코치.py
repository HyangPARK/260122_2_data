import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "ES_Pre.csv"

st.set_page_config(page_title="AI 학습코치", layout="wide")
st.title("05) AI 학습코치 (ML + LLM)")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    for c in ["age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_resource(show_spinner=False)
def train_pipeline(df: pd.DataFrame):
    target = "exam_score"
    feature_cols = [
        "age", "gender", "course", "study_hours", "class_attendance",
        "internet_access", "sleep_hours", "sleep_quality", "study_method",
        "facility_rating", "exam_difficulty"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    d = df[feature_cols + [target]].dropna(subset=[target]).copy()

    num_cols = [c for c in feature_cols if c in ["age", "study_hours", "class_attendance", "sleep_hours"]]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", preprocess), ("model", model)])

    X = d[feature_cols]
    y = d[target].astype(float)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    # 기본값(중앙값/최빈값)
    defaults = {}
    for c in num_cols:
        defaults[c] = float(d[c].median())
    for c in cat_cols:
        defaults[c] = d[c].mode(dropna=True)[0]

    return pipe, feature_cols, num_cols, cat_cols, defaults

def llm_coach_advice(user_inputs: dict, pred_score: float) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    prompt = f"""
당신은 학생을 돕는 학습코치입니다.
아래는 한 학생의 현재 학습/생활 정보와 ML이 예측한 시험점수입니다.
한국어로 쉽고 친절하게, '바로 실행 가능한' 조언을 해주세요.

학생 정보:
{user_inputs}

ML 예측 점수: {pred_score:.1f}점

출력 형식(꼭 지켜주세요):
1) 현재 상태 한 줄 진단(1문장)
2) 우선순위 TOP 3 행동(각 1~2문장)
3) 이번 주 실천 계획(월~일 중 3~5개 할 일 체크리스트)
4) 주의할 점(2~4개)
"""
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text

df = load_data()
pipe, feature_cols, num_cols, cat_cols, defaults = train_pipeline(df)

st.subheader("1) 내 조건 입력 → 예상 점수")
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
        options = sorted(df[c].dropna().unique().tolist())
        if not options:
            continue
        default = defaults.get(c, options[0])
        if default not in options:
            default = options[0]
        inputs[c] = st.selectbox(c, options, index=options.index(default))

row = pd.DataFrame([inputs])[feature_cols]
pred_score = float(pipe.predict(row)[0])

st.success(f"예상 시험점수: **{pred_score:.1f}점**")
st.caption("※ 이 점수는 데이터 패턴 기반의 예측값입니다. 실제 점수와 다를 수 있어요.")

st.divider()
st.subheader("2) AI 학습코치 조언 받기 (LLM)")

if st.button("조언 생성하기"):
    try:
        with st.spinner("AI 코치가 전략을 만드는 중..."):
            advice = llm_coach_advice(inputs, pred_score)
        st.write(advice)
    except Exception as e:
        st.error(f"오류: {e}")
        st.info("OpenAI 키가 secrets에 설정되어 있는지 확인해주세요.")

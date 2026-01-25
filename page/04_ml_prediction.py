import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "ES_Pre.csv"

FEATURE_COLS = [
    "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty"
]
TARGET = "exam_score"
NUM_COLS = ["age", "study_hours", "class_attendance", "sleep_hours"]

MIN_N_PER_COURSE_MODEL = 120  # 과목별 모델 학습 최소 표본(너무 작으면 불안정)

st.set_page_config(page_title="04 ML 예측", layout="wide")
st.title("04) ML 예측: 전체 모델 + 과목별 모델")

@st.cache_data(show_spinner=False)
def load_and_clean():
    df = pd.read_csv(DATA_PATH)

    # 숫자형 변환
    for c in NUM_COLS + [TARGET]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 범위 정리
    df.loc[(df[TARGET] < 0) | (df[TARGET] > 100), TARGET] = np.nan
    if "class_attendance" in df.columns:
        df.loc[(df["class_attendance"] < 0) | (df["class_attendance"] > 100), "class_attendance"] = np.nan
    for c in ["study_hours", "sleep_hours"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # 문자열 정리
    for c in df.columns:
        if c not in NUM_COLS + [TARGET]:
            df[c] = df[c].astype(str).str.strip()

    # 필요한 열만
    cols = [c for c in FEATURE_COLS + [TARGET] if c in df.columns]
    df = df[cols].copy()
    return df

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

df = load_and_clean()

# 유효 열만
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
num_cols = [c for c in NUM_COLS if c in feature_cols]
cat_cols = [c for c in feature_cols if c not in num_cols]

# 학습용 데이터
d = df.dropna(subset=[TARGET]).copy()
X = d[feature_cols]
y = d[TARGET].astype(float)

# 전체 모델
@st.cache_resource(show_spinner=False)
def train_global(X, y, feature_cols, num_cols, cat_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline(feature_cols, num_cols, cat_cols)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return pipe, float(mean_absolute_error(y_test, pred)), float(r2_score(y_test, pred))

global_pipe, global_mae, global_r2 = train_global(X, y, feature_cols, num_cols, cat_cols)

c1, c2 = st.columns(2)
c1.metric("전체 모델 MAE(평균 오차)", f"{global_mae:.2f}점")
c2.metric("전체 모델 R²", f"{global_r2:.3f}")
st.caption("※ 데이터/모델에 따라 수치는 달라질 수 있습니다.")

st.divider()

# --- 과목별 모델 학습 (표본 충분한 과목만)
st.subheader("1) 과목별 모델(표본 충분할 때만 학습)")

course_models = {}
course_counts = {}

if "course" in d.columns:
    for course, g in d.groupby("course"):
        course_counts[course] = len(g)
    eligible = [c for c, n in course_counts.items() if n >= MIN_N_PER_COURSE_MODEL]

    st.write(f"- 과목별 모델 학습 최소 표본수: **{MIN_N_PER_COURSE_MODEL}**")
    st.write(f"- 학습 가능한 과목 수: **{len(eligible)}**")

    @st.cache_resource(show_spinner=False)
    def train_course_models(d, feature_cols, num_cols, cat_cols, target, eligible_courses):
        models = {}
        for course in eligible_courses:
            g = d[d["course"] == course]
            Xc = g[feature_cols]
            yc = g[target].astype(float)
            pipe = build_pipeline(feature_cols, num_cols, cat_cols)
            pipe.fit(Xc, yc)
            models[course] = pipe
        return models

    course_models = train_course_models(d, feature_cols, num_cols, cat_cols, TARGET, eligible)

    table = pd.DataFrame([{"과목": c, "표본수": course_counts[c], "과목별모델": ("가능" if c in course_models else "불가")} for c in sorted(course_counts)])
    st.dataframe(table, use_container_width=True)
else:
    st.info("course 열이 없어 과목별 모델을 만들 수 없습니다.")

st.divider()

# --- 예측 UI
st.subheader("2) 내 조건 입력 → 예상 점수(전체/과목별)")

# 기본값
defaults = {}
for c in num_cols:
    defaults[c] = float(d[c].median())
for c in cat_cols:
    defaults[c] = d[c].mode(dropna=True)[0]

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

# 과목별 모델이 있으면 과목 기준으로 우선 사용
pred_course = None
used_model = "전체 모델"
if "course" in inputs and inputs["course"] in course_models:
    pred_course = float(course_models[inputs["course"]].predict(row)[0])
    used_model = "과목별 모델"

c1, c2 = st.columns(2)
c1.metric("예상 점수(전체 모델)", f"{pred_global:.1f}점")
if pred_course is not None:
    c2.metric("예상 점수(과목별 모델)", f"{pred_course:.1f}점")
else:
    c2.metric("예상 점수(과목별 모델)", "해당 과목 표본 부족")

st.caption(f"현재 표시: **{used_model} 우선** (과목별 모델이 있으면 과목별 예측을 참고하세요.)")

st.divider()

# --- What-if: 공부시간 변화
st.subheader("3) what-if: 공부시간을 늘리면 예측 점수는?")
if "study_hours" in feature_cols:
    base = defaults.copy()
    base.update(inputs)

    hours = np.linspace(0, 12, 25)
    rows = []
    for h in hours:
        tmp = base.copy()
        tmp["study_hours"] = float(h)
        rows.append(tmp)

    curve_df = pd.DataFrame(rows)[feature_cols]
    curve_df["pred_global"] = global_pipe.predict(curve_df)
    fig = px.line(curve_df, x="study_hours", y="pred_global", markers=True, title="공부시간 변화에 따른 예측(전체 모델)")
    fig.update_layout(xaxis_title="공부시간(시간)", yaxis_title="예측 점수")
    st.plotly_chart(fig, use_container_width=True)

    if pred_course is not None:
        curve_df["pred_course"] = course_models[inputs["course"]].predict(curve_df)
        fig2 = px.line(curve_df, x="study_hours", y="pred_course", markers=True,
                       title=f"공부시간 변화에 따른 예측(과목별 모델: {inputs['course']})")
        fig2.update_layout(xaxis_title="공부시간(시간)", yaxis_title="예측 점수")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("study_hours 열이 없어 what-if 그래프를 만들 수 없습니다.")

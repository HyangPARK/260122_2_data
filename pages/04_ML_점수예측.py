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
from sklearn.inspection import permutation_importance

DATA_PATH = "ES_Pre.csv"

st.set_page_config(page_title="ML 점수예측", layout="wide")
st.title("04) ML 점수 예측")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    for c in ["age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

target = "exam_score"
feature_cols = [
    "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty"
]
feature_cols = [c for c in feature_cols if c in df.columns]
df = df[feature_cols + [target]].dropna(subset=[target]).copy()

num_cols = [c for c in feature_cols if c in ["age", "study_hours", "class_attendance", "sleep_hours"]]
cat_cols = [c for c in feature_cols if c not in num_cols]

numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
pipe = Pipeline([("prep", preprocess), ("model", model)])

X = df[feature_cols]
y = df[target].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

c1, c2 = st.columns(2)
c1.metric("예측 오차(평균, MAE)", f"{mae:.2f}점")
c2.metric("설명력(R²)", f"{r2:.3f}")
st.caption("※ ML은 정답이 아니라 ‘데이터 패턴 기반의 대략적 예측’입니다.")

st.divider()

st.subheader("1) 내 조건을 넣고 예상 점수 보기")

defaults = {}
for c in num_cols:
    defaults[c] = float(df[c].median())
for c in cat_cols:
    defaults[c] = df[c].mode(dropna=True)[0]

with st.form("predict_form"):
    colA, colB = st.columns(2)
    inputs = {}

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

    submitted = st.form_submit_button("예상 점수 계산")

if submitted:
    row = pd.DataFrame([inputs])[feature_cols]
    yhat = float(pipe.predict(row)[0])
    st.success(f"예상 시험점수: **{yhat:.1f}점**")

st.divider()

st.subheader("2) 공부시간을 바꾸면 점수 예측이 어떻게 변할까? (what-if)")
if "study_hours" in feature_cols:
    base = defaults.copy()
    try:
        base.update(inputs)
    except Exception:
        pass

    hours = np.linspace(0, 12, 25)
    rows = []
    for h in hours:
        tmp = base.copy()
        tmp["study_hours"] = float(h)
        rows.append(tmp)

    curve_df = pd.DataFrame(rows)[feature_cols]
    curve_df["pred_score"] = pipe.predict(curve_df)

    fig = px.line(curve_df, x="study_hours", y="pred_score", markers=True,
                  title="공부시간에 따른 예상 점수(다른 조건은 고정)")
    fig.update_layout(xaxis_title="공부시간(시간)", yaxis_title="예상 점수")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("study_hours 변수가 없어 what-if 그래프를 만들 수 없습니다.")

st.divider()

st.subheader("3) 어떤 변수가 점수 예측에 더 영향을 줄까? (중요도)")
with st.spinner("중요도 계산 중..."):
    r = permutation_importance(pipe, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)

imp = pd.DataFrame({"변수": feature_cols, "중요도": r.importances_mean}).sort_values("중요도", ascending=False)
fig = px.bar(imp, x="중요도", y="변수", orientation="h", title="변수 중요도(대략)")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(imp, use_container_width=True)

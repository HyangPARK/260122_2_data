import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import requests
import trafilatura
from openai import OpenAI

st.set_page_config(page_title="시험점수 데이터 분석", layout="wide")

# -----------------------------
# 기본 설정
# -----------------------------
기본_데이터_파일 = "ES_Pre.csv"
데이터_출처_URL = "https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset"

기대_열 = [
    "student_id", "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty", "exam_score"
]
수치형_열 = ["student_id", "age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]

변수_설명 = [
    {"변수명": "student_id", "설명": "학생 고유 번호(식별자). 보통 분석에서는 제외합니다.", "예시": "10001"},
    {"변수명": "age", "설명": "나이", "예시": "19"},
    {"변수명": "gender", "설명": "성별 (male/female/other)", "예시": "female"},
    {"변수명": "course", "설명": "과목(수강 과목)", "예시": "Mathematics"},
    {"변수명": "study_hours", "설명": "공부시간(시간)", "예시": "2.5"},
    {"변수명": "class_attendance", "설명": "출석률(%)", "예시": "92.0"},
    {"변수명": "internet_access", "설명": "인터넷 이용 가능 여부 (yes/no)", "예시": "yes"},
    {"변수명": "sleep_hours", "설명": "수면시간(시간)", "예시": "7.0"},
    {"변수명": "sleep_quality", "설명": "수면의 질 (good/average/poor)", "예시": "good"},
    {"변수명": "study_method", "설명": "공부 방식(예: 혼자/그룹/온라인 등)", "예시": "Self Study"},
    {"변수명": "facility_rating", "설명": "시설 만족도(등급/범주형)", "예시": "High"},
    {"변수명": "exam_difficulty", "설명": "시험 난이도(등급/범주형)", "예시": "Medium"},
    {"변수명": "exam_score", "설명": "시험 점수(0~100점)", "예시": "78.0"},
]

# 대비색(요청 반영)
성별_색상 = {"male": "#1f77b4", "female": "#ff7f0e", "other": "#2ca02c"}

# 과목 한글 매핑(검색 보조용)
과목_한글_매핑 = {
    "Mathematics": "수학",
    "English": "영어",
    "History": "역사",
    "Science": "과학",
    "Biology": "생물",
    "Chemistry": "화학",
    "Physics": "물리",
    "Computer Science": "컴퓨터/코딩",
    "Economics": "경제",
    "Accounting": "회계",
    "Marketing": "마케팅",
    "Statistics": "통계",
    "Psychology": "심리",
    "Sociology": "사회",
}

# -----------------------------
# 데이터 로딩/검증
# -----------------------------
@st.cache_data(show_spinner=False)
def 기본_데이터_불러오기():
    return pd.read_csv(기본_데이터_파일)

def 업로드_csv_읽기(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read()
    return pd.read_csv(io.BytesIO(content))

def 스키마_검사(df: pd.DataFrame):
    cols = list(df.columns)
    missing = [c for c in 기대_열 if c not in cols]
    extra = [c for c in cols if c not in 기대_열]
    ok = (len(missing) == 0) and (len(extra) == 0)
    return ok, missing, extra

def 타입_정리(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in 수치형_열:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [c for c in 기대_열 if c not in 수치형_열]:
        df[c] = df[c].astype(str)
    return df

# -----------------------------
# UI(소개)
# -----------------------------
def 귀여운_SVG():
    return """
    <div style="display:flex; gap:16px; align-items:center; padding:12px 14px; border-radius:16px; background:#f6f8ff;">
      <svg width="96" height="96" viewBox="0 0 96 96" xmlns="http://www.w3.org/2000/svg" role="img">
        <defs>
          <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#b8c6ff"/>
            <stop offset="1" stop-color="#ffe3f1"/>
          </linearGradient>
        </defs>
        <circle cx="48" cy="48" r="44" fill="url(#g)"/>
        <circle cx="34" cy="42" r="6" fill="#2d2d2d"/>
        <circle cx="62" cy="42" r="6" fill="#2d2d2d"/>
        <circle cx="32" cy="44" r="2" fill="#ffffff"/>
        <circle cx="60" cy="44" r="2" fill="#ffffff"/>
        <path d="M44 52 Q48 58 52 52" stroke="#2d2d2d" stroke-width="3" fill="none" stroke-linecap="round"/>
        <path d="M30 60 Q48 72 66 60" stroke="#2d2d2d" stroke-width="3" fill="none" stroke-linecap="round"/>
        <circle cx="26" cy="54" r="6" fill="#ffb3c7" opacity="0.9"/>
        <circle cx="70" cy="54" r="6" fill="#ffb3c7" opacity="0.9"/>
      </svg>
      <div>
        <div style="font-size:18px; font-weight:700;">시험점수 데이터 분석</div>
        <div style="font-size:14px; color:#555; margin-top:4px;">
          공부전략은 ‘웹페이지 검색 → 요약’ 방식으로 제공합니다.
        </div>
      </div>
    </div>
    """

def 첫페이지_데이터_소개(df: pd.DataFrame):
    st.markdown(귀여운_SVG(), unsafe_allow_html=True)

    st.subheader("데이터 출처")
    st.write("이 웹앱은 아래 데이터셋을 기반으로 분석합니다.")
    st.markdown(f"- Kaggle: [{데이터_출처_URL}]({데이터_출처_URL})")

    st.subheader("데이터 기본 정보")
    c1, c2, c3 = st.columns(3)
    c1.metric("행(학생 수)", f"{df.shape[0]:,}")
    c2.metric("열(변수 수)", f"{df.shape[1]:,}")
    c3.metric("결측치(비어있는 값)", f"{int(df.isna().sum().sum()):,}")

    st.write("**데이터 예시(앞 10행)**")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("변수(열) 설명")
    st.dataframe(pd.DataFrame(변수_설명), use_container_width=True)

    st.subheader("간단히 보기")
    top_course = df["course"].value_counts().head(10).reset_index()
    top_course.columns = ["course", "count"]
    fig = px.bar(top_course, x="course", y="count", title="과목별 학생 수(상위 10개)")
    fig.update_layout(xaxis_title="과목", yaxis_title="학생 수")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(df.dropna(subset=["exam_score"]), x="exam_score", nbins=30, title="시험점수 분포")
    fig2.update_layout(xaxis_title="시험점수", yaxis_title="학생 수")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 분석 1~4
# -----------------------------
def 분석_공부시간_시험점수(df: pd.DataFrame):
    st.subheader("1) 공부시간이 늘면 시험점수가 올라가나요?")
    d = df[["study_hours", "exam_score"]].dropna()
    x = d["study_hours"].astype(float).values
    y = d["exam_score"].astype(float).values

    r, p = stats.pearsonr(x, y)
    reg = stats.linregress(x, y)

    c1, c2, c3 = st.columns(3)
    c1.metric("상관계수(r)", f"{r:.3f}")
    c2.metric("p값", f"{p:.3f}")
    c3.metric("대략적인 증가폭", f"+{reg.slope:.2f}점/1시간")

    fig = px.scatter(d, x="study_hours", y="exam_score", opacity=0.35,
                     title="공부시간과 시험점수 (점 + 추세선)")
    xs = np.linspace(x.min(), x.max(), 200)
    ys = reg.intercept + reg.slope * xs
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="추세선"))
    fig.update_layout(xaxis_title="공부시간(시간)", yaxis_title="시험점수")
    st.plotly_chart(fig, use_container_width=True)

def 분석_성별_평균(df: pd.DataFrame):
    st.subheader("2) 남/여 시험점수 평균이 다를까요?")
    d = df[df["gender"].isin(["male", "female"])][["gender", "exam_score"]].dropna()

    summary = d.groupby("gender")["exam_score"].agg(["count", "mean", "std"]).reset_index()
    summary["SE"] = summary["std"] / np.sqrt(summary["count"])
    summary = summary.rename(columns={"gender": "성별", "count": "사람수", "mean": "평균점수", "std": "표준편차"})
    st.dataframe(summary, use_container_width=True)

    fig1 = px.box(d, x="gender", y="exam_score", color="gender", points="outliers",
                  color_discrete_map=성별_색상, title="성별 시험점수 분포(박스플롯)")
    fig1.update_layout(xaxis_title="성별", yaxis_title="시험점수")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(summary, x="성별", y="평균점수", error_y="SE", color="성별",
                  color_discrete_map={"male": 성별_색상["male"], "female": 성별_색상["female"]},
                  title="성별 평균 시험점수(평균 ± 표준오차)")
    fig2.update_layout(yaxis_title="평균 시험점수")
    st.plotly_chart(fig2, use_container_width=True)

def 분석_인터넷_수면질(df: pd.DataFrame):
    st.subheader("3) 인터넷 이용 여부에 따라 수면의 질이 다를까요? (좋음/나쁨)")
    d = df[["internet_access", "sleep_quality"]].dropna()
    d2 = d[d["sleep_quality"].isin(["good", "poor"])].copy()

    ct = pd.crosstab(d2["internet_access"], d2["sleep_quality"])
    ct = ct.rename(index={"yes": "인터넷 사용", "no": "인터넷 미사용"},
                   columns={"good": "좋음", "poor": "나쁨"})
    st.dataframe(ct, use_container_width=True)

    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    st.write("**비율(%)**")
    st.dataframe(pct, use_container_width=True)

    long = pct.reset_index().melt(id_vars="internet_access", var_name="수면의 질", value_name="비율(%)")
    fig = px.bar(long, x="internet_access", y="비율(%)", color="수면의 질", barmode="stack",
                 title="인터넷 이용 여부에 따른 수면의 질 비율(좋음/나쁨)")
    fig.update_layout(xaxis_title="", yaxis_title="비율(%)")
    st.plotly_chart(fig, use_container_width=True)

def 분석_공부방식_평균(df: pd.DataFrame):
    st.subheader("4) 공부 방식에 따라 시험점수 평균이 다를까요?")
    d = df[["study_method", "exam_score"]].dropna()

    summary = (
        d.groupby("study_method")["exam_score"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"study_method": "공부 방식", "count": "사람수", "mean": "평균점수", "std": "표준편차"})
    )
    summary["표준오차(SE)"] = summary["표준편차"] / np.sqrt(summary["사람수"])
    summary = summary.sort_values("평균점수", ascending=False)

    st.dataframe(summary, use_container_width=True)

    fig = px.bar(summary, x="공부 방식", y="평균점수", error_y="표준오차(SE)",
                 title="공부 방식별 평균 시험점수(평균 ± 표준오차)")
    fig.update_layout(xaxis_title="공부 방식", yaxis_title="평균 시험점수")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 공부전략(웹): SerpAPI 검색 → 본문 추출 → LLM 요약
# -----------------------------
def serpapi_search(query: str, num: int = 10):
    api_key = st.secrets.get("SERPAPI_API_KEY", None)
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY가 설정되어 있지 않습니다.")

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "hl": "ko",
        "gl": "kr",
        "num": num,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = []
    for it in data.get("organic_results", []):
        title = it.get("title")
        link = it.get("link")
        snippet = it.get("snippet", "")
        if title and link:
            results.append({"title": title, "link": link, "snippet": snippet})
    return results

def extract_main_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError("웹페이지를 가져오지 못했습니다(접속 제한/차단/오류 가능).")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text or len(text.strip()) < 200:
        raise RuntimeError("본문 추출이 충분하지 않습니다(짧거나 추출 실패).")
    return text

def llm_summarize_web(text: str, user_goal: str = "") -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # 너무 길면 요약 전에 잘라서 비용/오류 방지
    # (대략 20,000자 정도로 제한)
    trimmed = text.strip()
    if len(trimmed) > 20000:
        trimmed = trimmed[:20000]

    prompt = f"""
당신은 학습코치입니다.
아래 웹페이지 본문을 읽고, 한국어로 쉽고 실용적으로 '공부 전략' 요약을 만들어주세요.
사용자 목적/검색 의도: {user_goal if user_goal else "공부 전략을 얻고 싶다"}

요약 형식(꼭 지켜주세요):
1) 한 줄 핵심(1문장)
2) 핵심 전략 5개(각 1~2문장, 바로 실천할 수 있게)
3) 오늘부터 체크리스트(3~6개)
4) 주의할 점(2~4개)
5) 이 글이 특히 도움이 되는 사람(1~2문장)

웹페이지 본문:
\"\"\"{trimmed}\"\"\"
"""
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text

def 공부전략_웹탭(df: pd.DataFrame):
    st.subheader("5) 공부전략(웹페이지 검색 → 요약)")

    st.write("자막 없는 영상 대신, **웹페이지(글)** 를 찾아서 요약합니다.")
    st.caption("검색 → 결과 선택 → 본문 추출 → 요약 순서입니다.")

    # 검색 UI: 자유 검색 + 옵션
    q = st.text_input("검색어(자유롭게 입력)", value="수학 성적 올리는 공부법 오답노트")

    col1, col2, col3 = st.columns([1.2, 1.2, 1])
    with col1:
        include_course = st.checkbox("데이터의 과목을 검색어에 추가(선택)", value=False)
    with col2:
        add_terms = st.checkbox("공부/시험 키워드 보정(추천)", value=True)
    with col3:
        num = st.selectbox("검색 결과 수", [5, 8, 10], index=1)

    if include_course:
        course = st.selectbox("과목 선택", sorted(df["course"].dropna().unique().tolist()))
        course_ko_manual = st.text_input("과목 한글명(선택)", value="")
        course_ko = course_ko_manual.strip() or 과목_한글_매핑.get(course, "")
    else:
        course = ""
        course_ko = ""

    query = q.strip()
    if include_course and course:
        query = f"{query} {course} {course_ko}".strip()
    if add_terms:
        query = f"{query} 공부법 시험공부 전략".strip()

    if "web_results" not in st.session_state:
        st.session_state["web_results"] = []

    if st.button("웹페이지 검색"):
        try:
            with st.spinner("검색 중..."):
                results = serpapi_search(query=query, num=num)
            st.session_state["web_results"] = results
            st.success(f"{len(results)}개 결과를 찾았습니다.")
            st.write(f"**실제 검색어:** {query}")
        except Exception as e:
            st.error(f"검색 오류: {e}")
            st.session_state["web_results"] = []

    results = st.session_state.get("web_results", [])
    if not results:
        st.write("검색 결과가 아직 없어요. 위에서 ‘웹페이지 검색’을 눌러주세요.")
        return

    st.divider()
    st.write("### 검색 결과(선택)")
    labels = [f"{i+1}. {r['title']}" for i, r in enumerate(results)]
    picked = st.selectbox("요약할 웹페이지를 선택하세요", options=labels, index=0)
    idx = int(picked.split(".")[0]) - 1
    chosen = results[idx]

    st.markdown(f"**선택한 페이지:** [{chosen['title']}]({chosen['link']})")
    if chosen.get("snippet"):
        st.write(f"요약(검색 결과 설명): {chosen['snippet']}")

    show_text = st.checkbox("본문(추출 텍스트)도 보기", value=False)

    if st.button("선택한 페이지 요약하기"):
        try:
            with st.spinner("웹페이지 본문을 추출 중..."):
                text = extract_main_text(chosen["link"])

            if show_text:
                st.subheader("본문(추출 텍스트)")
                st.text_area("추출된 본문", text, height=260)

            with st.spinner("LLM이 공부전략으로 요약 중..."):
                summary = llm_summarize_web(text, user_goal=query)

            st.subheader("요약 결과(공부 전략)")
            st.write(summary)

        except Exception as e:
            st.error(f"요약 오류: {e}")
            st.info("팁: 어떤 사이트는 본문 추출이 잘 안 될 수 있어요. 다른 결과를 선택해 보세요.")

# -----------------------------
# 메인
# -----------------------------
def main():
    st.title("시험점수 데이터 분석 웹앱")

    st.sidebar.header("데이터 선택")
    uploaded_files = st.sidebar.file_uploader(
        "같은 형식의 CSV를 업로드하면 업로드 데이터로 자동 분석됩니다",
        type=["csv"],
        accept_multiple_files=True
    )

    use_uploaded = uploaded_files is not None and len(uploaded_files) > 0

    if use_uploaded:
        dfs = []
        errors = []
        for f in uploaded_files:
            try:
                temp = 업로드_csv_읽기(f)
                ok, missing, extra = 스키마_검사(temp)
                if not ok:
                    errors.append((f.name, missing, extra))
                else:
                    dfs.append(temp)
            except Exception as e:
                errors.append((f.name, ["(읽기 오류)"], [str(e)]))

        if errors:
            st.sidebar.error("형식이 맞지 않는 업로드 파일이 있습니다.")
            for name, missing, extra in errors:
                st.sidebar.write(f"- {name}")
                st.sidebar.write(f"  - 없는 열: {missing}")
                st.sidebar.write(f"  - 추가 열: {extra}")

        if len(dfs) == 0:
            st.warning("정상 업로드 데이터가 없어 기본 탑재 데이터를 사용합니다.")
            df = 기본_데이터_불러오기()
        else:
            df = pd.concat(dfs, ignore_index=True)
            st.sidebar.success(f"업로드 데이터 사용: {len(dfs)}개 파일, 총 {df.shape[0]}행")
    else:
        df = 기본_데이터_불러오기()
        st.sidebar.info("기본 탑재 데이터 사용 중")

    df = df[[c for c in 기대_열 if c in df.columns]]
    df = 타입_정리(df)

    tabs = st.tabs([
        "0) 데이터 소개",
        "1) 공부시간",
        "2) 성별 비교",
        "3) 인터넷·수면질",
        "4) 공부 방식",
        "5) 공부전략(웹 요약)"
    ])

    with tabs[0]:
        첫페이지_데이터_소개(df)
    with tabs[1]:
        분석_공부시간_시험점수(df)
    with tabs[2]:
        분석_성별_평균(df)
    with tabs[3]:
        분석_인터넷_수면질(df)
    with tabs[4]:
        분석_공부방식_평균(df)
    with tabs[5]:
        공부전략_웹탭(df)

if __name__ == "__main__":
    main()

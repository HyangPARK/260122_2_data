import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import requests
from urllib.parse import quote_plus

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
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

성별_색상 = {"male": "#1f77b4", "female": "#ff7f0e", "other": "#2ca02c"}

# 과목 영어 -> 한글(검색 개선용: 필요시 추가)
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
# 0) 데이터 소개 탭 (귀여운 SVG)
# -----------------------------
def 귀여운_SVG():
    return """
    <div style="display:flex; gap:16px; align-items:center; padding:12px 14px; border-radius:16px; background:#f6f8ff;">
      <svg width="96" height="96" viewBox="0 0 96 96" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="cute mascot">
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
        <div style="font-size:18px; font-weight:700;">시험점수 데이터 분석 놀이터</div>
        <div style="font-size:14px; color:#555; margin-top:4px;">
          데이터 소개 → 분석 → 과목별 영상 찾기 → 영상 요약(공부전략) 순서로 활용해 보세요.
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
    st.write("아래는 데이터에 들어있는 변수들을 차례대로 설명한 것입니다.")
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
# 1) 공부시간 vs 시험점수
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

# -----------------------------
# 2) 성별 평균 비교 (대비색)
# -----------------------------
def 분석_성별_평균(df: pd.DataFrame):
    st.subheader("2) 남/여 시험점수 평균이 다를까요?")

    d = df[df["gender"].isin(["male", "female"])][["gender", "exam_score"]].dropna()

    summary = d.groupby("gender")["exam_score"].agg(["count", "mean", "std"]).reset_index()
    summary["SE"] = summary["std"] / np.sqrt(summary["count"])
    summary = summary.rename(columns={"gender": "성별", "count": "사람수", "mean": "평균점수", "std": "표준편차"})
    st.dataframe(summary, use_container_width=True)

    a = d[d["gender"] == "female"]["exam_score"].astype(float)
    b = d[d["gender"] == "male"]["exam_score"].astype(float)
    t = stats.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())
    st.write(f"- 평균 차이(여-남): **{diff:.3f}점** / p값: **{t.pvalue:.3f}**")

    fig1 = px.box(
        d, x="gender", y="exam_score", color="gender", points="outliers",
        color_discrete_map=성별_색상,
        title="성별 시험점수 분포(박스플롯)"
    )
    fig1.update_layout(xaxis_title="성별", yaxis_title="시험점수", legend_title_text="성별")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        summary, x="성별", y="평균점수", error_y="SE", color="성별",
        color_discrete_map={"male": 성별_색상["male"], "female": 성별_색상["female"]},
        title="성별 평균 시험점수(평균 ± 표준오차)"
    )
    fig2.update_layout(yaxis_title="평균 시험점수", legend_title_text="성별")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 3) 인터넷 이용 vs 수면의 질(좋음/나쁨)
# -----------------------------
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

    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        st.write(f"- p값: **{p:.3f}** (보통 0.05보다 작으면 ‘비율이 다르다’고 보는 경우가 많습니다)")

    long = pct.reset_index().melt(id_vars="internet_access", var_name="수면의 질", value_name="비율(%)")
    fig = px.bar(long, x="internet_access", y="비율(%)", color="수면의 질", barmode="stack",
                 title="인터넷 이용 여부에 따른 수면의 질 비율(좋음/나쁨)")
    fig.update_layout(xaxis_title="", yaxis_title="비율(%)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 4) 공부 방식별 평균 비교
# -----------------------------
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

    fig2 = px.box(d, x="study_method", y="exam_score", points="outliers",
                  title="공부 방식별 점수 분포(박스플롯)")
    fig2.update_layout(xaxis_title="공부 방식", yaxis_title="시험점수")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 유튜브 검색(영상 찾기) + 결과 저장(session_state)
# -----------------------------
def 유튜브_검색_리스트(query: str, max_results: int = 8):
    api_key = st.secrets.get("YOUTUBE_API_KEY", None)

    if not api_key:
        return {"mode": "link", "search_url": f"https://www.youtube.com/results?search_query={quote_plus(query)}", "items": []}

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": api_key,
        "relevanceLanguage": "ko",
        "safeSearch": "strict",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    items = []
    for it in data.get("items", []):
        vid = it["id"]["videoId"]
        title = it["snippet"]["title"]
        channel = it["snippet"]["channelTitle"]
        items.append({"video_id": vid, "title": title, "channel": channel, "url": f"https://www.youtube.com/watch?v={vid}"})
    return {"mode": "api", "items": items}

def 동영상_찾기_탭(df: pd.DataFrame):
    st.subheader("5) 과목별로 '점수 올리는 방법' 동영상 찾기")

    if "last_video_results" not in st.session_state:
        st.session_state["last_video_results"] = []

    col1, col2 = st.columns([2, 1])
    with col1:
        keyword = st.text_input("검색어(예: 공부법, 기출 분석, 시험 대비 등)", value="시험 점수 올리는 공부법")
    with col2:
        course = st.selectbox("과목 선택", sorted(df["course"].dropna().unique().tolist()))

    course_ko_manual = st.text_input("과목 한글명(선택): 비워두면 자동(매핑) 사용", value="")
    course_ko = course_ko_manual.strip() or 과목_한글_매핑.get(course, "")

    if course_ko:
        query = f"{course} {course_ko} {keyword} 공부법 시험공부"
    else:
        query = f"{course} {keyword} 한국어 공부법 시험공부"

    if st.button("동영상 찾기"):
        result = 유튜브_검색_리스트(query=query, max_results=8)
        st.write(f"**검색어:** {query}")

        if result["mode"] == "link":
            st.info("현재 YouTube API 키가 없어 유튜브 검색 링크로 안내합니다.")
            st.link_button("유튜브에서 검색 결과 열기", result["search_url"])
            st.session_state["last_video_results"] = []
            return

        items = result["items"]
        if not items:
            st.warning("검색 결과가 없습니다. 검색어를 바꿔보세요.")
            st.session_state["last_video_results"] = []
            return

        st.session_state["last_video_results"] = items
        st.success(f"{len(items)}개 영상을 찾았습니다. (공부전략 탭에서 요약 가능)")

        for it in items:
            st.markdown(f"- [{it['title']}]({it['url']})  \n  채널: {it['channel']}")
        st.video(items[0]["url"])

# -----------------------------
# 공부전략: 자막 가져오기 + LLM 요약
# -----------------------------
def 유튜브_video_id_추출(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if not s:
        return ""
    # 이미 video_id(11자리)만 넣은 경우
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    # watch?v=...
    m = re.search(r"v=([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    # youtu.be/...
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    # shorts/...
    m = re.search(r"shorts/([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    return ""

def 자막_가져오기(video_id: str, prefer_langs=("ko", "en")) -> str:
    # 가능한 한 한국어 자막 우선, 없으면 영어 등으로 대체
    # (자동번역 자막도 있으면 가져옵니다)
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # 우선순위 언어 찾기
    for lang in prefer_langs:
        try:
            tr = transcript_list.find_transcript([lang])
            lines = tr.fetch()
            return "\n".join([x["text"] for x in lines])
        except Exception:
            pass

    # 직접 자막이 없으면 "생성(자동)" 자막이라도 시도
    try:
        tr = transcript_list.find_manually_created_transcript(transcript_list._TranscriptList__transcripts.keys())
        lines = tr.fetch()
        return "\n".join([x["text"] for x in lines])
    except Exception:
        pass

    # 마지막: 그냥 아무거나 하나
    for tr in transcript_list:
        try:
            lines = tr.fetch()
            return "\n".join([x["text"] for x in lines])
        except Exception:
            continue

    raise NoTranscriptFound("자막을 찾지 못했습니다.")

def LLM_요약하기(transcript_text: str) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    prompt = f"""
당신은 학습코치입니다.
아래는 유튜브 공부법 영상의 자막(텍스트)입니다.
이 내용을 바탕으로 한국어로 쉽고 실용적으로 요약해주세요.

요약 형식(꼭 지켜주세요):
1) 한 줄 핵심(1문장)
2) 핵심 전략 5개(각 1~2문장, 행동으로 옮길 수 있게)
3) 오늘부터 바로 할 체크리스트(3~6개)
4) 주의할 점(2~4개)

자막:
\"\"\"{transcript_text}\"\"\"
"""

    # Responses API 사용(권장)
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text

def 공부전략_탭():
    st.subheader("공부전략: 영상 내용을 요약해서 전략으로 만들기")
    st.write("영상 자막(텍스트)을 가져와서, LLM이 자연스럽게 요약해줍니다.")
    st.caption("자막이 없는 영상은 요약이 어렵습니다. (자막이 있는 영상에서 가장 잘 동작)")

    # 1) 동영상 탭에서 저장된 결과가 있으면 선택 가능
    saved = st.session_state.get("last_video_results", [])
    선택옵션 = ["직접 URL/ID 입력"] + ([f"{i+1}. {x['title']}" for i, x in enumerate(saved)] if saved else [])

    pick = st.selectbox("요약할 영상 선택", options=선택옵션)

    url_or_id = ""
    if pick != "직접 URL/ID 입력" and saved:
        idx = int(pick.split(".")[0]) - 1
        url_or_id = saved[idx]["url"]
        st.write("선택한 영상:")
        st.video(url_or_id)
    else:
        url_or_id = st.text_input("유튜브 URL 또는 video_id(11자리)를 입력하세요", value="")

    video_id = 유튜브_video_id_추출(url_or_id)
    if url_or_id and not video_id:
        st.warning("유튜브 영상 ID를 찾지 못했습니다. URL이 올바른지 확인해주세요.")

    colA, colB = st.columns([1, 1])
    with colA:
        lang_pref = st.radio("자막 언어 우선순위", ["한국어 우선(없으면 영어)", "영어 우선(없으면 한국어)"], horizontal=True)
    with colB:
        show_transcript = st.checkbox("자막 원문도 보기", value=False)

    prefer = ("ko", "en") if "한국어" in lang_pref else ("en", "ko")

    if st.button("자막 가져오고 요약하기", disabled=(not video_id)):
        try:
            with st.spinner("자막을 가져오는 중..."):
                transcript = 자막_가져오기(video_id, prefer_langs=prefer)

            if show_transcript:
                st.subheader("자막(원문)")
                st.text_area("자막 텍스트", transcript, height=260)

            with st.spinner("요약을 만드는 중..."):
                summary = LLM_요약하기(transcript)

            st.subheader("요약 결과(공부 전략)")
            st.write(summary)

        except (TranscriptsDisabled, NoTranscriptFound):
            st.error("이 영상은 자막이 없거나 공개되지 않아 요약이 어렵습니다. 다른 영상을 선택해 주세요.")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# -----------------------------
# 메인
# -----------------------------
def main():
    st.title("시험점수 데이터 분석 웹앱")

    # 업로드 데이터(있으면 사용)
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

    tabs = st.tabs(["0) 데이터 소개", "1) 공부시간", "2) 성별 비교", "3) 인터넷·수면질", "4) 공부 방식", "5) 동영상 찾기", "6) 공부전략(요약)"])

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
        동영상_찾기_탭(df)
    with tabs[6]:
        공부전략_탭()

if __name__ == "__main__":
    main()


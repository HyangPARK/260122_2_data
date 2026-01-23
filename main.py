import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import requests
from urllib.parse import quote_plus

st.set_page_config(page_title="시험점수 데이터 분석", layout="wide")

# -----------------------------
# 기본 설정
# -----------------------------
기본_데이터_파일 = "ES_Pre.csv"

기대_열 = [
    "student_id", "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty", "exam_score"
]

수치형_열 = ["student_id", "age", "study_hours", "class_attendance", "sleep_hours", "exam_score"]

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
# 간단한 데이터 점검
# -----------------------------
def 데이터_요약(df: pd.DataFrame):
    st.subheader("데이터 점검")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.write("**데이터 크기**")
        st.write({"행(학생 수)": int(df.shape[0]), "열(변수 수)": int(df.shape[1])})

        st.write("**결측치(비어있는 값) 개수**")
        na = df.isna().sum().sort_values(ascending=False)
        st.dataframe(na)

    with c2:
        st.write("**수치형 요약(최소/평균/최대 등)**")
        st.dataframe(df[["age","study_hours","class_attendance","sleep_hours","exam_score"]].describe().T)

    st.caption("※ 업로드 데이터가 있으면 자동으로 업로드 데이터로 분석됩니다.")

# -----------------------------
# 분석 1: 공부시간 vs 시험점수
# -----------------------------
def 분석_공부시간_시험점수(df: pd.DataFrame):
    st.subheader("1) 공부시간이 늘면 시험점수가 올라가나요?")

    d = df[["study_hours","exam_score"]].dropna()
    x = d["study_hours"].astype(float).values
    y = d["exam_score"].astype(float).values

    r, p = stats.pearsonr(x, y)
    reg = stats.linregress(x, y)

    c1, c2, c3 = st.columns(3)
    c1.metric("상관계수(r)", f"{r:.3f}")
    c2.metric("p값", f"{p:.3f}")
    c3.metric("대략적인 증가폭", f"+{reg.slope:.2f}점/1시간")

    st.write("- r이 **0에 가까우면 관계가 약하고**, **1에 가까우면 함께 증가하는 경향**이 큽니다.")
    st.write("- 여기서는 **공부시간이 늘수록 시험점수가 높아지는 경향**이 보입니다.")

    fig = px.scatter(
        d, x="study_hours", y="exam_score",
        opacity=0.35,
        title="공부시간과 시험점수 (점 + 추세선)"
    )
    xs = np.linspace(x.min(), x.max(), 200)
    ys = reg.intercept + reg.slope * xs
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="추세선"))
    fig.update_layout(xaxis_title="공부시간(시간)", yaxis_title="시험점수")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 분석 2: 성별 평균 비교
# -----------------------------
def 분석_성별_평균(df: pd.DataFrame):
    st.subheader("2) 남/여 시험점수 평균이 다를까요?")

    d = df[df["gender"].isin(["male","female"])][["gender","exam_score"]].dropna()

    summary = d.groupby("gender")["exam_score"].agg(["count","mean","std"]).reset_index()
    summary["SE"] = summary["std"] / np.sqrt(summary["count"])
    summary = summary.rename(columns={"gender":"성별","count":"사람수","mean":"평균점수","std":"표준편차"})
    st.dataframe(summary)

    a = d[d["gender"]=="female"]["exam_score"].astype(float)
    b = d[d["gender"]=="male"]["exam_score"].astype(float)
    t = stats.ttest_ind(a, b, equal_var=False)

    diff = float(a.mean() - b.mean())
    st.write(f"- 평균 차이(여-남): **{diff:.3f}점**")
    st.write(f"- p값: **{t.pvalue:.3f}** (보통 p값이 0.05보다 작으면 '차이가 있다'고 보는 경우가 많습니다)")

    fig1 = px.box(d, x="gender", y="exam_score", points="outliers",
                  title="성별 시험점수 분포(박스플롯)")
    fig1.update_layout(xaxis_title="성별(male/female)", yaxis_title="시험점수")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        summary, x="성별", y="평균점수",
        error_y="SE",
        title="성별 평균 시험점수(평균 ± 표준오차)"
    )
    fig2.update_layout(yaxis_title="평균 시험점수")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 분석 3: 인터넷 이용 vs 수면의 질(좋음/나쁨 비율)
# -----------------------------
def 분석_인터넷_수면질(df: pd.DataFrame):
    st.subheader("3) 인터넷 이용 여부에 따라 수면의 질이 다를까요?")

    d = df[["internet_access","sleep_quality"]].dropna()
    d2 = d[d["sleep_quality"].isin(["good","poor"])].copy()

    ct = pd.crosstab(d2["internet_access"], d2["sleep_quality"])
    ct = ct.rename(index={"yes":"인터넷 사용", "no":"인터넷 미사용"},
                   columns={"good":"좋음", "poor":"나쁨"})

    st.write("**교차표(사람 수)**")
    st.dataframe(ct)

    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    st.write("**비율(%)**")
    st.dataframe(pct)

    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        c1, c2 = st.columns(2)
        c1.metric("p값", f"{p:.3f}")
        c2.metric("카이제곱", f"{chi2:.3f}")
        st.caption("※ p값이 0.05보다 작으면 '분포가 다르다'고 해석하는 경우가 많습니다.")

    long = pct.reset_index().melt(id_vars="internet_access", var_name="수면의 질", value_name="비율(%)")
    fig = px.bar(
        long, x="internet_access", y="비율(%)", color="수면의 질",
        barmode="stack",
        title="인터넷 이용 여부에 따른 수면의 질 비율(좋음/나쁨)"
    )
    fig.update_layout(xaxis_title="", yaxis_title="비율(%)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 4) 과목별 '점수 올리는 방법' 동영상 찾기(YouTube 검색)
# -----------------------------
def 유튜브_검색_리스트(query: str, max_results: int = 6):
    api_key = st.secrets.get("YOUTUBE_API_KEY", None)

    if api_key:
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
            items.append({
                "title": title,
                "channel": channel,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "video_id": vid
            })
        return {"mode": "api", "items": items}
    else:
        return {"mode": "link", "search_url": f"https://www.youtube.com/results?search_query={quote_plus(query)}", "items": []}

def 분석_과목별_동영상(df: pd.DataFrame):
    st.subheader("4) 과목별로 '점수 올리는 방법' 동영상 찾기")

    st.write("원하는 키워드를 입력하면, 과목과 함께 검색해 관련 동영상을 찾습니다.")
    st.caption("YouTube API 키가 있으면 앱에서 목록/미리보기가 더 깔끔합니다. (없어도 검색 링크는 제공됩니다)")

    col1, col2 = st.columns([2, 1])
    with col1:
        keyword = st.text_input("검색어(예: 공부법, 기출 분석, 시험 대비 등)", value="시험 점수 올리는 공부법")
    with col2:
        course = st.selectbox("과목 선택", sorted(df["course"].dropna().unique().tolist()))

    query = f"{course} {keyword}"

    if st.button("동영상 찾기"):
        try:
            result = 유튜브_검색_리스트(query=query, max_results=8)

            st.write(f"**검색어:** {query}")

            if result["mode"] == "link":
                st.info("현재 YouTube API 키가 설정되어 있지 않아, 유튜브 검색 링크로 안내합니다.")
                st.link_button("유튜브에서 검색 결과 열기", result["search_url"])
                st.write("원하시면 Streamlit Cloud의 Secrets에 `YOUTUBE_API_KEY`를 추가해 API 방식으로도 동작하게 할 수 있습니다.")
            else:
                items = result["items"]
                if not items:
                    st.warning("검색 결과가 없습니다. 검색어를 바꿔보세요.")
                    return

                st.success(f"{len(items)}개 영상을 찾았습니다.")
                st.write("**추천 목록**")
                for it in items:
                    st.markdown(f"- [{it['title']}]({it['url']})  \n  채널: {it['channel']}")

                st.write("**미리보기(첫 번째 영상)**")
                st.video(items[0]["url"])

        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {e}")

# -----------------------------
# 메인
# -----------------------------
def main():
    st.title("시험점수 데이터 분석 웹앱 (한글/쉬운 용어)")

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

    tabs = st.tabs(["데이터 점검", "1) 공부시간", "2) 성별 비교", "3) 인터넷·수면질", "4) 동영상 찾기"])

    with tabs[0]:
        데이터_요약(df)

    with tabs[1]:
        분석_공부시간_시험점수(df)

    with tabs[2]:
        분석_성별_평균(df)

    with tabs[3]:
        분석_인터넷_수면질(df)

    with tabs[4]:
        분석_과목별_동영상(df)

    st.divider()
    st.caption("배포: GitHub에 app.py, requirements.txt, Exam_Score_Prediction.csv를 올리고 Streamlit Cloud에서 app.py를 실행 파일로 지정하세요.")
    st.caption("YouTube API 사용: Streamlit Cloud > App settings > Secrets에 YOUTUBE_API_KEY를 추가하면 됩니다.")

if __name__ == "__main__":
    main()

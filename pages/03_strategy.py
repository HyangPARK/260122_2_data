import streamlit as st
import trafilatura
import requests
from urllib.parse import quote_plus
from openai import OpenAI

st.set_page_config(page_title="공부전략(웹/동영상)", layout="wide")
st.title("03) 공부전략 (웹페이지 요약 + 동영상 탐색)")

def extract_main_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError("웹페이지를 가져오지 못했습니다(접속 제한/차단 가능).")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text or len(text.strip()) < 200:
        raise RuntimeError("본문 추출이 충분하지 않습니다(짧거나 추출 실패).")
    return text

def llm_summarize_web(text: str, user_goal: str = "", source_url: str = "") -> str:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    trimmed = text.strip()
    if len(trimmed) > 20000:
        trimmed = trimmed[:20000]

    prompt = f"""
당신은 학습코치입니다.
아래 웹페이지 본문을 읽고, 한국어로 쉽고 실용적으로 '공부 전략' 요약을 만들어주세요.

출처 URL: {source_url}
사용자 의도/목표: {user_goal}

형식(꼭 지켜주세요):
1) 한 줄 핵심(1문장)
2) 핵심 전략 5개(각 1~2문장, 바로 실행 가능)
3) 오늘부터 체크리스트(3~6개)
4) 주의할 점(2~4개)

본문:
\"\"\"{trimmed}\"\"\"
"""
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text

def youtube_search(query: str, max_results: int = 10):
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
        items.append({"title": title, "channel": channel, "url": f"https://www.youtube.com/watch?v={vid}"})
    return {"mode": "api", "items": items}

tab_web, tab_video = st.tabs(["웹페이지 요약", "동영상 탐색(선택)"])

with tab_web:
    st.subheader("웹페이지 요약")
    q = st.text_input("검색어(구글/네이버에서 찾을 주제)", value="수학 성적 올리는 공부법 오답노트")

    c1, c2 = st.columns(2)
    with c1:
        st.link_button("구글에서 검색", f"https://www.google.com/search?q={quote_plus(q)}")
    with c2:
        st.link_button("네이버에서 검색", f"https://search.naver.com/search.naver?query={quote_plus(q)}")

    st.caption("검색 링크로 글을 찾은 뒤, 요약할 페이지의 URL을 아래에 붙여넣으세요.")
    url = st.text_input("요약할 웹페이지 URL", value="")
    show_text = st.checkbox("본문 텍스트도 보기", value=False)

    if st.button("웹페이지 요약하기", disabled=(not url.strip())):
        try:
            with st.spinner("본문 추출 중..."):
                text = extract_main_text(url.strip())
            if show_text:
                preview = text if len(text) <= 12000 else text[:12000] + "\n...(이하 생략)"
                st.text_area("추출된 본문(일부)", preview, height=260)
            with st.spinner("요약 생성 중..."):
                summary = llm_summarize_web(text, user_goal=q, source_url=url.strip())
            st.subheader("요약 결과(공부 전략)")
            st.write(summary)
        except Exception as e:
            st.error(f"오류: {e}")
            st.info("일부 사이트는 본문 추출이 어려울 수 있어요. 다른 URL로 시도해 보세요.")

with tab_video:
    st.subheader("동영상 탐색")
    vq = st.text_input("유튜브 검색어", value="오답노트 작성법")
    max_results = st.selectbox("결과 수", [5, 8, 10, 15], index=2)

    if st.button("유튜브에서 찾기"):
        try:
            result = youtube_search(vq, max_results=max_results)
            if result["mode"] == "link":
                st.info("YouTube API 키가 없어서 앱 안에서 목록을 보여주기 어렵습니다.")
                st.link_button("유튜브 검색 결과 열기", result["search_url"])
                st.session_state["yt_items"] = []
            else:
                st.session_state["yt_items"] = result["items"]
        except Exception as e:
            st.error(f"유튜브 검색 오류: {e}")
            st.session_state["yt_items"] = []

    items = st.session_state.get("yt_items", [])
    if items:
        for it in items:
            st.markdown(f"- [{it['title']}]({it['url']})  \n  채널: {it['channel']}")
        st.video(items[0]["url"])

    st.caption("영상 요약은 ‘자막 없음’ 문제가 많아서, 이 앱은 웹페이지 요약을 중심 기능으로 제공합니다.")

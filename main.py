import streamlit as st

st.set_page_config(page_title="Exam Score Analytics + ML + AI Coach", layout="wide")

st.title("시험점수 분석 · 과목별 상관관계 · ML 예측 · AI 학습코치")
st.write("""
이 앱은 **Exam Score Prediction 데이터**를 기반으로 아래 기능을 제공합니다.

- **01 데이터 소개**: 출처/변수/샘플
- **02 데이터 정리**: 결측치/이상치/정리 결과
- **03 분석(상관관계)**: 전체 + **과목별** 상관/그래프
- **04 ML 예측**: 전체 모델 + **과목별 모델** + what-if 시뮬레이션
- **05 AI 학습코치**: ML 결과를 바탕으로 쉬운 공부 전략 생성(LLM)
""")

st.info("왼쪽 메뉴에서 페이지를 선택하세요.")

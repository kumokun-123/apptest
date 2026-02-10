"""
🎬 マルチエージェント映画推薦システム (比較用)
Planner / Respond / Recommend Agent構成
"""

import streamlit as st
import json
import time
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# ========== 設定 ==========
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    OPENAI_API_KEY = "sk-..."

MODEL_NAME = "gpt-5-mini"
TEMPERATURE = 1

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    timeout=30.0,
    max_retries=2,
)

# ========== エージェント定義 ==========
# (PlannerAgent, RespondAgent, RecommendAgent クラスは変更なし)
class PlannerAgent:
    def run(self, user_input, history):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            あなたは対話の進行管理を行うPlanner Agentです。
            ユーザーの直前の発言を分析し、次の2つのアクションのうちどちらを行うべきか決定してください。
            1. "answer": ユーザーがシステムに対して質問をしている場合。
            2. "ask_more": ユーザーが自分の好みを伝えている、または前回の質問に回答している場合。
            JSON形式で出力してください: {{ "action": "answer" または "ask_more", "reason": "判定理由" }}
            """),
            ("human", "対話履歴:\n{history}\n\nユーザーの最新発言: {input}")
        ])
        chain = prompt | llm | JsonOutputParser()
        return chain.invoke({"input": user_input, "history": history})

class RespondAgent:
    def ask_guidance(self, liked, disliked, history):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            あなたは映画推薦のためのインタビュアー（Questioner）です。
            ユーザーの「好きな映画」「嫌いな映画」および「これまでの対話」に基づいて、
            おすすめの映画を絞り込むための**短い質問を1つだけ**してください。
            """),
            ("human", """
            好きな映画: {liked}
            嫌いな映画: {disliked}
            これまでの対話: {history}
            次の質問を作成してください。
            """)
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"liked": liked, "disliked": disliked, "history": history})

    def answer_user(self, user_query, history):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            あなたは映画に詳しいアシスタント（Answerer）です。
            ユーザーの質問に対して、親切かつ簡潔に回答してください。
            回答した後、さりげなくユーザーの好みをさらに聞くような一言を添えてください。
            """),
            ("human", """
            対話履歴: {history}
            ユーザーの質問: {query}
            """)
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"query": user_query, "history": history})

class RecommendAgent:
    def run(self, liked, disliked, history):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            あなたは熟練の映画コンシェルジュ（Recommender）です。
            これまでのユーザーとの対話、好みの映画、嫌いな映画を総合的に分析し、
            **ベストな映画を1本だけ**推薦してください。
            JSON形式で出力してください:
            {{
                "movie_title": "映画タイトル",
                "year": "公開年",
                "reason": "詳細な推薦理由",
                "genre": "ジャンル",
                "match_point": "ポイント"
            }}
            """),
            ("human", """
            好きな映画: {liked}
            嫌いな映画: {disliked}
            対話ログ:
            {history}
            最高の1本を選んでください。
            """)
        ])
        chain = prompt | llm | JsonOutputParser()
        return chain.invoke({"liked": liked, "disliked": disliked, "history": history})

# ========== メインアプリ ==========

def main():
    # ★修正: set_page_config は削除 (app.pyで設定済み)
    
    # CSS調整
    st.markdown("""
    <style>
    .agent-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .planner { background-color: #e3f2fd; border-left: 5px solid #2196f3; }
    .recommender { background-color: #e8f5e9; border-left: 5px solid #4caf50; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤖 マルチエージェント映画推薦 (比較実験)")
    st.caption("Planner / Respond / Recommend Agent Architecture")

    # セッション状態の初期化
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0  
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "liked_movie" not in st.session_state:
        st.session_state.liked_movie = ""
    if "disliked_movie" not in st.session_state:
        st.session_state.disliked_movie = ""
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None

    # エージェントインスタンス化
    planner = PlannerAgent()
    respond = RespondAgent()
    recommender = RecommendAgent()

    # --- ステップ0: 初期入力 (好きな映画/嫌いな映画) ---
    if st.session_state.turn_count == 0:
        st.markdown("### スタート: あなたの基準を教えてください")
        with st.form("init_form"):
            col1, col2 = st.columns(2)
            l_mov = col1.text_input("好きな映画を1つ", placeholder="例: インターステラー")
            d_mov = col2.text_input("嫌いな映画を1つ", placeholder="例: (特になければ空欄でも可)")
            
            submitted = st.form_submit_button("対話を開始する")
            if submitted and l_mov:
                st.session_state.liked_movie = l_mov
                st.session_state.disliked_movie = d_mov
                
                with st.spinner("エージェントが情報を分析中..."):
                    initial_q = respond.ask_guidance(l_mov, d_mov, "初期状態")
                    st.session_state.chat_history.append({"role": "assistant", "content": initial_q, "agent": "Respond (Guidance)"})
                    st.session_state.turn_count = 1
                st.rerun()

    # --- ステップ1-4: 対話ループ ---
    elif 1 <= st.session_state.turn_count < 5:
        progress = st.session_state.turn_count / 5
        st.progress(progress, text=f"対話フェーズ ({st.session_state.turn_count}/5)")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if "agent" in msg:
                    st.caption(f"🔧 {msg['agent']}")
                st.markdown(msg["content"])

        if user_input := st.chat_input("回答または質問を入力..."):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.status("Planner Agentが思考中...", expanded=True) as status:
                st.write("意図分析を実行中...")
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
                plan = planner.run(user_input, history_text)
                
                action = plan.get("action", "ask_more")
                reason = plan.get("reason", "")
                st.markdown(f"""
                <div class="agent-box planner">
                <b>Planner Decision:</b> {action}<br>
                <small>理由: {reason}</small>
                </div>
                """, unsafe_allow_html=True)
                
                response_content = ""
                agent_type = ""
                
                if action == "answer":
                    st.write("Respond Agent (Answer) を呼び出し中...")
                    response_content = respond.answer_user(user_input, history_text)
                    agent_type = "Respond (Answer)"
                else:
                    st.write("Respond Agent (Guidance) を呼び出し中...")
                    response_content = respond.ask_guidance(
                        st.session_state.liked_movie, 
                        st.session_state.disliked_movie, 
                        history_text
                    )
                    agent_type = "Respond (Guidance)"
                
                status.update(label="完了", state="complete", expanded=False)

            with st.chat_message("assistant"):
                st.caption(f"🔧 {agent_type}")
                st.markdown(response_content)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_content,
                "agent": agent_type
            })

            st.session_state.turn_count += 1
            if st.session_state.turn_count >= 5:
                time.sleep(1)
                st.rerun()

    # --- ステップ5: 推薦 ---
    elif st.session_state.turn_count >= 5:
        st.success("🎉 情報収集が完了しました。Recommend Agentが起動します。")
        
        if not st.session_state.recommendation:
            with st.spinner("Recommend Agentが最適な映画を選定中..."):
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
                rec_result = recommender.run(
                    st.session_state.liked_movie,
                    st.session_state.disliked_movie,
                    history_text
                )
                st.session_state.recommendation = rec_result
        
        rec = st.session_state.recommendation
        if rec:
            st.markdown(f"""
            <div class="agent-box recommender">
                <h2>🎬 推薦: {rec.get('movie_title')} ({rec.get('year')})</h2>
                <p><b>ジャンル:</b> {rec.get('genre')}</p>
                <hr>
                <h4>💡 推薦理由</h4>
                <p>{rec.get('reason')}</p>
                <p><b>🎯 マッチポイント:</b> {rec.get('match_point')}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("対話履歴を確認"):
                for msg in st.session_state.chat_history:
                    st.text(f"{msg['role']} ({msg.get('agent', '')}): {msg['content']}")

            # ★修正: リセット時に experiment_mode を保持する
            if st.button("最初からやり直す"):
                keys_to_delete = [k for k in st.session_state.keys() if k != "experiment_mode"]
                for key in keys_to_delete:
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    # 単体実行時のみconfig設定
    st.set_page_config(page_title="Agent Comparison System (B)", page_icon="🤖")
    main()
"""
🎬 インテリジェント映画推薦システム - DeepSeek APIバージョン
プロキシ不要、.envファイル不要、ワンクリック実行
"""

import streamlit as st
import json
import re
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import time

# ========== ステップ1：OpenAI API設定 ==========
# Streamlit Secretsからキーを取得（安全な運用のため）
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    # ローカルテスト用（必要に応じて書き換えてください、アップロード時は削除推奨）
    OPENAI_API_KEY = "sk-..." 

# モデル設定
# コストを抑えるなら "gpt-4o-mini"、性能重視なら "gpt-4o"
MODEL_NAME = "gpt-5-mini" 
TEMPERATURE = 1
TIMEOUT = 60.0

# ========== ステップ2：LangChain初期化 ==========
print("🚀 OpenAI API接続初期化中...")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

    # OpenAI LLM初期化
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        max_retries=2,
        # base_url は削除（OpenAI公式を使うため不要）
    )

    print("✅ OpenAI API接続初期化成功")

except ImportError as e:
    st.error(f"❌ 依存関係のインポートに失敗: {e}")
    st.info("依存関係をインストールしてください: pip install langchain-openai langchain-core")
    st.stop()
except Exception as e:
    st.error(f"❌ OpenAI初期化に失敗: {str(e)}")
    st.stop()


# ========== ステップ3：サイドバー状態表示 ==========
def show_sidebar():
    """サイドバーを表示"""
    with st.sidebar:
        st.title("🎬 映画推薦システム")
        st.markdown(f"**AIエンジン**: DeepSeek")
        st.markdown(f"**モデル**: {MODEL_NAME}")
        st.markdown(f"**キーステータス**: ✅ 設定済み")

        st.markdown("---")

        # 進捗インジケーター
        if 'step' in st.session_state:
            steps = ["嗜好入力", "ユーザペルソナ生成", "定量的分析", "対話型質問", "推薦取得"]
            step_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 5}  # ステップマッピング修正
            current_step_num = step_mapping.get(st.session_state.step, 1)

            for i, step in enumerate(steps, 1):
                if i < current_step_num:
                    st.markdown(f"✅ {step}")
                elif i == current_step_num:
                    st.markdown(f"▶️ {step}")
                else:
                    st.markdown(f"○ {step}")

        st.markdown("---")

        # 統計情報表示
        if 'liked_movies' in st.session_state and st.session_state.liked_movies:
            st.caption(f"🎬 選択済み映画: {len(st.session_state.liked_movies)}作品")

        if 'questions_asked' in st.session_state and st.session_state.questions_asked:
            st.caption(f"❓ 回答済み質問: {len(st.session_state.questions_asked)}個")

        if 'user_profiles' in st.session_state and st.session_state.user_profiles:
            st.caption(f"👤 残りユーザペルソナ: {len(st.session_state.user_profiles)}個")

        st.markdown("---")

        # クイック操作ボタン
        if st.button("🔄 アプリ再起動", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        # ステップナビゲーション
        st.markdown("### ステップナビゲーション")
        steps_options = {
            "1. 映画嗜好入力": 1,
            "2. ユーザペルソナ生成": 2,
            "3. 定量的分析": 3,
            "4. 対話型質問": 4,
            "6. 推薦生成": 6,
            "7. よくある質問": 7
        }

        current_step = st.session_state.get('step', 1)
        for step_name, step_num in steps_options.items():
            if step_num != current_step and step_num <= 7:
                if st.button(step_name, key=f"nav_{step_num}", use_container_width=True):
                    st.session_state.step = step_num
                    st.rerun()


# ========== ステップ4：セッション状態初期化 ==========
def init_session_state():
    """すべてのセッション状態変数を初期化"""
    default_states = {
        'step': 1,
        'user_profiles': [],
        'quantitative_analysis': [],
        'liked_movies': [],
        'disliked_movies': [],
        'questions_asked': [],
        'answers_given': [],
        'current_question': "",
        'current_scale': "",
        'final_profile': None,
        'recommendation': None,
        'qa_pairs': [],
        'elimination_history': [],
        'start_time': datetime.now(),
        'processing': False,
        'profiles_generated': False,
        'analysis_completed': False,
        'step_changed': False,
        'api_call_count': 0,  # 新規：API呼び出し回数記録
        'last_api_call': None,  # 新規：最終API呼び出し時間記録
        'elimination_completed': False,
        'used_scale_indices': [],
        'current_options': {},  # ★追加: 選択肢の内容(A/B)を保存
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ========== 安全なAPI呼び出し関数 ==========
def safe_llm_call(chain, inputs, max_retries=2):
    """安全なLLM呼び出し関数、リトライメカニズムを含む"""
    for attempt in range(max_retries):
        try:
            st.session_state.api_call_count += 1
            st.session_state.last_api_call = datetime.now()

            result = chain.invoke(inputs)

            # 遅延追加、API制限回避
            time.sleep(0.5)
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"API呼び出し失敗、再試行中... ({attempt + 1}/{max_retries})")
                time.sleep(2)  # 2秒待機後再試行
            else:
                st.error(f"API呼び出し失敗: {str(e)}")
                raise


# ========== ステップ5：アプリ機能関数 ==========

# ステップ1：映画嗜好入力
def step1_input_movies():
    st.title("🎬 インテリジェント映画推薦システム")
    st.markdown("### ステップ 1: 映画の嗜好を教えてください")

    # サンプル映画推薦
    with st.expander("💡 入力する映画がわからない？クリックしてサンプルを表示", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**クラシック映画**:")
            st.markdown("• ショーシャンクの空に")
            st.markdown("• ゴッドファーザー")
            st.markdown("• フォレスト・ガンプ")
            st.markdown("• インターステラー")
            st.markdown("• インセプション")
        with col2:
            st.markdown("**様々なジャンル**:")
            st.markdown("• コメディ：きっと、うまくいく")
            st.markdown("• アニメ：リメンバー・ミー")
            st.markdown("• SF：ブレードランナー2049")
            st.markdown("• ドラマ：ライフ・イズ・ビューティフル")
            st.markdown("• アクション：マトリックス")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("好きな映画")
        liked_input = st.text_area(
            "好きな映画を入力してください（1行1作品またはカンマ区切り）:",
            height=150,
            placeholder="例:\nショーシャンクの空に\nインセプション\nフォレスト・ガンプ\nインターステラー",
            key="liked_input"
        )
        st.caption("最低1作品、最大10作品")

    with col2:
        st.subheader("嫌いな映画")
        disliked_input = st.text_area(
            "嫌いな映画を入力してください（オプション）:",
            height=150,
            placeholder="例:\nトランスフォーマー/最後の騎士王\nトワイライト〜初恋〜\n\n（オプション、システムが嗜好をより理解するのに役立ちます）",
            key="disliked_input"
        )

    if st.button("🚀 映画嗜好分析を開始", type="primary", use_container_width=True):
        if liked_input.strip():
            # 入力処理
            liked_movies = []
            for item in re.split(r'[,\n]', liked_input):
                clean_item = item.strip()
                if clean_item and clean_item not in liked_movies:
                    liked_movies.append(clean_item)

            disliked_movies = []
            if disliked_input.strip():
                for item in re.split(r'[,\n]', disliked_input):
                    clean_item = item.strip()
                    if clean_item and clean_item not in disliked_movies:
                        disliked_movies.append(clean_item)

            if len(liked_movies) > 10:
                liked_movies = liked_movies[:10]
                st.info(f"好きな映画の上位10作品を選択しました")

            if len(disliked_movies) > 5:
                disliked_movies = disliked_movies[:5]
                st.info(f"嫌いな映画の上位5作品を選択しました")

            st.session_state.liked_movies = liked_movies
            st.session_state.disliked_movies = disliked_movies

            st.success(f"✅ {len(liked_movies)}作品の好きな映画を記録しました")
            if disliked_movies:
                st.success(f"✅ {len(disliked_movies)}作品の嫌いな映画を記録しました")

            # ユーザー選択映画表示
            with st.expander("📋 入力した映画リストを表示", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**好きな映画:**")
                    for movie in liked_movies:
                        st.markdown(f"• {movie}")
                with col2:
                    if disliked_movies:
                        st.markdown("**嫌いな映画:**")
                        for movie in disliked_movies:
                            st.markdown(f"• {movie}")
                    else:
                        st.markdown("**嫌いな映画:** なし")

            st.session_state.step = 2
            st.session_state.processing = False  # 処理状態リセット
            time.sleep(1)  # ユーザーが結果を確認する時間
            st.rerun()
        else:
            st.warning("⚠️ 少なくとも1作品の好きな映画を入力してください")


# ステップ2：ユーザペルソナ生成
def step2_generate_profiles():
    st.title("🎬 インテリジェント映画推薦システム")
    st.markdown("### ステップ 2: ユーザペルソナ分析")

    # ユーザー選択映画表示
    if st.session_state.liked_movies:
        st.info(
            f"**好きな映画**: {', '.join(st.session_state.liked_movies[:3])}{'...' if len(st.session_state.liked_movies) > 3 else ''}")
    if st.session_state.disliked_movies:
        st.info(
            f"**嫌いな映画**: {', '.join(st.session_state.disliked_movies[:3])}{'...' if len(st.session_state.disliked_movies) > 3 else ''}")
    # すでにペルソナ生成済みの場合、結果表示
    if st.session_state.profiles_generated and st.session_state.user_profiles:
        st.success("✅ ユーザペルソナ生成完了")

        # ユーザペルソナ数表示
        st.markdown(f"**{len(st.session_state.user_profiles)}個の可能なユーザペルソナを生成:**")

        for profile in st.session_state.user_profiles:
            with st.expander(f"👤 ユーザペルソナ {profile['profile_id']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**基本情報**")
                    st.markdown(f"{profile.get('basic_info', 'N/A')}")
                with col2:
                    st.markdown("**性格特徴**")
                    st.markdown(f"{profile.get('personality', 'N/A')}")

                st.markdown("**価値観と嗜好**")
                st.markdown(f"{profile.get('values', 'N/A')}")

        # 次へ進むボタン
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 定量的分析を開始", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.session_state.analysis_completed = False  # 定量的分析を再実行可能に
                st.rerun()
        with col2:
            if st.button("🔄 ペルソナ再生成", type="secondary", use_container_width=True):
                st.session_state.user_profiles = []
                st.session_state.profiles_generated = False
                st.session_state.processing = False
                st.rerun()
        return

    # ユーザペルソナ生成ボタン
    if not st.session_state.processing and st.button("🔍 ユーザペルソナ生成", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.rerun()

    # 生成ロジック処理
    if st.session_state.processing:
        with st.spinner("映画嗜好を分析中、ユーザペルソナを生成..."):
            try:
                profile_template = ChatPromptTemplate.from_messages([
                    ("system", """あなたは熟練の映画愛好家およびユーザー行動分析の専門家です。ユーザーの映画嗜好に基づいて、5つの可能なユーザペルソナを生成してください。
                     各ユーザペルソナには以下を含めてください：
                     1. 基本ユーザー情報（年齢、性別、職業、教育背景など）
                     2. 性格特徴分析
                     3. 価値観と審美嗜好

                     ペルソナが多様で、異なる可能性のある人々をカバーすることを確認してください。"""),
                    ("human", """
                    ユーザーの好きな映画: {liked_movies}
                    ユーザーの嫌いな映画: {disliked_movies}

                    JSON配列形式で出力し、各ペルソナオブジェクトに以下の3つのキーを含めてください：
                    1. "basic_info": 基本ユーザー情報予測
                    2. "personality": ユーザー性格分析予測
                    3. "values": ユーザー価値観と審美嗜好分析

                    JSON配列のみを返し、他の説明文は一切含めないでください。
                    """)
                ])

                # パーサーとチェーン作成
                parser = JsonOutputParser()
                chain = profile_template | llm | parser

                # チェーン呼び出し
                result = safe_llm_call(chain, {
                    "liked_movies": "\n".join([f"- {movie}" for movie in st.session_state.liked_movies]),
                    "disliked_movies": "\n".join([f"- {movie}" for movie in
                                                  st.session_state.disliked_movies]) if st.session_state.disliked_movies else "なし"
                })

                # 結果処理、profile_id追加
                if isinstance(result, list):
                    profiles = []
                    for i, profile in enumerate(result[:5]):  # 5つのみ取得を確認
                        profile["profile_id"] = i + 1
                        profiles.append(profile)

                    st.session_state.user_profiles = profiles
                    st.session_state.profiles_generated = True
                    st.session_state.processing = False

                    st.success("✅ ユーザペルソナを生成しました！")
                    st.rerun()

                else:
                    st.error("ユーザペルソナ生成時の返却形式が不正です。再試行してください。")
                    st.session_state.processing = False
                    st.rerun()

            except Exception as e:
                st.error(f"❌ ユーザペルソナ生成中にエラー: {str(e)}")
                st.session_state.processing = False

                # 代替案提供
                if st.button("サンプルデータで続行", type="secondary"):
                    st.session_state.user_profiles = [
                        {
                            "profile_id": 1,
                            "basic_info": "25-35歳、男性、テクノロジー業界従事者、大学卒以上",
                            "personality": "論理的思考が強く、複雑なナラティブを好み、論理的厳密さを追求",
                            "values": "映画の思想の深さと物語構造を重視"
                        },
                        {
                            "profile_id": 2,
                            "basic_info": "30-40歳、女性、文化教育業界、修士号",
                            "personality": "感情的で繊細、感情表現と人物造形を重視",
                            "values": "映画の感情的な共鳴と芸術的価値を重視"
                        }
                    ]
                    st.session_state.profiles_generated = True
                    st.session_state.processing = False
                    st.rerun()


# ステップ3：定量的分析
def step3_quantitative_analysis():
    st.title("🎬 インテリジェント映画推薦システム")
    st.markdown("### ステップ 3: ユーザペルソナの心理的分析")

    if not st.session_state.user_profiles:
        st.error("❌ ユーザペルソナデータが見つかりません。前のステップに戻って再生成してください")
        if st.button("前のステップに戻る", type="secondary"):
            st.session_state.step = 2
            st.rerun()
        return

    # 尺度定義
    quantitative_scales = [
        "1. 認知的複雑性 (SCC): [開放性(知)] 知的好奇心・自律",
        "2. 情動的強度 (ASI): [外向性] 刺激希求・快楽",
        "3. 道徳的整合性 (MVA): [協調性] 共感・調和・自己超越",
        "4. 心理的安全性 (PSF): [神経症傾向] 不安回避・安全・伝統",
        "5. 美的・抽象性 (AAO): [開放性(美)] 美的感受性・美",
        "6. 社会的密度 (SRD): [外向性・協調性] 社交性・慈善",
        "7. 実用的リアリズム (PRI): [誠実性] 現実・秩序・真理"
    ]

    # 分析完了表示
    if st.session_state.analysis_completed and st.session_state.quantitative_analysis:
        st.success("✅ 心理的・価値観分析が完了しました")

        tabs = st.tabs([f"ペルソナ {i + 1}" for i in range(len(st.session_state.user_profiles))])

        for idx, (analysis, tab) in enumerate(zip(st.session_state.quantitative_analysis, tabs)):
            with tab:
                profile = st.session_state.user_profiles[idx]
                st.markdown("#### ユーザペルソナ基本情報")
                st.info(f"**基本情報**: {profile.get('basic_info', 'N/A')}")

                st.markdown("#### 心理的特性・価値観スコア")
                
                # ★修正: データの安全な取得（KeyError防止）
                scores = analysis.get('scores', [])
                explanations = analysis.get('explanations', [])

                # もし辞書型で返ってきてしまっていたらリストの値のみに変換
                if isinstance(scores, dict):
                    scores = list(scores.values())
                if isinstance(explanations, dict):
                    explanations = list(explanations.values())
                
                # リストでない場合のフォールバック
                if not isinstance(scores, list): scores = [5] * 7
                if not isinstance(explanations, list): explanations = ["詳細なし"] * 7

                for i, scale in enumerate(quantitative_scales):
                    # インデックス範囲チェック
                    if i < len(scores) and i < len(explanations):
                        with st.container():
                            scale_display = scale.split(":")[0] + " " + scale.split(":")[1]
                            
                            col1, col2, col3 = st.columns([3, 1, 4])
                            with col1:
                                st.markdown(f"**{scale_display}**")
                            with col2:
                                # スコア表示の安全策
                                try:
                                    score_val = int(scores[i])
                                except:
                                    score_val = 5
                                st.progress(min(max(score_val, 0), 10) / 10)
                                st.markdown(f"**{score_val}/10**")
                            with col3:
                                st.caption(f"*{explanations[i]}*")

        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("➡️ 次へ進む：対話型質問", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.session_state.analysis_completed = True
                st.rerun()
        with col2:
            if st.button("🔄 再分析", type="secondary", use_container_width=True):
                st.session_state.quantitative_analysis = []
                st.session_state.analysis_completed = False
                st.rerun()
        with col3:
            if st.button("⬅️ 前のステップに戻る", type="secondary", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        return

    # 分析開始ボタン
    if st.button("📊 心理的分析を開始", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.rerun()

    # 分析ロジック
    if st.session_state.processing:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            all_analysis = []
            total_profiles = len(st.session_state.user_profiles)

            for idx, profile in enumerate(st.session_state.user_profiles):
                status_text.text(f"ユーザペルソナ {idx + 1}/{total_profiles} を分析中...")
                progress_bar.progress(idx / total_profiles)

                analysis_template = ChatPromptTemplate.from_messages([
                    ("system", "あなたは心理学者兼映画アナリストです。ユーザペルソナの性格と価値観を分析してください。"),
                    ("human", """
                    以下のユーザペルソナを、指定された心理的尺度（Big Fiveおよび価値観）に基づいて定量的に分析してください。
                    
                    評価尺度リスト（1-10で評価）:
                    {scales}

                    ユーザペルソナ情報:
                    {profile_info}

                    指示:
                    - "scores": 各尺度のスコア（1-10）を含むリスト
                    - "explanations": 各評価の理由（性格や価値観の観点）を含むリスト

                    必ずJSON形式で出力してください。
                    """)
                ])

                parser = JsonOutputParser()
                analysis_chain = analysis_template | llm | parser

                profile_info = f"""
                基本情報: {profile.get('basic_info', '')}
                性格特徴: {profile.get('personality', '')}
                価値観: {profile.get('values', '')}
                """

                try:
                    result = safe_llm_call(analysis_chain, {
                        "scales": "\n".join(quantitative_scales),
                        "profile_info": profile_info
                    })

                    # ★修正: 生成データの型チェックと正規化
                    if isinstance(result, dict):
                        # scoresの処理
                        raw_scores = result.get("scores", [])
                        if isinstance(raw_scores, dict): raw_scores = list(raw_scores.values())
                        
                        valid_scores = []
                        for score in raw_scores:
                            try:
                                num_score = int(score)
                                valid_scores.append(max(1, min(10, num_score)))
                            except:
                                valid_scores.append(5)
                        
                        # explanationsの処理
                        raw_explanations = result.get("explanations", [])
                        if isinstance(raw_explanations, dict): raw_explanations = list(raw_explanations.values())
                        if not isinstance(raw_explanations, list): raw_explanations = ["詳細なし"] * 7

                        # 長さ調整
                        target_len = 7
                        if len(valid_scores) < target_len:
                            valid_scores.extend([5] * (target_len - len(valid_scores)))
                        if len(raw_explanations) < target_len:
                            raw_explanations.extend(["詳細なし"] * (target_len - len(raw_explanations)))

                        result["scores"] = valid_scores[:target_len]
                        result["explanations"] = raw_explanations[:target_len]
                        result["profile_id"] = profile.get("profile_id")
                        all_analysis.append(result)
                    else:
                        raise ValueError("Invalid JSON structure")

                except Exception as e:
                    # エラー時のフォールバック
                    all_analysis.append({
                        "profile_id": profile.get("profile_id"),
                        "scores": [5] * 7,
                        "explanations": ["分析失敗"] * 7
                    })

            progress_bar.progress(1.0)
            status_text.text("✅ 分析完了！")

            st.session_state.quantitative_analysis = all_analysis
            st.session_state.analysis_completed = True
            st.session_state.processing = False

            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ 分析中にエラー: {str(e)}")
            st.session_state.processing = False


# ステップ4：対話型質問生成
def step4_generate_question():
    st.title("🎬 インテリジェント映画推薦システム")
    st.markdown("### ステップ 4: 個別化質問対話")

    remaining_profiles = len(st.session_state.user_profiles)
    st.info(f"**残りユーザペルソナ**: {remaining_profiles}個")

    if remaining_profiles <= 1:
        st.success("🎉 最終ユーザペルソナが確定しました！")
        if st.button("映画推薦を生成", type="primary"):
            st.session_state.final_profile = st.session_state.user_profiles[0] if st.session_state.user_profiles else None
            st.session_state.step = 6
            st.rerun()
        return

    if not st.session_state.quantitative_analysis:
        st.error("❌ データ不足です。ステップ3に戻ってください")
        if st.button("戻る"):
            st.session_state.step = 3
            st.rerun()
        return

    # ---------------------------------------------------------
    # 質問表示部分（UI改善）
    # ---------------------------------------------------------
    if st.session_state.current_question and st.session_state.current_scale:
        st.success("💡 あなたの好みについて教えてください")
        
        st.markdown("### 質問:")
        st.markdown(f"**{st.session_state.current_question}**")

        # 保存された選択肢を取得（ない場合はデフォルト）
        opts = st.session_state.get('current_options', {'a': '前者', 'b': '後者'})

        # ★修正: 選択肢の内容を明記したラジオボタンを作成
        options = [
            f"A: 【{opts.get('a', '前者')}】 を強く好む",
            f"どちらかといえば A ({opts.get('a', '前者')})",
            "どちらとも言えない / バランス重視",
            f"どちらかといえば B ({opts.get('b', '後者')})",
            f"B: 【{opts.get('b', '後者')}】 を強く好む"
        ]
        
        # 回答フォーム
        answer_selection = st.radio("あなたの感覚に近いのは？", options, key="preference_radio_4", index=2)

        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button("📤 回答を送信", type="primary", use_container_width=True):
                # 選択されたテキストそのものを回答として保存
                st.session_state.questions_asked.append(st.session_state.current_question)
                st.session_state.answers_given.append(str(answer_selection))
                
                st.success("✅ 送信しました")
                
                # 状態クリア
                st.session_state.current_question = ""
                st.session_state.current_scale = ""
                st.session_state.current_options = {} # オプションもクリア
                st.session_state.elimination_completed = False
                
                time.sleep(0.5)
                st.session_state.step = 5
                st.rerun()

        with col2:
            if st.button("🔄 変更", type="secondary"):
                st.session_state.current_question = ""
                st.session_state.current_scale = ""
                st.rerun()
        return

    # ---------------------------------------------------------
    # 質問生成ロジック（JSON出力に変更）
    # ---------------------------------------------------------
    st.info("質問を生成中...")
    with st.spinner("映画嗜好を分析中..."):
        try:
            # 1. 尺度定義
            scale_definitions = [
                {
                    "id": 0,
                    "technical": "認知的複雑性 (SCC)",
                    "keywords": "伏線回収、謎解き、考察、難解",
                    "simple_topic": "ストーリーの複雑さ"
                },
                {
                    "id": 1,
                    "technical": "情動的強度 (ASI)",
                    "keywords": "ハラハラドキドキ、衝撃的、アクション、スピード感",
                    "simple_topic": "刺激と興奮"
                },
                {
                    "id": 2,
                    "technical": "道徳的整合性 (MVA)",
                    "keywords": "社会派、正義、勧善懲悪、メッセージ性",
                    "simple_topic": "道徳的テーマ"
                },
                {
                    "id": 3,
                    "technical": "心理的安全性 (PSF)",
                    "keywords": "ハッピーエンド、王道、安心感、癒やし",
                    "simple_topic": "安心感"
                },
                {
                    "id": 4,
                    "technical": "美的・抽象性 (AAO)",
                    "keywords": "映像美、独特な世界観、雰囲気、芸術的",
                    "simple_topic": "映像と雰囲気"
                },
                {
                    "id": 5,
                    "technical": "社会的密度 (SRD)",
                    "keywords": "人間関係、恋愛、友情、会話劇",
                    "simple_topic": "人間ドラマ"
                },
                {
                    "id": 6,
                    "technical": "実用的リアリズム (PRI)",
                    "keywords": "実話ベース、リアリティ、ドキュメンタリータッチ",
                    "simple_topic": "リアリティ"
                }
            ]

            # 2. 分散計算
            scores_matrix = []
            for analysis in st.session_state.quantitative_analysis:
                scores = analysis.get('scores', [0] * 7)
                scores_matrix.append(scores)
            
            scores_array = np.array(scores_matrix)
            if scores_array.shape[1] < 7:
                 scores_array = np.pad(scores_array, ((0,0), (0, 7-scores_array.shape[1])), 'constant')
            
            variances = np.var(scores_array[:, :7], axis=0)

            # 3. 使用済み除外
            if 'used_scale_indices' not in st.session_state:
                st.session_state.used_scale_indices = []
            
            for idx in st.session_state.used_scale_indices:
                if idx < len(variances):
                    variances[idx] = -1.0

            # 4. 尺度選択
            max_var_index = int(np.argmax(variances))
            if variances[max_var_index] == -1.0:
                 st.session_state.used_scale_indices = []
                 variances = np.var(scores_array[:, :7], axis=0)
                 max_var_index = int(np.argmax(variances))

            selected_scale = scale_definitions[max_var_index]
            st.session_state.used_scale_indices.append(max_var_index)

            # 5. 質問生成プロンプト（JSON出力を強制）
            past_questions = "\n".join(st.session_state.questions_asked) if st.session_state.questions_asked else "なし"

            question_template = ChatPromptTemplate.from_messages([
                ("system", """
                 あなたは親しみやすい映画コンシェルジュです。
                 ユーザーの好みを「AかBか」の形式で尋ねる質問を作成してください。
                 """),
                ("human", """
                 【指示】
                 テーマ「{simple_topic}」について、対立する2つの選択肢（AとB）を提示する質問を作成してください。
                 専門用語は使わず、具体的な映画の楽しみ方で表現してください。

                 キーワード: {keywords}
                 過去の質問: {past_questions}

                 【出力形式】
                 以下のJSON形式のみを出力してください：
                 {{
                    "question": "質問文（例：映画のアクションシーンについてどう感じますか？）",
                    "option_a": "選択肢Aの具体的な内容（例：ハラハラする激しいアクションが好き）",
                    "option_b": "選択肢Bの具体的な内容（例：アクションより落ち着いた会話が好き）"
                 }}
                 """)
            ])

            parser = JsonOutputParser()
            chain = question_template | llm | parser

            result = safe_llm_call(chain, {
                "simple_topic": selected_scale["simple_topic"],
                "keywords": selected_scale["keywords"],
                "past_questions": past_questions
            })

            # 結果の保存
            st.session_state.current_question = result.get('question', '映画の好みについて')
            st.session_state.current_options = {
                'a': result.get('option_a', '前者'),
                'b': result.get('option_b', '後者')
            }
            st.session_state.current_scale = selected_scale["technical"]
            st.rerun()

        except Exception as e:
            st.error(f"質問生成エラー: {e}")
            # エラー時のフォールバック
            st.session_state.current_question = "映画のストーリーについて、どちらを好みますか？"
            st.session_state.current_options = {
                'a': '考察が必要な複雑なストーリー',
                'b': 'わかりやすくてスッキリするストーリー'
            }
            st.session_state.current_scale = "General"
            st.rerun()

# ステップ5：ユーザペルソナ淘汰
# ========== ステップ5：ユーザペルソナ淘汰（履歴参照版） ==========
def step5_eliminate_profile():
    st.title("🎬 インテリジェント映画推薦システム")
    st.markdown("### ステップ 5: ユーザペルソナ更新")

    remaining_profiles = len(st.session_state.user_profiles)

    # 1. 終了条件チェック
    if remaining_profiles <= 1:
        st.session_state.final_profile = st.session_state.user_profiles[0] if st.session_state.user_profiles else None
        st.success("🎉 最終ユーザペルソナが確定しました！")
        if st.button("映画推薦を生成", type="primary"):
            st.session_state.step = 6
            st.rerun()
        return

    # 2. 淘汰処理（未実施の場合のみ実行）
    if not st.session_state.elimination_completed:
        with st.spinner("これまでの対話履歴に基づいてユーザペルソナを更新中..."):
            try:
                # ★修正点1: これまでの全履歴を整形してテキスト化
                history_text = ""
                if st.session_state.questions_asked and st.session_state.answers_given:
                    for i, (q, a) in enumerate(zip(st.session_state.questions_asked, st.session_state.answers_given)):
                        history_text += f"質問{i+1}: {q}\n回答{i+1}: {a}\n---\n"
                else:
                    history_text = "履歴なし"

                # ★修正点2: プロンプトを「履歴全体」を見るように変更
                elimination_template = ChatPromptTemplate.from_messages([
                    ("system", "あなたはユーザー行動分析の専門家です。これまでのユーザーとの全対話履歴を分析し、ユーザーの回答パターンと最も矛盾する（一致しない）ユーザペルソナを淘汰してください。"),
                    ("human", """
                    以下の情報に基づいて、最も一致しないユーザペルソナを1つ淘汰してください：

                    【ユーザペルソナ集合】
                    {profiles}

                    【これまでの対話履歴（全質問と回答）】
                    {history}

                    【指示】
                    - 最新の回答だけでなく、これまでの全ての回答との整合性を総合的に判断してください。
                    - ユーザーの一貫した好みや傾向と、最も矛盾が大きいペルソナを選んでください。

                    JSON形式で出力し、以下を含めてください：
                    - eliminated_id: 淘汰されたペルソナID（整数である必要があります）
                    - reason: 淘汰理由（対話履歴のどの部分と矛盾したか、具体的に記述してください）

                    JSONのみを出力し、他のテキストは含めないでください。
                    """)
                ])

                parser = JsonOutputParser()
                chain = elimination_template | llm | parser

                # ★修正点3: 履歴テキストを渡す
                result = safe_llm_call(chain, {
                    "profiles": json.dumps(st.session_state.user_profiles, ensure_ascii=False),
                    "history": history_text
                })

                eliminated_id = result.get('eliminated_id', 1)
                reason = result.get('reason', 'ユーザーの回答履歴と一致しない')

                try:
                    eliminated_id = int(eliminated_id)
                except:
                    eliminated_id = 1

                # 履歴記録（最新のQ&Aを記録用に取得）
                last_q = st.session_state.questions_asked[-1] if st.session_state.questions_asked else "N/A"
                last_a = st.session_state.answers_given[-1] if st.session_state.answers_given else "N/A"

                st.session_state.elimination_history.append({
                    "eliminated_id": eliminated_id,
                    "reason": reason,
                    "question": last_q,
                    "answer": last_a
                })

                # データ更新
                new_profiles = [p for p in st.session_state.user_profiles if p.get('profile_id') != eliminated_id]
                new_analysis = [a for a in st.session_state.quantitative_analysis if a.get('profile_id') != eliminated_id]

                st.session_state.user_profiles = new_profiles
                st.session_state.quantitative_analysis = new_analysis

                # フラグをTrueにして再実行
                st.session_state.elimination_completed = True
                st.rerun()

            except Exception as e:
                st.error(f"❌ ユーザペルソナ淘汰中にエラー: {str(e)}")
                # エラー時の安全策
                if st.session_state.user_profiles:
                    eliminated_profile = st.session_state.user_profiles[0]
                    st.session_state.user_profiles = st.session_state.user_profiles[1:]
                    st.session_state.elimination_history.append({
                        "eliminated_id": eliminated_profile.get('profile_id', 1),
                        "reason": "システムエラー",
                        "question": "N/A", "answer": "N/A"
                    })
                    st.session_state.elimination_completed = True
                    st.rerun()

    # 3. 結果表示と「次へ」ボタン（処理済みの場合に表示）
    else:
        # 最新の淘汰結果を表示
        if st.session_state.elimination_history:
            last = st.session_state.elimination_history[-1]
            st.success(f"✅ ユーザペルソナ {last['eliminated_id']} を淘汰しました")
            st.info(f"理由: {last['reason']}")

        # 履歴アコーディオン
        with st.expander("📋 淘汰履歴を表示", expanded=False):
            for h in reversed(st.session_state.elimination_history):
                st.markdown(f"**ペルソナ {h['eliminated_id']}** (理由: {h['reason']})")

        # 次へ進むボタン
        if len(st.session_state.user_profiles) > 1:
            st.markdown(f"### 残り {len(st.session_state.user_profiles)} 個のユーザペルソナ")
            if st.button("次の質問へ", type="primary"):
                st.session_state.step = 4
                st.rerun()
        else:
            st.session_state.final_profile = st.session_state.user_profiles[0]
            st.success("🎉 最終ユーザペルソナが確定しました！")
            if st.button("映画推薦を生成", type="primary"):
                st.session_state.step = 6
                st.rerun()


# ステップ6：映画推薦生成
def step6_generate_recommendation():
    st.title("🎬 インテリジェント映画推薦システム")
    st.markdown("### ステップ 6: あなたの個別化推薦")

    if not st.session_state.final_profile:
        st.error("最終ユーザペルソナが見つかりません")
        if st.button("戻って再選択"):
            st.session_state.step = 4
            st.rerun()
        return

    # すでに推薦生成済みの場合、直接表示
    if st.session_state.recommendation:
        rec = st.session_state.recommendation
        display_recommendation(rec)
        return

    with st.spinner("あなたのための個別化映画推薦を生成中..."):
        try:
            # ★修正: プロンプトに「属性情報への言及禁止」を追加
            recommendation_template = ChatPromptTemplate.from_messages([
                ("system", """あなたはプロの映画推薦専門家です。
                ユーザペルソナと嗜好に基づいて、最も適した1本の映画を推薦してください。
                """),
                ("human", """
                以下の情報に基づいて、ユーザーに映画を1本推薦してください：

                最終確定ユーザペルソナ:
                {final_profile}

                ユーザーの好きな映画: {liked_movies}
                ユーザーの嫌いな映画: {disliked_movies}

                対話記録:
                質問: {questions}
                回答: {answers}

                【重要: 推薦理由（reason）の書き方について】
                - ユーザーの「年齢」「性別」「職業」などの予測属性には**絶対に言及しないでください**（予測が外れていると不快感を与えるため）。
                - 代わりに、ユーザーの「性格」「価値観」「映画のトーンへの好み」に焦点を当てて理由を説明してください。
                - 例：「あなたは30代のエンジニアなので」→ 禁止❌
                - 例：「あなたは論理的なストーリー構成と、静かな感動を好む傾向があるため」→ 推奨⭕️

                JSON形式で出力し、以下のフィールドを含めてください：
                - recommended_movie: 推薦映画名（実際に存在する映画である必要があります）
                - year: 公開年
                - genre: ジャンル（リスト）
                - director: 監督
                - main_cast: 主要キャスト（リスト）
                - reason: 詳細な推薦理由（少なくとも100文字・属性情報には触れないこと）
                - match_points: マッチポイントリスト（少なくとも3つ）
                - streaming_platforms: 視聴可能なストリーミングプラットフォーム（リスト）

                JSONのみを出力し、他のテキストは含めないでください。
                """)
            ])

            parser = JsonOutputParser()
            chain = recommendation_template | llm | parser

            result = safe_llm_call(chain, {
                "final_profile": json.dumps(st.session_state.final_profile, ensure_ascii=False),
                "liked_movies": ", ".join(st.session_state.liked_movies),
                "disliked_movies": ", ".join(
                    st.session_state.disliked_movies) if st.session_state.disliked_movies else "なし",
                "questions": "\n".join(st.session_state.questions_asked) if st.session_state.questions_asked else "なし",
                "answers": "\n".join(st.session_state.answers_given) if st.session_state.answers_given else "なし"
            })

            st.session_state.recommendation = result
            display_recommendation(result)

        except Exception as e:
            st.error(f"❌ 推薦生成中にエラー: {str(e)}")
            # デフォルト推薦提供
            st.session_state.recommendation = {
                "recommended_movie": "ショーシャンクの空に",
                "year": "1994",
                "genre": ["ドラマ", "犯罪"],
                "director": "フランク・ダラボン",
                "main_cast": ["ティム・ロビンス", "モーガン・フリーマン"],
                "reason": "これは古典的な感動的な映画で、刑務所の中で希望と尊厳を保ち続ける物語を描いています。映画は人間性の輝きを示し、深い哲学的思考と感情的な力に満ちており、あなたの好む重厚なテーマと一致します。",
                "match_points": ["深いテーマ", "優れた物語構成", "心を動かす", "俳優の優れた演技"],
                "streaming_platforms": ["Netflix", "Amazon Prime Video", "Hulu"]
            }
            display_recommendation(st.session_state.recommendation)


def display_recommendation(rec):
    """推薦結果を表示"""
    st.success("🎉 あなたに最も適した映画を見つけました！")

    st.markdown("---")
    st.markdown(f"## 🎬 **{rec.get('recommended_movie', '未知の映画')}** ({rec.get('year', '未知の年')})")

    # 基本情報
    col1, col2 = st.columns([2, 1])
    with col1:
        genre = rec.get('genre', '未知のジャンル')
        if isinstance(genre, list):
            genre = " · ".join(genre)
        st.markdown(f"**ジャンル**: {genre}")
        st.markdown(f"**監督**: {rec.get('director', '未知の監督')}")

        cast = rec.get('main_cast', [])
        if isinstance(cast, list) and cast:
            cast_str = " · ".join(cast[:3])  # 主要キャスト上位3名のみ表示
            if len(cast) > 3:
                cast_str += "など"
            st.markdown(f"**主演**: {cast_str}")

    with col2:
        platforms = rec.get('streaming_platforms', [])
        if isinstance(platforms, list) and platforms:
            st.markdown("**視聴可能プラットフォーム**:")
            for platform in platforms[:3]:  # 上位3プラットフォームのみ表示
                st.markdown(f"• {platform}")

    st.markdown("---")
    st.markdown("### 📝 推薦理由")
    st.markdown(rec.get('reason', '推薦理由なし'))

    st.markdown("### ✅ マッチポイント分析")
    match_points = rec.get('match_points', [])
    if isinstance(match_points, list):
        cols = st.columns(2)
        for i, point in enumerate(match_points):
            with cols[i % 2]:
                st.markdown(f"✓ {point}")
    elif isinstance(match_points, str):
        st.markdown(match_points)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🤔 よくある質問を表示", type="primary", use_container_width=True):
            st.session_state.step = 7
            st.rerun()
    with col2:
        if st.button("🔄 推薦を再生成", type="secondary", use_container_width=True):
            st.session_state.recommendation = None
            st.rerun()
    with col3:
        if st.button("📋 分析レポートを表示", type="secondary", use_container_width=True):
            show_analysis_report()


def show_analysis_report():
    """完全な分析レポートを表示"""
    with st.expander("📊 完全分析レポート", expanded=True):
        st.markdown("### ユーザー嗜好分析レポート")

        # 基本情報
        st.markdown("#### 利用統計")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("好きな映画数", len(st.session_state.liked_movies))
        with col2:
            st.metric("対話質問数", len(st.session_state.questions_asked))
        with col3:
            st.metric("分析所要時間", f"{(datetime.now() - st.session_state.start_time).seconds}秒")

        # 好きな映画
        st.markdown("#### 登録された好みの映画")
        cols = st.columns(3)
        for i, movie in enumerate(st.session_state.liked_movies):
            with cols[i % 3]:
                st.markdown(f"• {movie}")

        # ★修正: 最終ユーザペルソナ（JSON）の表示部分を削除しました。
        # 予測された属性情報はユーザーに見せないようにします。
        
        # 代わりに、分析された好みのキーワード（もしあれば）などを表示するのが適切ですが、
        # ここではシンプルに削除のみ行います。
        st.markdown("---")
        st.info("※ この分析に基づき、最適な映画を選出しました。")


# ステップ7：予測質問
def step7_generate_qa():
    st.title("🎬 インテリジェント映画推薦システム")
    st.markdown("### ステップ 7: よくある質問")

    with st.spinner("あなたが気になる可能性のある質問を予測中..."):
        try:
            qa_template = ChatPromptTemplate.from_messages([
                ("system", "あなたは映画の専門家で、ユーザーが推薦映画について持つ可能性のある質問を予測し、正確で役立つ回答を提供できます。"),
                ("human", """
                以下の情報に基づいて、ユーザーが持つ可能性のある3つの質問を予測し、回答を提供してください：

                推薦映画: {recommendation}

                JSON形式で出力し、以下を含めてください：
                - qa_pairs: 質問-回答ペアのリスト、各要素に"question"と"answer"フィールドを含む

                質問は以下の側面をカバーするべきです：映画評価、視聴アドバイス、類似推薦など。
                回答は詳細で正確であり、少なくとも50文字以上であるべきです。

                JSONのみを出力し、他のテキストは含めないでください。
                """)
            ])

            parser = JsonOutputParser()
            chain = qa_template | llm | parser

            result = safe_llm_call(chain, {
                "recommendation": json.dumps(st.session_state.recommendation, ensure_ascii=False)
            })

            qa_pairs = result.get('qa_pairs', [])
            st.session_state.qa_pairs = qa_pairs[:3]

        except Exception as e:
            st.session_state.qa_pairs = [
                {
                    "question": "なぜこの映画を推薦したのですか？",
                    "answer": "この映画はあなたのユーザペルソナと非常に一致しています。あなたが好む映画のジャンルや価値観が、この映画のテーマやスタイルと高度に一致しています。"
                },
                {
                    "question": "どこでこの映画を視聴できますか？",
                    "answer": "この映画は主要なストリーミングプラットフォームで視聴できます。例えばNetflix、Amazon Prime Video、Huluなどです。具体的にはお住まいの地域のストリーミングサービスをご確認ください。"
                },
                {
                    "question": "類似の映画はありますか？",
                    "answer": "あなたの嗜好に基づいて、以下の映画もおすすめします：XXXX、XXXX、XXXX。これらの映画はテーマ、スタイル、または感情的な側面であなたが好きな映画と類似しています。"
                }
            ]

    st.success("📚 以下はあなたが気になる可能性のある質問です：")

    for i, qa in enumerate(st.session_state.qa_pairs, 1):
        with st.expander(f"❓ {qa.get('question', '')}", expanded=(i == 1)):
            st.markdown(qa.get('answer', ''))

    st.markdown("---")
    st.success("✨ 推薦プロセスが完了しました！")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 最初からやり直す", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with col2:
        if st.button("📊 レポートを表示", type="secondary", use_container_width=True):
            show_analysis_report()
    with col3:
        if st.button("⬅️ 推薦に戻る", type="secondary", use_container_width=True):
            st.session_state.step = 6
            st.rerun()


# ========== ステップ6：メインアプリケーションフロー ==========
def main():
    # ページ設定

    # カスタムCSS
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .css-1d391kg {  /* メインコンテナ */
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # セッション状態初期化
    init_session_state()

    with st.sidebar:
        st.title("🎬 映画推薦システム (A)")
        st.markdown(f"**モデル**: {MODEL_NAME}")
        st.markdown("---")
        
        # 進捗表示などは元のコード通り実装
        if 'step' in st.session_state:
            steps = ["嗜好入力", "ユーザペルソナ生成", "定量的分析", "対話型質問", "推薦取得"]
            step_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 5}
            current_step_num = step_mapping.get(st.session_state.step, 1)
            for i, step in enumerate(steps, 1):
                if i < current_step_num: st.markdown(f"✅ {step}")
                elif i == current_step_num: st.markdown(f"▶️ {step}")
                else: st.markdown(f"○ {step}")
        
        st.markdown("---")
        
        # リセットボタン（experiment_mode保護版）
        if st.button("🔄 最初からやり直す", use_container_width=True):
             keys_to_delete = [k for k in st.session_state.keys() if k != "experiment_mode"]
             for key in keys_to_delete:
                 del st.session_state[key]
             st.rerun()

    # メインコンテンツエリアルーティング
    current_step = st.session_state.step

    # 明示的なif-elifチェーンを使用し、ステップ遷移を確実に
    if current_step == 1:
        step1_input_movies()
    elif current_step == 2:
        step2_generate_profiles()
    elif current_step == 3:
        step3_quantitative_analysis()
    elif current_step == 4:
        step4_generate_question()
    elif current_step == 5:
        step5_eliminate_profile()
    elif current_step == 6:
        step6_generate_recommendation()
    elif current_step == 7:
        step7_generate_qa()
    else:
        st.error(f"無効なステップ: {current_step}")
        if st.button("アプリをリセット"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ========== アプリケーション実行 ==========
if __name__ == "__main__":
    # 単体実行時のみページ設定を行う
    st.set_page_config(
        page_title="インテリジェント映画推薦システム (A)",
        page_icon="🎬",
        layout="wide"
    )

    main()


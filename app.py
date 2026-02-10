# app.py
import streamlit as st
import random
import system_a  # ペルソナ型
import system_b  # エージェント型

# この set_page_config は必ずファイルの先頭（インポート直後）に配置
st.set_page_config(page_title="映画推薦システム実験", layout="wide")

# セッション状態で実験モードを管理
if "experiment_mode" not in st.session_state:
    # A/Bテストのためにランダムに割り当てる場合
    # st.session_state.experiment_mode = random.choice(["A", "B"])
    
    # または、ユーザーに選ばせる場合（デバッグ用）
    st.session_state.experiment_mode = None

# モード選択画面
if st.session_state.experiment_mode is None:
    st.title("映画推薦システム評価実験")
    st.write("ご協力ありがとうございます。以下のボタンを押して実験を開始してください。")
    
    col1, col2 = st.columns(2)
    if col1.button("システムAで開始"):
        st.session_state.experiment_mode = "A"
        st.rerun()
    if col2.button("システムBで開始"):
        st.session_state.experiment_mode = "B"
        st.rerun()

else:
    # 割り当てられたシステムを実行
    if st.session_state.experiment_mode == "A":
        system_a.main() 
    elif st.session_state.experiment_mode == "B":

        system_b.main()

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#ストリーミング表示用：11/4時点で微妙なためコメントアウト
#from langchain.callbacks import StreamlitCallbackHandler
import configparser

# ====================================
# 定数定義
# ====================================
# 設定ファイルの読み込み
config = configparser.ConfigParser() #Configのハンドル設定
config.read("./private/config.ini")

#OpenAI部分の設定参照
openai_config = config["OPENAI"] 
input_openai_api_key = openai_config["OPENAI_API_KEY"]

# ====================================
# アプリケーション
# ====================================
def main():        
    #ウェブページの設定
    st.set_page_config(
        page_title="Test App",
        page_icon="./pic/figure_chatGPT.png"
    )
    
    # 横に並べて画像とヘッダーを表示
    col1, col2 = st.columns([1, 5])  # カラムの幅を調整
    with col1:
        st.image("./pic/figure_chatGPT.png", width=50)  # アイコンの幅を調整
    with col2:
        st.header("PoC with ChatGPT")

    #サイドバーの表示
    options_customs()
    
    # サイドバー：履歴の削除
    total_tokens_used = 0
    total_cost = 0    
    init_messages(total_tokens_used,total_cost)
    
    #LangChain設定
    #ChatOpenAIクラスのインスタンス化
    llm = select_model()
    #モデルのプロンプト設定
    opt_system=sidebar_opt_system()
    prompt=ChatPromptTemplate.from_messages([
        ("system",opt_system),
        ("user","{input}")
    ])
    #GPTの返答をパースするための処理
    output_parser=StrOutputParser()
    #LCELでの記法
    chain = prompt | llm | output_parser
    
    #サイドバー：コスト計上用
    total_tokens_display, total_cost_display = initialize_cost_display()
    calc_cost(total_tokens_used,total_cost,total_tokens_display, total_cost_display)
    
    # ユーザーの入力を監視
    if user_input := st.chat_input("質問事項を入力"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        #st.chat_message("user").markdown(user_input)
        with st.spinner("ChatGPT is typing ..."):
        #上部にスピナーとは別のアニメーションが入る。
        #with st.chat_message("assistant"):
            with get_openai_callback() as cb:
                #response = llm(st.session_state.messages)
                #LangChain利用
                response = chain.invoke({"input": st.session_state.messages})
                
                #ストリーミング表示用：11/4時点で微妙なためコメントアウト
                #システムメッセージの上にスピナーが残り続ける+ストリーミング表示になってなさそう
                #st_callback = StreamlitCallbackHandler(st.container())
                #response = llm(st.session_state.messages,callbacks=[st_callback])
                #ストリーミング表示用：11/5データの型が「write_stream」で想定しているものと異なるためエラー
                #response = st.write_stream(chain.invoke({"input": st.session_state.messages}))
                
                # トークン数とコストを変数に格納
                total_tokens_used += cb.total_tokens
                total_cost += cb.total_cost
                #サイドバーに反映
                calc_cost(total_tokens_used,total_cost,total_tokens_display, total_cost_display)
        #st.session_state.messages.append(AIMessage(content=response.content))
        #LangChainを利用したため、responceの型が変更になったことへの対応
        st.session_state.messages.append(AIMessage(content=response))

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")
# ====================================
# サイドバー
# ====================================
#サイドバー：基礎の設定
def options_customs():
    # サイドバーのタイトルを表示
    st.sidebar.title("Options")    
    
    # カスタムCSSを使って背景色を設定
    st.markdown(
        """
        <style>
        /* メインコンテンツエリアの背景色 */
        .main {
            background-color: rgb(241, 235, 227); /* RGBで任意の色に設定 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# サイドバー：モデルの選択
def select_model():
    # モデルの選択・設定
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

    return ChatOpenAI(openai_api_key=input_openai_api_key,temperature=temperature, model_name=model_name,streaming=True)

def sidebar_opt_system():
    # サイドバーにテキスト入力ウィジェットを追加
    opt_system = st.sidebar.text_input("Enter the system prompt:")
    return opt_system

# サイドバー：履歴の削除
def init_messages(tokens,cost):
    # サイドバーにボタンを設置
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="本アプリケーションは●●●を試験的に実施しているものです。")
        ]
        st.session_state.costs = []
        tokens=0
        cost=0
    
    return tokens,cost    

# 初期のコスト表示をサイドバーに設定する関数
def initialize_cost_display():
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown("**Total cost**")
    total_tokens_display = st.sidebar.empty()  # トークン数のプレースホルダー
    total_cost_display = st.sidebar.empty()    # コストのプレースホルダー
    return total_tokens_display,total_cost_display
#サイドバー：コストの計上
def calc_cost(tokens,cost,total_tokens_display,total_cost_display):
    # Streamlitはmarkdownを書けばいい感じにHTMLで表示してくれます
    # (もちろんメイン画面でも使えます)
    total_tokens_display.markdown(f"- total tokens:{tokens}")
    total_cost_display.markdown(f"- total cost ${cost}")

# ====================================
# プログラムの実行
# ====================================
if __name__ == '__main__':
    main()
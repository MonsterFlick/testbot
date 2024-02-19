"""
NeuraTalk
======================

This is a Streamlit chatbot app with LLaMA2 that includes session chat history and an option to select multiple LLM
API endpoints on Replicate. The 7B and 13B models run on Replicate on one A100 40Gb. The 70B runs in one A100 80Gb. The weights have been tensorized.
"""
#External libraries:
import streamlit as st
import replicate
from dotenv import load_dotenv
load_dotenv()
import os
from utils import debounce_replicate_run
from auth0_component import login_button
import io
import streamlit.components.v1 as components


if 'url' not in st.session_state:
    st.session_state['url'] = "https://monsterchat.streamlit.app/"

###Global variables:###
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', default='')
#Your your (Replicate) models' endpoints:
REPLICATE_MODEL_ENDPOINT7B = os.environ.get('REPLICATE_MODEL_ENDPOINT7B', default='')
REPLICATE_MODEL_ENDPOINT13B = os.environ.get('REPLICATE_MODEL_ENDPOINT13B', default='')
REPLICATE_MODEL_ENDPOINT70B = os.environ.get('REPLICATE_MODEL_ENDPOINT70B', default='')
PRE_PROMPT = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant."
#Auth0 for auth
AUTH0_CLIENTID = os.environ.get('AUTH0_CLIENTID', default='')
AUTH0_DOMAIN = os.environ.get('AUTH0_DOMAIN', default='')

if not (REPLICATE_API_TOKEN and REPLICATE_MODEL_ENDPOINT13B and REPLICATE_MODEL_ENDPOINT7B and 
        AUTH0_CLIENTID and AUTH0_DOMAIN):
    st.warning("Add a `.env` file to your app directory with the keys specified in `.env_template` to continue.")
    st.stop()

###Initial UI configuration:###
st.set_page_config(page_title=" Revolutionize The World Of AI: NeuraTalk", page_icon="❤️", layout="wide")

def render_app():

    # reduce font sizes for input text boxes
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    #Left sidebar menu
    st.sidebar.header("NeuraTalk Chatbot")

    #Set config for a cleaner menu, footer & background:
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()
    #Set up/Initialize Session State variables:
    if 'chat_dialogue' not in st.session_state:
        st.session_state['chat_dialogue'] = []
    if 'llm' not in st.session_state:
        #st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT13B
        st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT70B
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.1
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 0.9
    if 'max_seq_len' not in st.session_state:
        st.session_state['max_seq_len'] = 512
    if 'pre_prompt' not in st.session_state:
        st.session_state['pre_prompt'] = PRE_PROMPT
    if 'string_dialogue' not in st.session_state:
        st.session_state['string_dialogue'] = ''

    #Dropdown menu to select the model edpoint:
    selected_option = st.sidebar.selectbox('Choose a LLaMA2 model:', ['LLaMA2-70B', 'LLaMA2-13B', 'LLaMA2-7B'], key='model')
    if selected_option == 'LLaMA2-7B':
        st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT7B
    elif selected_option == 'LLaMA2-13B':
        st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT13B
    else:
        st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT70B
    #Model hyper parameters:
    st.session_state['temperature'] = st.sidebar.slider('Temperature:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    st.session_state['top_p'] = st.sidebar.slider('Top P:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state['max_seq_len'] = st.sidebar.slider('Max Sequence Length:', min_value=64, max_value=4096, value=2048, step=8)

    NEW_P = st.sidebar.text_area('Prompt before the chat starts. Edit here if desired:', PRE_PROMPT, height=60)
    if NEW_P != PRE_PROMPT and NEW_P != "" and NEW_P != None:
        st.session_state['pre_prompt'] = NEW_P + "\n\n"
    else:
        st.session_state['pre_prompt'] = PRE_PROMPT

    btn_col1, btn_col2 = st.sidebar.columns(2)

    # Add the "Clear Chat History" button to the sidebar
    def clear_history():
        st.session_state['chat_dialogue'] = []
    clear_chat_history_button = btn_col1.button("Clear History",
                                            use_container_width=True,
                                            on_click=clear_history)

    # add logout button
    def logout():
        del st.session_state['user_info']
    logout_button = btn_col2.button("Logout",
                                use_container_width=True,
                                on_click=logout)
        
    # add links to relevant resources for users to select
    st.sidebar.write(" ")

    text1 = 'Om Thakur' 
    text2 = 'Karishma Shinde' 
    text3 = 'GitHub'

    text1_link = "https://github.com/MonsterFlick"
    text2_link = "https://github.com/MonsterFlick"
    text3_link = "https://github.com/MonsterFlick"

    logo1 = 'https://github.com/MonsterFlick/MonsterFPS/blob/main/Monster.gif?raw=true'
    logo2 = 'https://github.com/MonsterFlick/MonsterFPS/blob/main/Monster.gif?raw=true'

    st.sidebar.markdown(
        "**About Us**  \n"
        f"<img src='{logo2}' style='height: 1em'> [{text2}]({text2_link})  \n"
        f"<img src='{logo1}' style='height: 1em'> [{text1}]({text1_link})  \n"
        f"<img src='{logo1}' style='height: 1em'> [{text3}]({text3_link})",
        unsafe_allow_html=True)

    st.sidebar.write(" ")
    st.sidebar.markdown("*Created by Om Thakur and Karishma Shinde. Not associated with Meta Platforms, Inc.*")

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Let The Journey Begin"):
        # Add user message to chat history
        st.session_state.chat_dialogue.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            string_dialogue = st.session_state['pre_prompt']
            for dict_message in st.session_state.chat_dialogue:
                if dict_message["role"] == "user":
                    string_dialogue = string_dialogue + "User: " + dict_message["content"] + "\n\n"
                else:
                    string_dialogue = string_dialogue + "Assistant: " + dict_message["content"] + "\n\n"
            print (string_dialogue)
            output = debounce_replicate_run(st.session_state['llm'], string_dialogue + "Assistant: ",  st.session_state['max_seq_len'], st.session_state['temperature'], st.session_state['top_p'], REPLICATE_API_TOKEN)
            for item in output:
                full_response += item
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.chat_dialogue.append({"role": "assistant", "content": full_response})


if 'user_info' in st.session_state:
    render_app()
else:
    login_url = f"https://{AUTH0_DOMAIN}/authorize?" \
            f"response_type=token&" \
            f"client_id={AUTH0_CLIENTID}&" \
            f"redirect_uri={st.session_state['url']}"

    st.markdown(f"Please wait while you're being redirected to the login page...")
    st.markdown(f'<meta http-equiv="refresh" content="0;URL={login_url}" />', unsafe_allow_html=True)
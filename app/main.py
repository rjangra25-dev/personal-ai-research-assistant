import streamlit as st
from agent import run_agent

st.set_page_config(page_title="Personal AI Research Assistant", page_icon="📚")

st.title("📚 Personal AI Research Assistant")
st.caption("Ask about your documents, do quick math, or ask general questions — the agent decides how to answer.")

# Keep the chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# The input box at the bottom
if question := st.chat_input("Ask a question..."):
    # Show the user's message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get the agent's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = run_agent(question)
        st.markdown(answer)

    # Save the assistant's answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
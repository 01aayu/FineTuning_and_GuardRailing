import streamlit as st
import requests

st.title("Medical Chatbot")

# Set News API key from Streamlit secrets
api_key = st.secrets["NEWS_API"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about current news topics or categories"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch news based on the user's input (for simplicity, treating the prompt as a category or keyword)
    url = f"https://newsapi.org/v2/everything?q={prompt}&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()

    if response.status_code == 200 and news_data["totalResults"] > 0:
        articles = news_data["articles"]
        news_response = "\n\n".join([f"**{article['title']}**\n{article['description']}" for article in articles[:3]])
    else:
        news_response = "Sorry, no news found for that topic."

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(news_response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": news_response})

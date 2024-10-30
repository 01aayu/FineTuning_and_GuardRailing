import streamlit as st
import requests
import json

# Ollama API endpoint
API_URL = "http://localhost:11434/api/generate"

st.title("Ollama Model Chat Interface")

# Initialize chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Initialize prompt state if not already done
if "prompt" not in st.session_state:
    st.session_state["prompt"] = ""

# Display chat history
st.write("### Chat History")
for entry in st.session_state["chat_history"]:
    st.markdown(f"**User**: {entry['user']}")
    st.markdown(f"**Model**: {entry['model']}")
    st.markdown("---")

# Input box for the prompt
prompt = st.text_input("Enter your prompt:", value=st.session_state["prompt"])

# Automatically execute when a prompt is entered
if prompt:
    payload = {
        "model": "llama3.1_medical",
        "prompt": prompt
    }
    
    # Initialize empty response text
    response_text = ""
    
    # Add user prompt to chat history
    st.session_state["chat_history"].append({"user": prompt, "model": ""})
    
    try:
        # Send POST request to Ollama API with stream=True
        response = requests.post(API_URL, json=payload, stream=True)
        
        if response.status_code == 200:
            # Placeholder to display the model's response as it's being streamed
            response_placeholder = st.empty()
            
            # Append response parts to build a continuous response
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse each line as JSON
                        line_data = json.loads(line)
                        
                        # Append each part of the response
                        response_text += line_data.get('response', " ")
                        
                        # Update placeholder with the latest accumulated response in paragraph format
                        response_placeholder.markdown(f"<p style='white-space: pre-wrap;'>{response_text}</p>", unsafe_allow_html=True)
                        
                    except json.JSONDecodeError:
                        st.error("Failed to decode part of the response.")
                        st.write(line)
            
            # Update chat history with the model's final response in paragraph format
            st.session_state["chat_history"][-1]["model"] = response_text
            
            # Clear the input box after processing
            st.session_state["prompt"] = ""
        else:
            st.error(f"Error: {response.status_code}")
            st.write("Response text:", response.text)

    except requests.exceptions.RequestException as e:
        st.error("Failed to connect to the Ollama API.")
        st.write(e)
else:
    st.info("Enter a prompt to start the conversation.")

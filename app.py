import streamlit as st
import requests
from ollama import chat
from guardrails import Guard
import json
import xml.etree.ElementTree as ET
import logging
from difflib import unified_diff

# Setup logging
logging.basicConfig(
    filename="guard/logs.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

st.title("Base Model via API and Guardrailed Model Locally")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Dropdown options for models
base_model_options = ["llama", "mistral", "gemma"]
guardrailed_model_options = ["llama3.1_medical", "mistral_medical", "gemma1.1_medical"]

# Sidebar for model selection
st.sidebar.header("Model Selection")
base_model = st.sidebar.selectbox("Select Base Model", base_model_options)
guardrailed_model = st.sidebar.selectbox("Select Guardrailed Model", guardrailed_model_options)

# Load Guardrails XML configuration
def load_guardrails_config():
    try:
        tree = ET.parse("config/rails.xml")
        xml_str = ET.tostring(tree.getroot(), encoding="unicode")
        return xml_str
    except Exception as e:
        logging.error(f"Error loading Guardrails config: {e}")
        st.error(f"Error loading Guardrails config: {e}")
        return None

# Initialize Guard instance
rail_config = load_guardrails_config()
if rail_config:
    guard = Guard.from_rail_string(rail_config)
else:
    guard = None

# Apply moderation using Guardrails
def moderate_response(bot_output):
    if not guard:
        return {"passed": False, "issues": "Guardrails configuration not loaded."}

    try:
        structured_output = json.dumps({"bot_output": {"response": bot_output}})
        validated_output = guard.parse(structured_output)
        return {"passed": True, "validated_response": validated_output}
    except Exception as e:
        logging.error(f"Guardrails validation failed: {e}")
        return {"passed": False, "issues": str(e)}

# Process streamed JSON response
def process_streamed_json(response):
    full_text = ""
    for line in response.iter_lines():
        try:
            chunk = json.loads(line.decode("utf-8"))
            if "response" in chunk:
                full_text += chunk["response"]
        except json.JSONDecodeError as e:
            logging.error(f"Streamed JSON decode error: {e}")
            st.error(f"Malformed chunk in streamed response: {line}")
    return full_text

# Input box for user prompt
prompt = st.text_input("Enter your prompt:")

if prompt:
    # Get response from base model via Ollama API
    try:
        base_response = chat(
            model=base_model,
            messages=[{"role": "user", "content": prompt}]
        )
        base_output = base_response.message.content

        # Payload for guardrailed model
        guardrailed_payload = {
            "model": guardrailed_model,
            "prompt": prompt,
            "max_tokens": 150
        }

        # Get response from guardrailed model locally
        try:
            guard_response = requests.post(
                "http://localhost:11434/api/generate", 
                json=guardrailed_payload, 
                stream=True,  # Enable streaming
                timeout=10
            )

            if guard_response.status_code == 200:
                guard_output = process_streamed_json(guard_response)

                # Validate guardrailed output
                moderation_result = moderate_response(guard_output)

                if moderation_result["passed"]:
                    # Extract only the response content
                    validated_response_text = moderation_result["validated_response"].validated_output["bot_output"]["response"]

                    # Save to chat history
                    st.session_state["chat_history"].append({
                        "user": prompt,
                        "base_model": base_output,
                        "guard_model": validated_response_text
                    })

                else:
                    st.error("Guardrails validation failed.")
                    st.write(f"Issues: {moderation_result.get('issues', 'Unknown error')}")

            else:
                st.error(f"Guardrailed Model API Error: {guard_response.status_code}")
                logging.error(f"Guardrailed Model API Error: {guard_response.text}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Guardrailed Model request failed: {e}")
            st.error(f"Guardrailed Model request failed: {e}")

    except Exception as e:
        logging.error(f"Base Model API request failed: {e}")
        st.error(f"Base Model API request failed: {e}")

# Display chat history
st.write("## Chat History")

for idx, entry in enumerate(st.session_state["chat_history"]):
    st.subheader(f"Conversation #{idx + 1}")
    
    # User prompt
    st.markdown(f"### **User Prompt**: {entry['user']}")

    # Side-by-side layout for outputs
    col1, col2 = st.columns(2)

    # Base Model Output
    with col1:
        st.write("### Base Model Output")
        st.code(entry['base_model'], language="text")

    # Guardrailed Model Output
    with col2:
        st.write("### Guardrailed Model Output")
        st.code(entry['guard_model'], language="text")

    # Highlight differences
    st.write("### Differences")
    diff = unified_diff(
        entry['base_model'].splitlines(), 
        entry['guard_model'].splitlines(), 
        lineterm=""
    )
    st.code("\n".join(diff), language="diff")

    st.markdown("---")

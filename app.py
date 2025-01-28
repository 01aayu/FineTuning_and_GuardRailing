import streamlit as st
import pandas as pd
import requests, os, yaml, re, json, logging
from ollama import chat
from guardrails import Guard
from datetime import datetime
import xml.etree.ElementTree as ET
from difflib import unified_diff
from transformers import pipeline

# Paths
config_yaml_path = os.path.join("config", "config.yml")
rails_xml_path = os.path.join("config", "rails.xml")

# Load jailbreak commands from the CSV file
csv_path = "data/jailbreaks_dataset.csv" 
try:
    jailbreak_commands_df = pd.read_csv(csv_path)
    jailbreak_commands = jailbreak_commands_df['Command'].tolist() 
except Exception as e:
    logging.error(f"Error loading jailbreak commands from CSV: {e}")
    jailbreak_commands = []

# Setup logging
logging.basicConfig(
    filename="guard/logs.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

st.title("MAFLONG - Medical Assist Finetuned LLM using Ollama and Nemo Guardrails")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Load Guardrails XML configuration
def load_guardrails_config():
    try:
        # Load YAML configuration
        with open(config_yaml_path, "r") as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)
            print("YAML Config Loaded:", yaml_config)
        # Load XML configuration using the rails_xml_path variable
        tree = ET.parse(rails_xml_path)
        xml_str = ET.tostring(tree.getroot(), encoding="unicode")
        return xml_str
    except Exception as e:
        logging.error(f"Error loading Guardrails config: {e}")
        st.error(f"Error loading Guardrails config: {e}")
        print(f"Error loading YAML or XML file: {e}")
        return None      

# Initialize Guard instance
rail_config = load_guardrails_config()
if rail_config:
    guard = Guard.from_rail_string(rail_config)
else:
    guard = None

# Load a fine-tuned model for detecting jailbreak prompts
classifier = pipeline("text-classification", model="madhurjindal/Jailbreak-Detector")

def detect_jailbreak_dynamic(input_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Check against commands in the CSV file
    for command in jailbreak_commands:
        if command.lower() in input_text.lower():
            log_message = f"[{timestamp}] Jailbreak attempt detected: '{command}' | Input: '{input_text}'"
            logging.warning(log_message)
            # Log the jailbreak attempt
            with open("jailbreak_attempts.log", "a") as log_file:
                log_file.write(log_message + "\n")
            return True, command, timestamp
    return False, command, timestamp

def moderate_response(bot_output):
    if not guard:
        return {"passed": False, "issues": "Guardrails configuration not loaded."}
    try:
        # Validate bot output using guardrails
        structured_output = json.dumps({"bot_output": {"response": bot_output}})
        validated_output = guard.parse(structured_output)
        return {
            "passed": True,
            "validated_response": validated_output
        }
    except Exception as e:
        logging.error(f"Guardrails validation failed: {e}")
        return {
            "passed": False,
            "issues": str(e)
        }

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
    # Detect jailbreak attempts in user input
    jailbreak_detected, score, timestamp = detect_jailbreak_dynamic(prompt)
    if jailbreak_detected:
        # Log and notify user of the jailbreak attempt
        st.warning(f"Jailbreak attempt detected. Proceeding with the prompt.")
        logging.warning(f"Jailbreak attempt detected: {prompt} | Logged at: {timestamp}")

    # Proceed to generate responses regardless of jailbreak detection
    with st.spinner("Generating response, please wait..."):
        try:
            # Get response from base model via Ollama API
            base_response = chat(
                model='gemma',
                messages=[{"role": "user", "content": prompt}]
            )
            base_output = base_response.message.content

            # Payload for guardrailed model
            guardrailed_payload = {
                "model": 'gemma1.1_medical',
                "prompt": prompt,
                "max_tokens": 150
            }

            # Get response from guardrailed model locally
            try:
                guard_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=guardrailed_payload,
                    stream=True,
                    timeout=30
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
                            "guard_model": validated_response_text,
                            "jailbreak_detected": jailbreak_detected,
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
    # Base Model Output
    st.write("### Base Model Output")
    st.code(entry['base_model'], language="text")
    # Guardrailed Model Output
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


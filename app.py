import json
import os

import streamlit as st

from data_handling.data_loaders import load_json_data
from graph_operations.graph_builder import build_graph, prune_nodes_by_degree_threshold
from graph_operations.graph_utils import get_top_percentile_scores, \
    check_goals_and_run_pagerank
from llm_chain.node_name_extraction import extract_goal_tech


def failure_list(description: str, top_percentile: int = 85):
    extraction_result = extract_goal_tech(description)
    if "error" not in extraction_result:
        specified_goals = extraction_result.get("goals", [])
        specified_technologies = extraction_result.get("technologies", [])
        start_nodes = set(specified_goals).union(set(specified_technologies))
        result = check_goals_and_run_pagerank(G, start_nodes)
        if isinstance(result, str):
            return json.dumps({"status": "info", "message": result})
        else:
            top_scores = get_top_percentile_scores(result, percentile=top_percentile)
            if top_scores:  # Ensure there is data to process
                # Extract names and scores into separate lists
                failures = list(top_scores.keys())
                scores = list(top_scores.values())

                # Construct and return the JSON object containing these lists
                return json.dumps({
                    "status": "success",
                    "failure_modes": failures,
                    "scores": scores
                })
            else:
                # Return an error JSON if no data is available
                return json.dumps({
                    "status": "error",
                    "message": f"No data available for the top {top_percentile} percentile"
                })
    else:
        return json.dumps({"status": "info", "message": "No goals or technologies extracted."})


file_path = None
for path in ['data/aiidb_full.json', '/mnt/code/data/aiidb_full.json']:
    if os.path.exists(path):
        file_path = path
        break

if 'process_complete' not in st.session_state:
    incident_data = load_json_data(file_path)
    # incident_data = sample_incidents(incident_data, n=30)
    G = build_graph(incident_data)
    G = prune_nodes_by_degree_threshold(G, degree_threshold=1)


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Tell me about your AI system and I'll tell you what might lead to failures"}
    ]


# App title
st.set_page_config(page_title="RAIAssist", layout="wide")

# App sidebar
with st.sidebar:
    # App sidebar
    st.write(
        "<h1>Hi, I'm <font color='#ffcdc2'>RAI-Bot</font> - your AI system helper</h1>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.button("Clear Chat ", on_click=clear_chat_history, type="primary")

# Store generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Tell me about your AI system and I'll tell you where to dig deeper"}
    ]

# And display all stored chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Seek new input from user
if prompt := st.chat_input("Chat with RAI Assist"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            user_response = "I'm sorry, I couldn't find any failure modes for your AI system."
            response = failure_list(prompt)
            if response:
                # Parse the JSON response
                response_dict = json.loads(response)
                if response_dict["status"] == "success":
                    # Extract the failure modes
                    failure_modes = response_dict.get("failure_modes", [])

                    # Convert the failure modes to a Markdown list
                    markdown_list = "\n".join(f"{i + 1}. {mode}" for i, mode in enumerate(failure_modes))

                    # Introductory text
                    intro_text = """The following is a list of identified areas to test, check for the AI system. 
                    These areas represent different types of failures or biases that may occur within the system. 
                    Addressing these issues is crucial for responsibly deploying the AI system:"""

                    # Display the introductory text and Markdown list in Streamlit
                    user_response = f"{intro_text} \n {markdown_list}"

    st.markdown(user_response.lstrip())
    message = {"role": "assistant", "content": user_response}
    st.session_state.messages.append(message)

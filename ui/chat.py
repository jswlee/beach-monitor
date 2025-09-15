"""
Simple Streamlit chat UI for beach monitoring
"""
import streamlit as st
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from agent.graph import BeachMonitorAgent

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Kaanapali Beach Monitor",
        page_icon="🏖️",
        layout="wide"
    )
    
    st.title("🏖️ Kaanapali Beach Monitor")
    st.markdown("Ask me about current beach conditions!")
    
    # Initialize agent
    if "agent" not in st.session_state:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("Please set your OPENAI_API_KEY in the .env file")
            st.stop()
        
        st.session_state.agent = BeachMonitorAgent(openai_api_key=openai_api_key)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How busy is the beach now?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                # Check if the user is responding to the image offer
                if st.session_state.get("show_image_prompt"):
                    if prompt.lower() in ["yes", "sure", "ok", "yep"]:
                        annotated_image_path = st.session_state.get("annotated_image_path")
                        if annotated_image_path and Path(annotated_image_path).exists():
                            st.image(annotated_image_path, caption="Annotated Beach Snapshot")
                            response = "Here is the annotated image!"
                        else:
                            response = "I'm sorry, I couldn't find the annotated image."
                        st.session_state.show_image_prompt = False
                    else:
                        response = "No problem. Let me know if you need anything else!"
                        st.session_state.show_image_prompt = False
                else:
                    agent_state = st.session_state.agent.query(prompt)
                    response = agent_state['messages'][-1].content
                    annotated_image_path = agent_state.get('annotated_image_path')

                    if annotated_image_path:
                        st.session_state.annotated_image_path = annotated_image_path
                        st.session_state.show_image_prompt = True
                    else:
                        st.session_state.show_image_prompt = False

            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This beach monitor uses AI to analyze live camera feeds from Kaanapali Beach.
        
        **What I can tell you:**
        - Current number of people and boats
        - Beach activity level (quiet, moderate, busy)
        - Real-time conditions
        
        **Example questions:**
        - "How busy is the beach now?"
        - "Are there many people in the water?"
        - "What's the current beach activity?"
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()

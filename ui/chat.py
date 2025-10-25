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
            # Display any images first
            if "image_path" in message and message["image_path"]:
                image_path = message["image_path"]
                if Path(image_path).exists():
                    caption = message.get("image_caption", "Beach Image")
                    st.image(str(image_path), caption=caption, width="stretch")
            
            # Then display the text
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
                snapshot_displayed = False  # Initialize for all branches
                image_to_save = None  # Initialize image path for chat history
                image_caption = None  # Initialize caption for chat history
                
                if st.session_state.get("show_image_prompt"):
                    if prompt.lower() in ["yes", "sure", "ok", "yep", "yeah", "show me"]:
                        annotated_image_path = st.session_state.get("annotated_image_path")
                        if annotated_image_path and Path(annotated_image_path).exists():
                            st.image(annotated_image_path, caption="Annotated Beach Snapshot", width="stretch")
                            response = "Here is the annotated image showing all detected people and boats!"
                            # Store image path for persistence
                            image_to_save = str(annotated_image_path)
                            image_caption = "Annotated Beach Snapshot"
                        else:
                            response = "I'm sorry, I couldn't find the annotated image."
                            image_to_save = None
                            image_caption = None
                        st.session_state.show_image_prompt = False
                    elif prompt.lower() in ["no", "nope", "no thanks", "skip"]:
                        response = "No problem. Let me know if you need anything else!"
                        image_to_save = None
                        image_caption = None
                        st.session_state.show_image_prompt = False
                    else:
                        # User asked a new question - clear the prompt flag and process normally
                        st.session_state.show_image_prompt = False
                        agent_state = st.session_state.agent.query(prompt)
                        response = agent_state['messages'][-1].content
                        annotated_image_path = agent_state.get('annotated_image_path')
                        snapshot_path = agent_state.get('snapshot_path')

                        # Display snapshot if available
                        if snapshot_path:
                            if not Path(snapshot_path).is_absolute():
                                snapshot_path = Path.cwd() / snapshot_path
                            
                            if Path(snapshot_path).exists():
                                if "original" in response.lower():
                                    image_caption = "Original Beach Snapshot (from recent analysis)"
                                else:
                                    image_caption = "Fresh Beach Snapshot"
                                st.image(str(snapshot_path), caption=image_caption, width="stretch")
                                snapshot_displayed = True
                                image_to_save = str(snapshot_path)
                            else:
                                st.warning(f"Image file not found: {snapshot_path}")
                                image_to_save = None
                                image_caption = None
                        else:
                            image_to_save = None
                            image_caption = None

                        if annotated_image_path:
                            st.session_state.annotated_image_path = annotated_image_path
                            st.session_state.show_image_prompt = True
                else:
                    agent_state = st.session_state.agent.query(prompt)
                    response = agent_state['messages'][-1].content
                    annotated_image_path = agent_state.get('annotated_image_path')
                    snapshot_path = agent_state.get('snapshot_path')
                    
                    # Display snapshot if available (from either tool)
                    if snapshot_path:
                        # Convert to absolute path if needed
                        if not Path(snapshot_path).is_absolute():
                            snapshot_path = Path.cwd() / snapshot_path
                        
                        if Path(snapshot_path).exists():
                            # Determine caption based on which tool was used
                            if "original" in response.lower():
                                image_caption = "Original Beach Snapshot (from recent analysis)"
                            else:
                                image_caption = "Fresh Beach Snapshot"
                            
                            try:
                                st.image(str(snapshot_path), caption=image_caption, width="stretch")
                                snapshot_displayed = True
                                image_to_save = str(snapshot_path)
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                                st.write(f"Path: {snapshot_path}")
                                image_to_save = None
                                image_caption = None
                        else:
                            st.warning(f"Image file not found at: {snapshot_path}")
                            image_to_save = None
                            image_caption = None
                    else:
                        image_to_save = None
                        image_caption = None
                    
                    # Check if we have an annotated image to offer
                    if annotated_image_path:
                        st.session_state.annotated_image_path = annotated_image_path
                        st.session_state.show_image_prompt = True
                    else:
                        st.session_state.show_image_prompt = False

            # Display response text
            if snapshot_displayed:
                # If we showed an image, show a cleaner message
                if "original" in response.lower():
                    st.markdown("Here is the original image from the recent analysis:")
                else:
                    st.markdown("Here is a fresh snapshot of Kaanapali Beach:")
            else:
                st.markdown(response)

        # Add assistant response to chat history with image if present
        message_data = {
            "role": "assistant", 
            "content": response
        }
        if image_to_save:
            message_data["image_path"] = image_to_save
            message_data["image_caption"] = image_caption
        
        st.session_state.messages.append(message_data)
    
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
        - "How many people are on the beach vs the water?"
        - "What's the current beach activity?"
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()

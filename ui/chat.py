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
            # Display the text first (if any)
            if message["content"]:
                st.markdown(message["content"])

            # Then display any images
            if "image_path" in message and message["image_path"]:
                image_path = message["image_path"]
                if Path(image_path).exists():
                    caption = message.get("image_caption", "Beach Image")
                    st.image(str(image_path), caption=caption, use_column_width=True)
    
    # Chat input
    if prompt := st.chat_input("How busy is the beach now?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                # Get agent response
                agent_state = st.session_state.agent.query(prompt)
                response = agent_state['messages'][-1].content
                snapshot_path = agent_state.get('snapshot_path')
                image_caption = agent_state.get('image_caption')
                counts_text = agent_state.get('counts_text')
                
                # Debug logging
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"LLM Response: {response}")
                logger.info(f"Counts text: {counts_text}")
                logger.info(f"Snapshot path: {snapshot_path}")

                # First display the main text response (so narration appears before images)
                # Strip any markdown image lines like '![Label](...)' to avoid broken
                # image links such as a stray 'Segmented Image' label.
                cleaned_response = "".join(
                    line + "\n"
                    for line in response.splitlines()
                    if not line.lstrip().startswith("![")
                ).strip()

                primary_text = cleaned_response if cleaned_response else (counts_text or "")
                if primary_text:
                    st.markdown(primary_text)

                # Then display image if agent provided one
                image_to_save = None
                snapshot_displayed = False
                if snapshot_path:
                    # Convert to absolute path if needed
                    if not Path(snapshot_path).is_absolute():
                        snapshot_path = Path.cwd() / snapshot_path
                    else:
                        snapshot_path = Path(snapshot_path)
                    
                    if snapshot_path.exists():
                        caption = image_caption or "Beach Image"
                        st.image(str(snapshot_path), caption=caption, use_column_width=True)
                        # Optionally emphasize counts under the image
                        if counts_text and counts_text != primary_text:
                            st.markdown(f"**{counts_text}**")
                        snapshot_displayed = True
                        image_to_save = str(snapshot_path)
                    else:
                        st.error(f"Image file not found at: {snapshot_path}")

        # Add assistant response to chat history with image if present
        # Use same primary_text that was shown live so history ordering matches
        content_to_save = primary_text
        save_content = content_to_save if content_to_save.strip() else ""
        
        message_data = {
            "role": "assistant", 
            "content": save_content
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
        - Number of people in the water and on the beach
        - Beach activity level (quiet, moderate, busy) with number of people and boats
        - Qualitative description of weather conditions
        
        **Example questions:**
        - "How many people are on the beach vs the water?"
        - "How busy is the beach now?"
        - "What's the weather like?"
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()

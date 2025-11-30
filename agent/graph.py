"""
LangGraph agent for beach monitoring queries
"""
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from agent.api_client import get_api_client
from .tools import (
    capture_snapshot_tool,
    analyze_beach_tool,
    get_weather_tool,
    get_original_image_tool,
    get_annotated_image_tool,
    get_regions_image_tool,
)

logger = logging.getLogger(__name__)

# Define the agent state
class AgentState(Dict):
    messages: List[BaseMessage]
    annotated_image_path: str = None
    snapshot_path: str = None


class BeachMonitorAgent:
    """Beach monitoring conversational agent"""
    
    def __init__(self, openai_api_key: str = None):
        self.tools = [
            capture_snapshot_tool,
            analyze_beach_tool,
            get_weather_tool,
            get_original_image_tool,
            get_annotated_image_tool,
            get_regions_image_tool,
        ]
        self.tool_executor = ToolExecutor(self.tools)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=openai_api_key
        ).bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("action", self._call_tool)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "action",
                "end": END,
            }
        )
        
        # Add edge from action back to agent
        workflow.add_edge("action", "agent")
        
        return workflow.compile()
    
    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the LLM with current state"""
        messages = state["messages"]
        
        # Add system message for beach monitoring context
        system_message = HumanMessage(content="""You are a helpful beach monitoring assistant for Kaʻanapali Beach. 
        You can provide real-time information about beach activity including people and boat counts, as well as 
        counts for total number of people on the beach vs in the water.
        
        Available tools:
        1. capture_snapshot_tool - Capture snapshot (force_new=True for fresh, False for cache)
        2. analyze_beach_tool - Analyze beach activity (DEFAULT: force_fresh=True, always uses current data)
        3. get_weather_tool - Get weather conditions from snapshot
        4. get_original_image_tool - Get most recent original snapshot (no new capture)
        5. get_annotated_image_tool - Get most recent annotated image (no new capture)
        6. get_regions_image_tool - Get most recent segmented image (no new capture)
        
        Workflow:
        - "show me the beach" or "show me the beach NOW" → capture_snapshot_tool with force_new=True
        - "how many people" → analyze_beach_tool (defaults to fresh analysis)
        - "beach vs water" → analyze_beach_tool (defaults to fresh analysis)
        - "show me the original image" → get_original_image_tool
        - "show me the annotated image" → get_annotated_image_tool
        - "show me the segmented image" → get_regions_image_tool
        - "what's the weather" → get_weather_tool
        
        NOTE: analyze_beach_tool ALWAYS uses fresh data by default (force_fresh=True).
        Only set force_fresh=False if user explicitly asks for cached/historical analysis.
        
        CRITICAL RULES:
        - When user asks to "show" an image, ONLY call the image tool. Do NOT add any text response.
        - For count questions (especially "beach vs water" or "people on beach vs in water"): You MUST call BOTH analyze_beach_tool AND get_regions_image_tool in the same response. The segmented image must always be shown for these questions. Then provide a concise text summary with the specific numbers (e.g., "There are 9 people total: 3 on the beach and 5 in the water").
        - The UI will display images automatically from tool outputs. Never mention file paths or links in your response.
        """)
        
        if not any(isinstance(msg, HumanMessage) and "beach monitoring assistant" in msg.content for msg in messages):
            messages = [system_message] + messages
        
        response = self.llm.invoke(messages)
        return {"messages": messages + [response]}
    
    def _call_tool(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools based on the last AI message"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Execute tools
        tool_calls = last_message.tool_calls
        tool_messages = []
        
        annotated_image_path = None
        snapshot_path = None
        regions_image_path = None
        image_caption = None
        counts_text = None
        people_count = None
        
        called_tools = []
        for tool_call in tool_calls:
            tool_result = self.tool_executor.invoke(
                ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"]
                )
            )

            result_text = str(tool_result)
            tool_name = tool_call["name"]
            called_tools.append(tool_name)
            
            logger.info(f"Tool {tool_name} returned: {result_text[:200]}")
            
            # Extract paths and set captions based on tool name and output
            if tool_name == "analyze_beach_tool":
                # Try to parse people_count from the formatted analysis text
                for line in result_text.splitlines():
                    if "People:" in line:
                        # Expected format: "👥 People: X total" (or similar)
                        try:
                            part = line.split("People:")[-1]
                            num_str = part.strip().split()[0]
                            people_count = int(num_str)
                        except Exception:
                            pass
                # Also look for a downloaded annotated image path
                if "Image saved at:" in result_text:
                    path_part = result_text.split("Image saved at:")[-1]
                    annotated_image_path = path_part.split("\n")[0].strip()
                    snapshot_path = annotated_image_path
                    image_caption = "Beach Image"

            elif tool_name == "get_annotated_image_tool" and "Annotated image is available at:" in result_text:
                path_part = result_text.split("Annotated image is available at: ")[-1]
                annotated_image_path = path_part.split("\n")[0].strip()
                snapshot_path = annotated_image_path
                image_caption = "Annotated Beach Snapshot"
                
            elif tool_name == "get_regions_image_tool" and "Regions image is available at:" in result_text:
                # Only use the segmented image if we actually detected people.
                # When people_count == 0, prefer showing the original/annotated image
                # from the analysis instead of a stale segmented image.
                if people_count is None or people_count > 0:
                    path_part = result_text.split("Regions image is available at: ")[-1]
                    regions_image_path = path_part.split("\n")[0].strip()
                    snapshot_path = regions_image_path
                    image_caption = "Segmented Image (Beach vs Water)"
                
            elif tool_name == "get_original_image_tool" and "Image saved at:" in result_text:
                path_part = result_text.split("Image saved at:")[-1]
                snapshot_path = path_part.split("\n")[0].strip()
                image_caption = "Original Beach Snapshot"
                
            elif tool_name == "capture_snapshot_tool" and "Image saved at:" in result_text:
                path_part = result_text.split("Image saved at:")[-1]
                snapshot_path = path_part.split("\n")[0].strip()
                image_caption = "Fresh Beach Snapshot"

            # Extract counts text if present
            if "\nCounts:" in result_text:
                counts_text = result_text.split("\nCounts:")[-1].strip()
                counts_text = "Counts: " + counts_text

            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)

        # If the analysis found zero people, prefer showing the original, unmasked
        # beach image instead of any masked/annotated image from detection.
        if people_count == 0:
            try:
                client = get_api_client()
                original_path = client.get_latest_original_image()
                if original_path:
                    snapshot_path = original_path
                    image_caption = "Beach Image"
            except Exception as e:
                logger.warning(f"Failed to get original image for zero-people case: {e}")

        # Suppress image display if no explicit image tool was called
        explicit_image_tools = {
            "capture_snapshot_tool",
            "get_original_image_tool",
            "get_annotated_image_tool",
            "get_regions_image_tool",
            "analyze_beach_tool",
        }
        if not any(t in explicit_image_tools for t in called_tools):
            snapshot_path = None
            image_caption = None

        return {
            "messages": messages + tool_messages,
            "annotated_image_path": annotated_image_path,
            "snapshot_path": snapshot_path,
            "image_caption": image_caption,
            "counts_text": counts_text
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to execute them
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        else:
            return "end"
    
    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user query and return response
        
        Args:
            user_input: User's question about beach conditions
            
        Returns:
            Agent's response
        """
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=user_input)]
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Return the final state
            return result
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "messages": [AIMessage(content="I'm sorry, there was an error processing your request.")],
                "annotated_image_path": None
            }

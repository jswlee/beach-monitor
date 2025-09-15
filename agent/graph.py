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
from .tools import beach_status_tool, get_weather_tool

logger = logging.getLogger(__name__)

# Define the agent state
class AgentState(Dict):
    messages: List[BaseMessage]
    annotated_image_path: str = None


class BeachMonitorAgent:
    """Beach monitoring conversational agent"""
    
    def __init__(self, openai_api_key: str = None):
        self.tools = [beach_status_tool, get_weather_tool]
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
        system_message = HumanMessage(content="""You are a helpful beach monitoring assistant for Kaanapali Beach. 
        You can provide real-time information about beach activity including people and boat counts.
        
        If the user asks about how busy the beach is, use the beach_status_tool to get current information.
        After providing the summary, ask the user if they would like to see the annotated image.
        
        If the user asks about weather conditions, use the get_weather_tool to get current information.
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
        for tool_call in tool_calls:
            tool_result = self.tool_executor.invoke(
                ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"]
                )
            )

            # Extract the annotated image path from the tool result
            if "Annotated image is available at:" in str(tool_result):
                annotated_image_path = str(tool_result).split("Annotated image is available at: ")[-1].strip()

            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)

        return {
            "messages": messages + tool_messages,
            "annotated_image_path": annotated_image_path
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

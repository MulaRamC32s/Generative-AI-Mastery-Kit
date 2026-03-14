from typing import List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# Mock Tools for the Agent
@tool
def calculate_area(length: float, width: float) -> float:
    """Calculates the area of a rectangle."""
    return length * width

@tool
def get_current_date() -> str:
    """Returns the current date for context."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_product_availability(product_name: str) -> str:
    """Mocks a database search for product availability."""
    # Logic for DB search would go here
    availability = {
        "laptop": "In Stock (12 units)",
        "monitor": "Out of Stock",
        "keyboard": "Limited Stock (3 units)"
    }
    status = availability.get(product_name.lower(), "Unknown Product")
    return f"Product '{product_name}' status: {status}"

class AgentConfig(BaseModel):
    """Configuration for the LLM Agent."""
    model_name: str = Field(default="gpt-4-0125-preview")
    temperature: float = Field(default=0)
    system_message: str = Field(
        default="You are a helpful and professional AI assistant with access to tools."
    )

class AgentManager:
    """Manages the lifecycle and execution of LLM Agents."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature
        )
        self.tools = [calculate_area, get_current_date, search_product_availability]
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self) -> AgentExecutor:
        """Creates the LangChain Agent Executor."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, input_text: str, chat_history: Optional[List[Any]] = None) -> str:
        """Executes the agent with the given input."""
        history = chat_history or []
        response = self.agent_executor.invoke({
            "input": input_text,
            "chat_history": history
        })
        return response["output"]

# Example usage for testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    manager = AgentManager()
    output = manager.run("What is the area of a rectangle with length 15 and width 32?")
    print(f"Agent Response: {output}")
    
    output2 = manager.run("What date is it today?")
    print(f"Agent Response: {output2}")
    
    output3 = manager.run("Is the keyboard in stock?")
    print(f"Agent Response: {output3}")

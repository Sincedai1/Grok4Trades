"""
Base class for all agents in the QuantumSol system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type
from pydantic import BaseModel, Field
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads import Run
import logging
import json

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class AgentTool(BaseModel):
    """Base class for agent tools."""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters."""
        pass

class BaseAgent(ABC, Generic[T]):
    """Base class for all agents in the QuantumSol system."""
    
    def __init__(
        self,
        name: str,
        description: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tools: Optional[List[AgentTool]] = None,
        **kwargs
    ):
        self.name = name
        self.description = description
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []
        self.client = None  # Will be set by the agent runner
        self.assistant: Optional[Assistant] = None
        self.thread_id: Optional[str] = None
        self.run: Optional[Run] = None
        
        # Initialize any additional state
        self.initialize(**kwargs)
    
    def initialize(self, **kwargs):
        """Initialize any additional state for the agent."""
        pass
    
    @abstractmethod
    async def process_message(self, message: str, **kwargs) -> T:
        """Process a message and return a response."""
        pass
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name with the given parameters."""
        for tool in self.tools:
            if tool.name == tool_name:
                return await tool.execute(**kwargs)
        raise ValueError(f"Tool {tool_name} not found")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the agent to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": [tool.dict() for tool in self.tools] if self.tools else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseAgent':
        """Create an agent from a dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

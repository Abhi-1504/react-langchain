from typing import Any, Dict, List

from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running"""
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("*" * 10)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running"""
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("*" * 10)

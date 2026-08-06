"""LLM abstract interface and default OpenAI implementation"""
from abc import ABC, abstractmethod
from typing import List, Optional
import os
import json
import re


class BaseLLM(ABC):
    """Abstract base class for LLMs"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text"""
        pass

    @abstractmethod
    async def verify(self, claim: str, evidence: str) -> float:
        """Verify the relevance of a claim against evidence; return a 0-1 score"""
        pass

    @abstractmethod
    async def extract_skills(self, trajectories: List[str]) -> str:
        """Extract shared skills/patterns from multiple trajectories"""
        pass

    @abstractmethod
    async def refine_skill(self, skill_text: str, feedback: str) -> str:
        """Rewrite/refine a skill based on feedback"""
        pass

    @abstractmethod
    async def attribute_failure(self, context: str, feedback: str) -> dict:
        """Attribute the cause of failure; return {'type': 'connection'|'unit', 'action': 'expand'|'prune'|'reshape', 'details': ...}"""
        pass

    @abstractmethod
    async def reshape_content(
        self, node_content: str, target_granularity: str, context: str
    ) -> str:
        """Reshape the granularity of node content"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI API implementation"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAILLM. "
                "Please install it with: pip install openai"
            ) from exc

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = AsyncOpenAI(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text via the OpenAI Chat Completions API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    async def verify(self, claim: str, evidence: str) -> float:
        """Verify the relevance of a claim against evidence; return a 0-1 score"""
        system_prompt = (
            "You are a verification assistant. Your task is to judge how well "
            "the evidence supports the claim. Return ONLY a single float number "
            "between 0.0 and 1.0, where 0.0 means completely unrelated and 1.0 "
            "means the evidence fully supports the claim."
        )
        user_prompt = (
            f"Claim: {claim}\n\nEvidence: {evidence}\n\n"
            "Relevance score (0.0-1.0):"
        )
        raw = await self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )
        # Try to extract a floating-point number from the returned text
        match = re.search(r"(\d+\.?\d*)", raw.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.0

    async def extract_skills(self, trajectories: List[str]) -> str:
        """Extract shared skills/patterns from multiple trajectories"""
        system_prompt = (
            "You are an expert at analyzing task trajectories and extracting "
            "shared skills and common patterns. Given multiple trajectories, "
            "identify the recurring strategies, decision patterns, and skills "
            "that are shared across them. Present the extracted skills in a "
            "structured format."
        )
        traj_text = "\n\n---\n\n".join(
            f"Trajectory {i+1}:\n{t}" for i, t in enumerate(trajectories)
        )
        user_prompt = (
            f"Please analyze the following trajectories and extract the shared "
            f"skills/patterns:\n\n{traj_text}"
        )
        return await self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
        )

    async def refine_skill(self, skill_text: str, feedback: str) -> str:
        """Rewrite/refine a skill based on feedback"""
        system_prompt = (
            "You are a skill refinement assistant. You will be given a current "
            "skill description and feedback about its performance. Rewrite and "
            "improve the skill based on the feedback while preserving the core "
            "intention of the original skill."
        )
        user_prompt = (
            f"Current skill:\n{skill_text}\n\n"
            f"Feedback:\n{feedback}\n\n"
            "Please refine the skill based on the feedback above:"
        )
        return await self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
        )

    async def attribute_failure(self, context: str, feedback: str) -> dict:
        """Attribute the cause of failure"""
        system_prompt = (
            "You are a failure analysis assistant. Given a context and failure "
            "feedback, determine the type of failure and recommended action. "
            "You MUST respond with a valid JSON object with these keys:\n"
            '- "type": either "connection" (failure due to missing/wrong link '
            "between nodes) or \"unit\" (failure within a single node's content)\n"
            '- "action": either "expand" (add more content/connections), '
            '"prune" (remove irrelevant content), or "reshape" (change granularity)\n'
            '- "details": a string explaining the reasoning\n\n'
            "Example response:\n"
            '{"type": "connection", "action": "expand", '
            '"details": "Missing link between concept A and concept B"}'
        )
        user_prompt = (
            f"Context:\n{context}\n\nFailure feedback:\n{feedback}\n\n"
            "Analyze the failure and respond with JSON:"
        )
        raw = await self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        )
        # Try to parse JSON
        try:
            # Extract the JSON block (it may be wrapped in a markdown code block)
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(raw)
            # Validate required fields
            if "type" not in result or "action" not in result:
                return {
                    "type": "unit",
                    "action": "reshape",
                    "details": f"Failed to parse structured failure attribution. Raw: {raw}",
                }
            if result["type"] not in ("connection", "unit"):
                result["type"] = "unit"
            if result["action"] not in ("expand", "prune", "reshape"):
                result["action"] = "reshape"
            if "details" not in result:
                result["details"] = ""
            return result
        except (json.JSONDecodeError, ValueError):
            return {
                "type": "unit",
                "action": "reshape",
                "details": f"Failed to parse failure attribution response. Raw: {raw}",
            }

    async def reshape_content(
        self, node_content: str, target_granularity: str, context: str
    ) -> str:
        """Reshape the granularity of node content"""
        system_prompt = (
            "You are a content reshaping assistant. You will be given node "
            "content, a target granularity level, and surrounding context. "
            "Rewrite the content to match the specified granularity level while "
            "preserving the key information and semantic meaning.\n\n"
            "Granularity levels:\n"
            "- 'fine': Break down into detailed, specific sub-points\n"
            "- 'medium': Moderate level of detail, balanced summary\n"
            "- 'coarse': High-level abstract summary"
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Current node content:\n{node_content}\n\n"
            f"Target granularity: {target_granularity}\n\n"
            "Please reshape the content:"
        )
        return await self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
        )

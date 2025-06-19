#!/usr/bin/env python3
"""Agent definitions for tool counting experiments.

This module provides a global agent variable that can be dynamically configured
with custom tool names via importlib manipulation.
"""

import os
import random
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class ExperimentDeps:
    """Dependencies for the experiment - tracks tool call count."""

    call_count: int = 0


TOOL_NAME = os.getenv("AGENT_TOOL_NAME", "foo")

# singleton agent with dynamic tool name
agent = Agent[ExperimentDeps, int](
    name="Tool Counter Agent",
    instructions=f"""
    You are an agent whose job is to call the '{TOOL_NAME}' tool a random number of times (between 1 and 100).

    After calling the tool the chosen number of times, respond with the total number of calls made.

    You should:
    1. Decide on a random number between 1-100 (you can pick any number in this range)
    2. Call the {TOOL_NAME} tool that many times
    3. Return the final count as an integer
    """,
    deps_type=ExperimentDeps,
    output_type=int,
)


@agent.tool(name=TOOL_NAME)
def counting_tool(ctx: RunContext[ExperimentDeps]) -> str:
    """Tool that increments the counter and returns a random choice."""
    ctx.deps.call_count += 1
    choices = [
        "fish",
        "dog",
        "mouse",
        "snake",
    ]
    return random.choice(choices)

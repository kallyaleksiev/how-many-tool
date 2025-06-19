#!/usr/bin/env python3
"""Simple pydantic-ai experiment to count tool calls.

An agent that calls a tool a random number of times and tracks the count.
"""

import asyncio
import importlib
import os
import sys
from collections import Counter
from typing import Dict, Tuple

import click


def get_agent_with_tool_name(tool_name: str):
    """Dynamically import agent with custom tool name"""

    os.environ["AGENT_TOOL_NAME"] = tool_name

    # Remove the module from cache if it exists to force reimport
    if "agents" in sys.modules:
        del sys.modules["agents"]

    # Import the agents module, which will now use the new tool name
    agents_module = importlib.import_module("agents")

    return agents_module.agent, agents_module.ExperimentDeps


async def run_single_experiment(
    agent, experiment_deps_class, model: str, experiment_id: int
) -> Tuple[int, int, bool]:
    """Run a single experiment and return (reported_count, actual_count, is_accurate)."""
    deps = experiment_deps_class(call_count=0)

    try:
        result = await agent.run(
            "Please call the tool a random number of times between 1-100, then tell me the total count.",
            deps=deps,
            model=model,
        )
        reported_count = result.output
        actual_count = deps.call_count
        is_accurate = reported_count == actual_count

        return reported_count, actual_count, is_accurate
    except Exception as e:
        click.echo(f"Experiment {experiment_id} failed: {e}", err=True)
        return 0, deps.call_count, False


async def run_experiments_concurrently(
    agent, experiment_deps_class, model: str, num_experiments: int
) -> Dict:
    """Run experiments concurrently and return detailed statistics."""
    click.echo(f"🔄 Running {num_experiments} experiments concurrently with model: {model}")

    # Run experiments concurrently
    tasks = [
        run_single_experiment(agent, experiment_deps_class, model, i + 1)
        for i in range(num_experiments)
    ]
    results = await asyncio.gather(*tasks)

    # Extract data
    reported_counts = [r[0] for r in results]
    actual_counts = [r[1] for r in results]
    accuracies = [r[2] for r in results]

    # Calculate statistics
    accuracy_rate = sum(accuracies) / len(accuracies) * 100
    actual_count_frequencies = Counter(actual_counts)
    most_common_count, most_common_frequency = actual_count_frequencies.most_common(1)[0]
    most_common_percentage = (most_common_frequency / num_experiments) * 100

    return {
        "model": model,
        "num_experiments": num_experiments,
        "accuracy_rate": accuracy_rate,
        "most_common_count": most_common_count,
        "most_common_percentage": most_common_percentage,
        "reported_counts": reported_counts,
        "actual_counts": actual_counts,
        "accuracies": accuracies,
        "actual_count_frequencies": actual_count_frequencies,
    }


@click.command()
@click.option(
    "--model",
    "-m",
    default="anthropic:claude-4-sonnet-20250514",
    help="Model name to use for the agent",
)
@click.option("--experiments", "-n", default=10, help="Number of experiments to run")
@click.option(
    "--tool-name",
    "-t",
    default="foo",
    help="Name of the tool that the agent will call",
)
def main(model: str, experiments: int, tool_name: str):
    """Run pydantic-ai tool counting experiments with configurable model and count."""

    async def run():
        # Get agent with custom tool name using dynamic import
        agent, experiment_deps_class = get_agent_with_tool_name(tool_name)
        click.echo(f"🤖 Loaded agent with tool name: '{tool_name}'")

        stats = await run_experiments_concurrently(agent, experiment_deps_class, model, experiments)

        click.echo("\n📊 Results:")
        click.echo(f"Model: {stats['model']}")
        click.echo(f"Experiments: {stats['num_experiments']}")
        click.echo(f"Accuracy: {stats['accuracy_rate']:.1f}%")
        click.echo(
            f"Most common actual calls: {stats['most_common_count']} ({stats['most_common_percentage']:.1f}%)"
        )

        click.echo("\n📈 Detailed Statistics:")
        click.echo(f"Actual call distribution: {dict(stats['actual_count_frequencies'])}")
        click.echo(
            f"Average actual calls: {sum(stats['actual_counts']) / len(stats['actual_counts']):.1f}"
        )
        click.echo(
            f"Min/Max actual calls: {min(stats['actual_counts'])}/{max(stats['actual_counts'])}"
        )

        return stats

    asyncio.run(run())


if __name__ == "__main__":
    main()

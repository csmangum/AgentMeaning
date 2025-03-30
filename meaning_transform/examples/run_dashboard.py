#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script to run the interactive agent state dashboard.

This script:
1. Generates sample agent states
2. Starts the interactive dashboard
3. Allows exploration of agent states and their relationships
"""

import argparse

import numpy as np

from meaning_transform.src.data import AgentState
from meaning_transform.src.interactive import AgentStateDashboard, run_dashboard


def generate_sample_agents(n_agents=20):
    """Generate sample agent states for the dashboard."""
    agents = []

    # Create sample roles
    roles = ["explorer", "gatherer", "defender", "builder", "leader"]

    # Create some sample agents
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        role = roles[i % len(roles)]
        position = (
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-1, 1),
        )
        health = np.random.uniform(0.5, 1.0)
        energy = np.random.uniform(0.3, 1.0)

        # Create inventory based on role
        inventory = {}
        if role == "gatherer":
            inventory = {
                "wood": np.random.randint(5, 20),
                "stone": np.random.randint(5, 15),
            }
        elif role == "explorer":
            inventory = {"map": 1, "food": np.random.randint(1, 5)}
        elif role == "defender":
            inventory = {"sword": 1, "shield": np.random.randint(0, 2)}
        elif role == "builder":
            inventory = {
                "wood": np.random.randint(10, 30),
                "stone": np.random.randint(10, 20),
                "tools": np.random.randint(1, 3),
            }
        elif role == "leader":
            inventory = {"plan": 1, "food": np.random.randint(3, 8)}

        # Create goals based on role
        goals = []
        if role == "gatherer":
            goals = ["gather resources", "return to base"]
        elif role == "explorer":
            goals = ["explore map", "find resources"]
        elif role == "defender":
            goals = ["patrol area", "protect agents"]
        elif role == "builder":
            goals = ["build structure", "gather materials"]
        elif role == "leader":
            goals = ["coordinate agents", "plan strategy"]

        # Create agent
        agent = AgentState(
            position=position,
            health=health,
            energy=energy,
            inventory=inventory,
            role=role,
            goals=goals,
            agent_id=agent_id,
            step_number=0,
            resource_level=np.random.uniform(0.2, 0.9),
            current_health=health * np.random.uniform(0.8, 1.0),
            is_defending=role == "defender",
            age=np.random.randint(10, 100),
            total_reward=np.random.uniform(10, 100),
        )

        agents.append(agent)

    return agents


def main(args):
    """Main function to run the dashboard."""
    print(f"Starting Agent State Dashboard on port {args.port}")
    print("Generate sample data using the 'Load Sample Data' button")

    # Run the dashboard
    dashboard = AgentStateDashboard(port=args.port)
    dashboard.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Interactive Agent State Dashboard"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the dashboard on"
    )

    args = parser.parse_args()
    main(args)

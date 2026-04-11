"""Plot dynamic subplots (one per agent) with variable number of goals."""

import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
EXPERIMENT_NAME = "bagtest"
FILE_TEMPLATE = "{exp_name}_{agent}_goal_{goal_number}.csv"


def _discover_agents_and_goals(exp_name):
    """
    Scan outputs/ directory for files matching {exp_name}_{agent}_goal_{goal_number}.csv.
    Returns:
      agents: sorted list of unique agent names
      goal_data: dict mapping agent -> sorted list of goal numbers
    """
    agent_goal_pairs = set()
    
    pattern = os.path.join(OUTPUT_DIR, f"{exp_name}_*_goal_*.csv")
    files = glob.glob(pattern)
    regex = rf"{re.escape(exp_name)}_([a-zA-Z0-9_]+)_goal_(\d+)\.csv"
    
    for filepath in files:
        filename = os.path.basename(filepath)
        match = re.match(regex, filename)
        if match:
            agent, goal_number = match.groups()
            agent_goal_pairs.add((agent, int(goal_number)))
    
    # Organize by agent
    agent_goal_map = {}
    for agent, goal_number in agent_goal_pairs:
        if agent not in agent_goal_map:
            agent_goal_map[agent] = []
        agent_goal_map[agent].append(goal_number)
    
    # Sort agents and goal numbers within each agent
    agents = sorted(agent_goal_map.keys())
    for agent in agents:
        agent_goal_map[agent].sort()
    
    return agents, agent_goal_map


def _load_one_csv(path):
    """
    Load CSV file with required columns.
    """
    if os.path.exists(path):
        df = pd.read_csv(path, sep=";")
        required = {"x", "y", "x_goal", "y_goal"}
        missing = required.difference(df.columns)
        if missing:
            print(f"[WARN] {path} missing columns: {sorted(missing)}")
            return None
        return df
    
    print(f"[WARN] Missing file: {path}")
    return None


def _get_color_from_goal(goal_number, colormap_name="viridis", num_goals=None):
    """
    Map goal number to color from a matplotlib colormap.
    For discrete colormaps like tab10, uses modulo indexing.
    Args:
        goal_number: integer goal number (1-indexed)
        colormap_name: name of the colormap to use
        num_goals: total number of goals (for reference, not used for tab10)
    Returns:
        RGBA color tuple
    """
    cmap = cm.get_cmap(colormap_name)
    
    # For discrete colormaps, use modulo indexing
    # Assuming goal_number is 1-indexed, convert to 0-indexed
    color_index = (goal_number - 1) % cmap.N
    return cmap(color_index)


# Discover agents and goals from the outputs directory
agents, goal_data = _discover_agents_and_goals(EXPERIMENT_NAME)

if not agents:
    print(f"[ERROR] No files found matching pattern: {EXPERIMENT_NAME}_*_goal_*.csv in outputs/")
    exit(1)

# Calculate all unique goal numbers across all agents for consistent colormap normalization
all_goal_numbers = set()
for goals in goal_data.values():
    all_goal_numbers.update(goals)
num_goals_total = len(all_goal_numbers)

# Create subplots dynamically based on number of agents
num_agents = len(agents)
num_cols = min(3, num_agents)  # Max 3 cols for reasonable layout
num_rows = (num_agents + num_cols - 1) // num_cols  # Ceiling division
figsize = (6 * num_cols, 5 * num_rows)

fig, axes_flat = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=False, sharey=False)

# Handle single agent case (subplots returns 1D array, not 2D)
if num_agents == 1:
    axes_flat = [axes_flat]
elif num_rows == 1 or num_cols == 1:
    axes_flat = axes_flat.flatten()
else:
    axes_flat = axes_flat.flatten()

# Plot trajectories for each agent
for ax, agent in zip(axes_flat[:num_agents], agents):
    goals = sorted(goal_data.get(agent, []))
    
    for goal_number in goals:
        csv_name = FILE_TEMPLATE.format(
            exp_name=EXPERIMENT_NAME,
            agent=agent,
            goal_number=goal_number
        )
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        df = _load_one_csv(csv_path)
        if df is None or df.empty:
            continue

        # Get color based on goal number using discrete tab10 colors
        color = _get_color_from_goal(goal_number, colormap_name="tab10")
        ax.plot(df["x"], df["y"], color=color, linewidth=1.5, label=f"Goal {goal_number} trajectory")
        ax.plot(df["x"].iloc[0], df["y"].iloc[0], "o", color=color, markersize=7)
        gx, gy = df["x_goal"].iloc[0], df["y_goal"].iloc[0]
        ax.plot(gx, gy, "*", color=color, markersize=13, markeredgecolor="k", markeredgewidth=0.5)

    ax.set_title(agent.upper(), fontsize=14, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for ax in axes_flat[num_agents:]:
    ax.set_visible(False)

fig.suptitle("Agent Trajectories & Goals", fontsize=16, fontweight="bold")
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_trajectories.png")
plt.savefig(save_path, dpi=150)
print(f"Saved to {save_path}")
print(f"\nDiscovered agents: {agents}")
print(f"Goal distribution: {goal_data}")
plt.show()
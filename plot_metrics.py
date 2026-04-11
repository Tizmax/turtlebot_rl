#!/usr/bin/env python3
"""
Plot experiment metrics curves from aggregated experiment data.

Usage:
    python plot_metrics.py --experiment <name> [--output-dir .]
    
For each agent, generates a 2x2 subplot figure with:
1. Normalized reach time per goal (bar chart)
2. Distance evolution per goal (line plot)
3. Success rate per goal (line plot)
4. Cumulative reward evolution per goal (line plot)
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')


def load_trial_results(experiment_name):
    """Load trial results from all {experiment}_{agent}.csv files - aggregates all agents"""
    results_by_goal = defaultdict(list)
    
    # Find all results files matching pattern {experiment}_{agent}.csv
    from pathlib import Path
    results_files = list(Path("outputs").glob(f"{experiment_name}_*.csv"))
    
    # Filter to get only main results files (not goal-specific ones which have _goal_ in the name)
    main_results_files = [f for f in results_files if "_goal_" not in f.name]
    
    if not main_results_files:
        print(f"Warning: No results files found for {experiment_name}")
        return results_by_goal
    
    print(f"Found {len(main_results_files)} agent results files for experiment '{experiment_name}'")
    
    for results_file in main_results_files:
        agent_name = results_file.stem.replace(f"{experiment_name}_", "")
        print(f"  Loading agent: {agent_name}")
        
        try:
            with open(results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    goal_idx = int(row["goal_in_sequence"])
                    results_by_goal[goal_idx].append(row)
        except Exception as e:
            print(f"Warning: Error reading {results_file}: {e}")
    
    return results_by_goal


def load_aggregated_trajectories(experiment_name):
    """Load and aggregate trajectory data from {experiment}_{agent}_goal_{idx}.csv files"""
    trajectories = defaultdict(lambda: defaultdict(list))  # traj[controller][goal_idx] = [steps]
    reward_trajectories = defaultdict(lambda: defaultdict(list))  # reward[controller][goal_idx] = [steps]
    
    # Find all goal CSV files for this experiment
    from pathlib import Path
    parent_dir = Path("outputs")
    goal_files = list(parent_dir.glob(f"{experiment_name}_*_goal_*.csv"))
    
    if not goal_files:
        print(f"Warning: No goal position CSV files found for {experiment_name}")
        return trajectories, reward_trajectories
    
    # Parse each goal file and aggregate
    for goal_file in goal_files:
        # Extract controller and goal_idx from filename: {exp}_{controller}_goal_{idx}.csv
        stem = goal_file.stem  # Remove .csv
        parts = stem.split("_goal_")
        if len(parts) != 2:
            continue
        
        # Extract goal_idx
        goal_part = parts[1]
        try:
            goal_idx = int(goal_part)
        except ValueError:
            continue
        
        controller_part = parts[0].replace(f"{experiment_name}_", "")
        controller = controller_part
        
        # Read position samples and aggregate by step
        samples_by_trial = defaultdict(list)
        reward_by_trial = defaultdict(list)
        
        try:
            with open(goal_file, "r") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    trial = int(row["trial"])
                    timestamp_s = float(row["timestamp_s"])
                    x = float(row["x"])
                    y = float(row["y"])
                    x_goal = float(row["x_goal"])
                    y_goal = float(row["y_goal"])
                    cumulative_reward = float(row.get("cumulative_reward", 0.0))
                    
                    # Calculate distance to goal
                    dist_to_goal = math.sqrt((x - x_goal)**2 + (y - y_goal)**2)
                    
                    samples_by_trial[trial].append({
                        "time": timestamp_s,
                        "dist": dist_to_goal,
                    })
                    
                    reward_by_trial[trial].append({
                        "time": timestamp_s,
                        "cumulative_reward": cumulative_reward,
                    })
        except Exception as e:
            print(f"Warning: Could not read {goal_file}: {e}")
            continue
        
        # Aggregate distance trajectories across trials
        if samples_by_trial:
            max_steps = max(len(s) for s in samples_by_trial.values())
            for step_num in range(max_steps):
                times = []
                dists = []
                
                for trial_samples in samples_by_trial.values():
                    if step_num < len(trial_samples):
                        sample = trial_samples[step_num]
                        times.append(sample["time"])
                        dists.append(sample["dist"])
                
                if times:
                    mean_time = np.mean(times)
                    std_time = np.std(times) if len(times) > 1 else 0
                    mean_dist = np.mean(dists)
                    std_dist = np.std(dists) if len(dists) > 1 else 0
                    
                    trajectories[controller][goal_idx].append({
                        "step": step_num,
                        "time": mean_time,
                        "dist": mean_dist,
                        "dist_std": std_dist,
                    })
        
        # Aggregate reward trajectories across trials
        if reward_by_trial:
            max_steps = max(len(r) for r in reward_by_trial.values())
            for step_num in range(max_steps):
                times = []
                rewards = []
                
                for trial_rewards in reward_by_trial.values():
                    if step_num < len(trial_rewards):
                        sample = trial_rewards[step_num]
                        times.append(sample["time"])
                        rewards.append(sample["cumulative_reward"])
                
                if times:
                    mean_time = np.mean(times)
                    mean_reward = np.mean(rewards)
                    
                    reward_trajectories[controller][goal_idx].append({
                        "step": step_num,
                        "time": mean_time,
                        "cumulative_reward": mean_reward,
                    })
    
    return trajectories, reward_trajectories


def load_reach_stats(experiment_name):
    """Load reach time statistics from all {experiment}_{agent}.csv files"""
    reach_stats = defaultdict(lambda: defaultdict(lambda: {"reach_times": [], "normalized_reach_times": []}))
    success_rates_by_goal = {}
    
    # Find all results files matching pattern {experiment}_{agent}.csv
    from pathlib import Path
    results_files = list(Path("outputs").glob(f"{experiment_name}_*.csv"))
    
    # Filter to get only main results files (not goal-specific ones which have _goal_ in the name)
    main_results_files = [f for f in results_files if "_goal_" not in f.name]
    
    if not main_results_files:
        print(f"Warning: No results files found for {experiment_name}")
        return reach_stats, success_rates_by_goal
    
    for results_file in main_results_files:
        agent_name = results_file.stem.replace(f"{experiment_name}_", "")
        
        try:
            with open(results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get("goal_in_sequence") or row["goal_in_sequence"].strip() == "":
                        continue
                    
                    goal_idx = int(row["goal_in_sequence"])
                    controller = row["controller"]
                    
                    # Load reach statistics (pre-computed and stored in CSV)
                    mean_reach = float(row["mean_reach_time_s"])
                    mean_norm_reach = float(row["mean_normalized_reach_time"])
                    
                    reach_stats[controller][goal_idx] = {
                        "mean_reach_time": mean_reach,
                        "mean_normalized_reach_time": mean_norm_reach,
                    }
                    
                    # Track success rate per goal (use first occurrence)
                    if goal_idx not in success_rates_by_goal:
                        success_rates_by_goal[goal_idx] = float(row["success_rate"])
        except Exception as e:
            print(f"Warning: Error reading {results_file}: {e}")
    
    return reach_stats, success_rates_by_goal


def plot_reach_time_normalized(ax, reach_stats, agent, goal_indices):
    """Plot normalized reach time (reach_time / initial_distance) as bar plot per goal"""
    
    norm_reach_times = []
    found_data = False
    
    for goal_idx in goal_indices:
        if agent in reach_stats and goal_idx in reach_stats[agent]:
            norm_time = reach_stats[agent][goal_idx]["mean_normalized_reach_time"]
            norm_reach_times.append(norm_time)
            found_data = True
        else:
            norm_reach_times.append(0)
    
    if not found_data:
        ax.text(0.5, 0.5, f"No data for {agent}", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return
    
    x = np.arange(len(goal_indices))
    bars = ax.bar(x, norm_reach_times, color="#1f77b4", alpha=0.8, edgecolor="white", linewidth=2)
    
    ax.set_xlabel("Goal Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Reach Time (s/m)", fontsize=12, fontweight="bold")
    ax.set_title("Normalized Reach Time per Goal", fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(goal_indices, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=10)


def plot_success_rate(ax, success_rates, goal_indices):
    """Plot success rate per goal"""
    rates = [success_rates.get(g, 0) for g in goal_indices]
    
    ax.plot(goal_indices, rates, marker="o", linewidth=3, markersize=10, 
            color="#9467bd", markerfacecolor="#9467bd", markeredgewidth=2, 
            markeredgecolor="white", alpha=0.9, label="Success Rate")
    ax.set_xlabel("Goal Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Success Rate", fontsize=12, fontweight="bold")
    ax.set_title("Success Rate per Goal", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(goal_indices)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=10)
    
    # Add percentage labels on points
    for i, rate in zip(goal_indices, rates):
        ax.text(i, rate + 0.05, f"{rate*100:.0f}%", ha="center", fontsize=9, fontweight="bold")


def plot_reach_time_per_goal(ax, aggregated_trajectories, agent, goal_indices):
    """Plot distance over time for each goal (1 line per goal)"""
    
    if agent not in aggregated_trajectories or not aggregated_trajectories[agent]:
        ax.text(0.5, 0.5, f"No data for {agent}", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return
    
    goals = aggregated_trajectories[agent]
    colors_goals = plt.cm.tab10(np.linspace(0, 1, len(goal_indices)))
    
    for i, goal_idx in enumerate(goal_indices):
        if goal_idx not in goals:
            continue
        
        steps = goals[goal_idx]
        if not steps:
            continue
        
        # Sort by time
        steps_sorted = sorted(steps, key=lambda x: x["time"])
        
        # Normalize time to start at 0 from first measurement
        start_time = steps_sorted[0]["time"]
        times = [s["time"] - start_time for s in steps_sorted]
        distances = [s["dist"] for s in steps_sorted]
        
        ax.plot(times, distances, marker="o", label=f"Goal {goal_idx}", 
                color=colors_goals[i], linewidth=2.5, markersize=5, alpha=0.8)
    
    ax.set_xlabel("Time from Goal Start (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance to Goal (m)", fontsize=12, fontweight="bold")
    ax.set_title("Distance Evolution per Goal", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, ncol=2, framealpha=0.95, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=10)


def plot_cumulative_reward(ax, reward_trajectories, agent, goal_indices):
    """Plot cumulative reward over time for each goal (1 line per goal)"""
    
    if agent not in reward_trajectories or not reward_trajectories[agent]:
        ax.text(0.5, 0.5, f"No data for {agent}", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return
    
    goals = reward_trajectories[agent]
    colors_goals = plt.cm.tab10(np.linspace(0, 1, len(goal_indices)))
    
    for i, goal_idx in enumerate(goal_indices):
        if goal_idx not in goals:
            continue
        
        steps = goals[goal_idx]
        if not steps:
            continue
        
        # Sort by time
        steps_sorted = sorted(steps, key=lambda x: x["time"])
        
        # Normalize time to start at 0 from first measurement
        start_time = steps_sorted[0]["time"]
        times = [s["time"] - start_time for s in steps_sorted]
        rewards = [s["cumulative_reward"] for s in steps_sorted]
        
        ax.plot(times, rewards, marker="o", label=f"Goal {goal_idx}", 
                color=colors_goals[i], linewidth=2.5, markersize=5, alpha=0.8)
    
    ax.set_xlabel("Time from Goal Start (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax.set_title("Cumulative Reward Evolution per Goal", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, ncol=2, framealpha=0.95, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=10)


def main():
    parser = argparse.ArgumentParser(
        description="Plot experiment metrics curves"
    )
    parser.add_argument(
        "--experiment",
        default="experiment",
        help="Experiment name (default: experiment)",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Output directory for PNG",
    )
    parser.add_argument(
        "--no-show-plot",
        action="store_true",
        help="Skip matplotlib.show()",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading experiment data for '{args.experiment}'...")
    results_by_goal = load_trial_results(args.experiment)
    aggregated_trajectories, reward_trajectories = load_aggregated_trajectories(args.experiment)
    reach_stats, success_rates = load_reach_stats(args.experiment)
    
    if not aggregated_trajectories:
        print("Error: No data loaded. Check that CSV files exist and are properly formatted.")
        return
    
    goal_indices = sorted(results_by_goal.keys())
    agents = sorted(aggregated_trajectories.keys())
    
    print(f"Found agents: {agents}")
    
    # Generate one figure per agent
    for agent in agents:
        print(f"Generating figures for agent: {agent}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot 1: Normalized reach time per goal (bar plot)
        plot_reach_time_normalized(axes[0], reach_stats, agent, goal_indices)
        
        # Plot 2: Distance evolution per goal (line plot)
        plot_reach_time_per_goal(axes[1], aggregated_trajectories, agent, goal_indices)
        
        # Plot 3: Success rate per goal
        plot_success_rate(axes[2], success_rates, goal_indices)
        
        # Plot 4: Cumulative reward evolution per goal (line plot)
        plot_cumulative_reward(axes[3], reward_trajectories, agent, goal_indices)
        
        fig.tight_layout(pad=3.0)
        
        # Add overall title
        fig.suptitle(f"Experiment: {args.experiment} | Agent: {agent.upper()}", 
                     fontsize=16, fontweight="bold", y=0.995)
        
        # Save figure
        output_path = Path(args.output_dir) / f"{args.experiment}_{agent}_analysis_curves.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"  ✓ Plot saved to {output_path}")
        
        if not args.no_show_plot:
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close(fig)


if __name__ == "__main__":
    main()

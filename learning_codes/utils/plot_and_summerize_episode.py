import matplotlib.pyplot as plt
from Region import RegionController
from typing import Dict
import os
import re

PERIODIC_SUMMARY_PLOT_FREQ = 10

def _generate_summary_plots_from_files(region_id: str, summary_dir: str):
    """
    Helper function to parse summary text files and generate plots of 
    average loss/reward vs. episode number.
    """
    summary_file_path = os.path.join(summary_dir, f"region_{region_id}_summary.txt")

    if not os.path.exists(summary_file_path):
        print(f"Warning: Summary file not found for region {region_id}. Skipping summary plot generation.")
        return

    # Delete old summary plots before creating new ones
    for plot_type in ["avg_loss", "avg_reward"]:
        old_plot_path = os.path.join(summary_dir, f"region_{region_id}_{plot_type}_summary.png")
        try:
            os.remove(old_plot_path)
        except FileNotFoundError:
            pass # It's okay if the file doesn't exist yet

    episodes, avg_losses, avg_rewards = [], [], []
    with open(summary_file_path, 'r') as f:
        content = f.read()
    episode_blocks = re.findall(r"Episode (\d+).*?Loss -> Avg: (None|-?[\d.]+).*?Reward -> Avg: (None|-?[\d.]+)", content, re.DOTALL)    
    for episode, loss_str, reward_str in episode_blocks:
        episodes.append(int(episode))
        # Append data only if it's not 'None'
        if loss_str != "None":
            avg_losses.append((int(episode), float(loss_str)))
        if reward_str != "None":
            avg_rewards.append((int(episode), float(reward_str)))

    if avg_losses:
        # Sort by episode number
        avg_losses.sort(key=lambda x: x[0])
        e_vals, l_vals = zip(*avg_losses)
        plt.figure()
        plt.plot(e_vals, l_vals, marker='o', linestyle='-')
        plt.title(f"Region {region_id} - Average Loss per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        plt.grid(True)
        loss_plot_path = os.path.join(summary_dir, f"region_{region_id}_avg_loss_summary.png")
        plt.savefig(loss_plot_path)
        plt.close()
    
    if avg_rewards:
        # Sort by episode number
        avg_rewards.sort(key=lambda x: x[0])
        e_vals, r_vals = zip(*avg_rewards)
        plt.figure()
        plt.plot(e_vals, r_vals, marker='o', linestyle='-')
        plt.title(f"Region {region_id} - Average Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(True)
        reward_plot_path = os.path.join(summary_dir, f"region_{region_id}_avg_reward_summary.png")
        plt.savefig(reward_plot_path)
        plt.close()


def plot_and_summarize_episode(episode: int, _region_controllers: Dict[str, RegionController], num_episodes: int):
    """
    Processes episode data for each region to:
      - Create a detailed folder per region for its plots.
      - Plot per-episode loss and reward, saving them in the region's folder.
      - Create a central 'summary' folder.
      - Append episode summary stats (avg, min, max) to a text file in the summary folder.
      - Every 10 episodes, generate summary plots (avg loss/reward vs episode) from the text files.
      - At the very end, generate and save aggregated plots comparing all episodes for each region.
    """
    summary_dir = "logs/region_logs/summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    for region_id, region_controller in _region_controllers.items():
        region_plot_dir = os.path.join("logs","region_logs", f"region_{region_id}_in_detail")
        os.makedirs(region_plot_dir, exist_ok=True)

        logs = region_controller.episode_data
        if not logs:
            continue

        # --- Data Extraction ---
        train_steps = [entry["train_step"] for entry in logs]
        time_axis = [ts * 20 for ts in train_steps]
        
        losses = []
        for entry in logs:
            try:
                loss_val = float(entry["loss"])
            except:
                loss_val = None
            losses.append(loss_val)
        rewards = [entry["region_reward"] for entry in logs]
        
        valid_loss_data = [(t, l) for t, l in zip(time_axis, losses) if l is not None]

        # --- Per-Episode Plotting (saved in region-specific folder) ---
        if valid_loss_data:
            x_vals, y_vals = zip(*valid_loss_data)
            plt.figure()
            plt.plot(x_vals, y_vals)
            plt.title(f"Region {region_id}, Episode {episode} - Loss")
            plt.xlabel("Simulation Time (seconds)")
            plt.ylabel("Loss")
            loss_plot_path = os.path.join(region_plot_dir, f"region_{region_id}_episode_{episode}_loss.png")
            plt.savefig(loss_plot_path)
            plt.close()

        plt.figure()
        plt.plot(time_axis, rewards)
        plt.title(f"Region {region_id}, Episode {episode} - Reward")
        plt.xlabel("Simulation Time (seconds)")
        plt.ylabel("Reward")
        reward_plot_path = os.path.join(region_plot_dir, f"region_{region_id}_episode_{episode}_reward.png")
        plt.savefig(reward_plot_path)
        plt.close()

        # --- Summary File Generation (saved in central summary folder) ---
        avg_loss = sum(y_vals) / len(y_vals) if valid_loss_data else None
        min_loss = min(y_vals) if valid_loss_data else None
        max_loss = max(y_vals) if valid_loss_data else None
        
        avg_reward = sum(rewards) / len(rewards) if rewards else None
        min_reward = min(rewards) if rewards else None
        max_reward = max(rewards) if rewards else None

        summary_text = (
            f"#############################################\n"
            f"Episode {episode} Summary for Region {region_id}:\n"
            f"  Loss -> Avg: {avg_loss}, Min: {min_loss}, Max: {max_loss}\n"
            f"  Reward -> Avg: {avg_reward}, Min: {min_reward}, Max: {max_reward}\n"
        )
        
        summary_path = os.path.join(summary_dir, f"region_{region_id}_summary.txt")
        mode = "a" if os.path.exists(summary_path) and episode > 1 else "w"
        with open(summary_path, mode) as sf:
            sf.write(summary_text)

        # Update aggregated data for later aggregated plotting.
        if not hasattr(region_controller, "aggregated_loss_data"):
            region_controller.aggregated_loss_data = []
        if not hasattr(region_controller, "aggregated_reward_data"):
            region_controller.aggregated_reward_data = []

        region_controller.aggregated_loss_data.append({
            "episode": episode,
            "time_axis": time_axis,
            "losses": losses
        })
        region_controller.aggregated_reward_data.append({
            "episode": episode,
            "time_axis": time_axis,
            "rewards": rewards
        })
        region_controller.episode_data.clear() # TODO move to Main_test

    # --- Periodic Summary Plot Generation (from summary text files) ---
    if episode % PERIODIC_SUMMARY_PLOT_FREQ == 0:
        print(f"Generating summary plots at episode {episode}...")
        for region_id in _region_controllers.keys():
            _generate_summary_plots_from_files(region_id, summary_dir)

    # --- Final Aggregated Plot Generation (at the very end) ---
    if episode == num_episodes:
        print("Generating final aggregated plots for all episodes...")
        for region_id, region_controller in _region_controllers.items():
            region_plot_dir = os.path.join("logs","region_logs", f"region_{region_id}_in_detail")

            plt.figure()
            for data in region_controller.aggregated_loss_data:
                valid_data = [(t, l) for t, l in zip(data["time_axis"], data["losses"]) if l is not None]
                if valid_data:
                    x_vals, y_vals = zip(*valid_data)
                    plt.plot(x_vals, y_vals, label=f"Ep {data['episode']}", alpha=0.7)
            plt.title(f"Region {region_id} - Aggregated Loss Comparison")
            plt.xlabel("Simulation Time (seconds)")
            plt.ylabel("Loss")
            plt.legend()
            aggregated_loss_plot_path = os.path.join(region_plot_dir, f"region_{region_id}_loss_aggregated.png")
            plt.savefig(aggregated_loss_plot_path)
            plt.close()

            plt.figure()
            for data in region_controller.aggregated_reward_data:
                plt.plot(data["time_axis"], data["rewards"], label=f"Ep {data['episode']}", alpha=0.7)
            plt.title(f"Region {region_id} - Aggregated Reward Comparison")
            plt.xlabel("Simulation Time (seconds)")
            plt.ylabel("Reward")
            plt.legend()
            aggregated_reward_plot_path = os.path.join(region_plot_dir, f"region_{region_id}_reward_aggregated.png")
            plt.savefig(aggregated_reward_plot_path)
            plt.close()
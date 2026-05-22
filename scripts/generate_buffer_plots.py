import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────
results_dir = "results/buffer_ablations"
plots_dir = os.path.join(results_dir, "plots")
data_dir = os.path.join(results_dir, "raw_data")

for d in [results_dir, plots_dir, data_dir]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# INPUT RUN DIRECTORIES
# ─────────────────────────────────────────────
runs_dir = "runs_buffer_ablations"

if not os.path.exists(runs_dir):
    print(f"Error: {runs_dir} does not exist. Run the training first!")
    exit(1)

run_files = os.listdir(runs_dir)

# Key mapping: exp-name prefix → friendly key
prefixes = {
    "k20": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k20__",
    "k50": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k50__",
    "k80": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k80__",
    "k100": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k100__",
    "k150": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k150__",
    "k200": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k200__",
    "k500": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k500__",
}

paths = {}
csv_paths = {}

for key, prefix in prefixes.items():
    dirs = [d for d in run_files if d.startswith(prefix) and os.path.isdir(os.path.join(runs_dir, d))]
    if dirs:
        dirs.sort()
        paths[key] = os.path.join(runs_dir, dirs[-1])
    csv_files = [f for f in run_files if f.startswith(prefix) and f.endswith("_coords.csv")]
    if csv_files:
        csv_files.sort()
        csv_paths[key] = os.path.join(runs_dir, csv_files[-1])

print("Detected Updated Buffer Ablation Paths:")
print(json.dumps(paths, indent=2))
print("Detected Updated Buffer CSV Paths:")
print(json.dumps(csv_paths, indent=2))

# ─────────────────────────────────────────────
# VISUAL STYLING
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": "#1E293B",
    "axes.labelcolor": "#1E293B",
    "axes.edgecolor": "#CBD5E1",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#F1F5F9",
    "axes.titlepad": 16,
    "axes.labelpad": 12,
    "figure.titlesize": 20,
    "figure.titleweight": "bold"
})

# Sequential Indigo sequential theme transitions:
colors = {
    "k20": "#EF4444",   # Red
    "k50": "#F59E0B",   # Amber
    "k80": "#10B981",   # Emerald Green
    "k100": "#06B6D4",  # Cyan
    "k150": "#3B82F6",  # Blue
    "k200": "#6366F1",  # Indigo
    "k500": "#8B5CF6",  # Violet
}

names = {
    "k20": "ES-RND (K=20, Tau=0.95)",
    "k50": "ES-RND (K=50, Tau=0.95)",
    "k80": "ES-RND (K=80, Tau=0.95)",
    "k100": "ES-RND (K=100, Tau=0.95)",
    "k150": "ES-RND (K=150, Tau=0.95)",
    "k200": "ES-RND (K=200, Tau=0.95)",
    "k500": "ES-RND (K=500, Tau=0.95)",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def smooth_data(data, weight=0.9):
    if len(data) == 0:
        return data
    last = data[0]
    smoothed = []
    for point in data:
        sv = last * weight + (1 - weight) * point
        smoothed.append(sv)
        last = sv
    return smoothed

_accumulators = {}

def get_accumulator(run_path):
    if run_path not in _accumulators:
        if run_path is None or not os.path.exists(run_path):
            return None
        ea = EventAccumulator(run_path, size_guidance={"scalars": 0})
        try:
            print(f"  Loading TensorBoard events for {os.path.basename(run_path)}...")
            ea.Reload()
            _accumulators[run_path] = ea
        except Exception as e:
            print(f"  TensorBoard load error for {run_path}: {e}")
            _accumulators[run_path] = None
    return _accumulators[run_path]

def load_tb_scalar(run_path, tag):
    """Load a scalar tag from a TensorBoard event file."""
    ea = get_accumulator(run_path)
    if ea is None:
        return [], []
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals  = [e.value for e in events]
    return steps, vals

# ─────────────────────────────────────────────
# STATS COMPILATION
# ─────────────────────────────────────────────
summary_stats = {}

for key in prefixes.keys():
    run_path = paths.get(key)
    csv_path = csv_paths.get(key)

    stat = {
        "key": key,
        "name": names[key],
        "max_return": 0.0,
        "mean_return": 0.0,
        "final_return": 0.0,
        "noisy_tv_visits": 0,
        "noisy_tv_occupancy_pct": 0.0,
        "first_key_step": None,
        "first_door_step": None,
        "mean_episode_length": 200.0,
        "mean_cos_sim": None,
        "mean_penalty_ratio": None,
    }

    # --- TensorBoard scalars ---
    steps_ret, vals_ret = load_tb_scalar(run_path, "charts/avg_episodic_return")
    if vals_ret:
        stat["max_return"]   = float(np.max(vals_ret))
        stat["mean_return"]  = float(np.mean(vals_ret))
        stat["final_return"] = float(np.mean(vals_ret[-20:])) if len(vals_ret) >= 20 else float(np.mean(vals_ret))

    steps_len, vals_len = load_tb_scalar(run_path, "charts/episodic_length")
    if vals_len:
        stat["mean_episode_length"] = float(np.mean(vals_len))

    steps_cs, vals_cs = load_tb_scalar(run_path, "charts/avg_max_cos_sim")
    if vals_cs:
        stat["mean_cos_sim"] = float(np.mean(vals_cs))

    steps_pr, vals_pr = load_tb_scalar(run_path, "charts/penalty_applied_ratio")
    if vals_pr:
        stat["mean_penalty_ratio"] = float(np.mean(vals_pr))

    # --- First key / first door / TV occupancy from CSV ---
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            total = len(df)
            
            # Noisy TV cell is at (3, 0)
            tv_mask = (df["x"] == 3) & (df["y"] == 0)
            tv_visits = int(tv_mask.sum())
            stat["noisy_tv_visits"] = tv_visits
            stat["noisy_tv_occupancy_pct"] = round(tv_visits / total * 100, 4) if total > 0 else 0.0
            
            # First key step: minimum step where has_key == 1.0
            key_df = df[df["has_key"] == 1.0]
            if not key_df.empty:
                stat["first_key_step"] = int(key_df["step"].min())
                
            # First door step: minimum step where door_unlocked == 1.0
            door_df = df[df["door_unlocked"] == 1.0]
            if not door_df.empty:
                stat["first_door_step"] = int(door_df["step"].min())
        except Exception as e:
            print(f"  Error processing CSV for {key}: {e}")

    summary_stats[key] = stat

# Save CSV + JSON
stats_df = pd.DataFrame(list(summary_stats.values()))
stats_df.to_csv(os.path.join(data_dir, "updated_buffer_stats.csv"), index=False)
with open(os.path.join(data_dir, "updated_buffer_summary.json"), "w") as f:
    json.dump(summary_stats, f, indent=2)
print("Saved updated_buffer_stats.csv and updated_buffer_summary.json")

# ─────────────────────────────────────────────
# PLOT 1: EPISODIC RETURN LEARNING CURVES
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
for key in prefixes.keys():
    run_path = paths.get(key)
    steps, vals = load_tb_scalar(run_path, "charts/avg_episodic_return")
    if steps and vals:
        smoothed = smooth_data(vals, weight=0.92)
        ax.plot(steps, smoothed, label=names[key], color=colors[key], linewidth=2.2, alpha=0.9)
        ax.fill_between(steps,
                         [max(0, v * 0.85) for v in smoothed],
                         [min(1, v * 1.15) for v in smoothed],
                         color=colors[key], alpha=0.08)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Avg Episodic Return (20-ep rolling)", fontsize=13)
ax.set_title("Buffer Size Ablation (Tau=0.95)\nES-RND on 9×9 Noisy-TV Key-Door Gridworld (500k steps)", fontsize=15, weight="bold")
ax.legend(frameon=True, fontsize=11)
ax.set_ylim(bottom=-0.02)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "buffer_success_curves.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved buffer_success_curves.png in {plots_dir}")

# ─────────────────────────────────────────────
# PLOT 2: TV OCCUPANCY BAR CHART
# ─────────────────────────────────────────────
keys_list = list(prefixes.keys())
occupancies = [summary_stats[k]["noisy_tv_occupancy_pct"] for k in keys_list]
bar_colors  = [colors[k] for k in keys_list]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(keys_list)), occupancies, color=bar_colors, edgecolor="#CBD5E1", width=0.6)
ax.set_xticks(range(len(keys_list)))
ax.set_xticklabels([names[k] for k in keys_list], rotation=18, ha="right", fontsize=10)
ax.set_ylabel("Noisy-TV Occupancy %", fontsize=13)
ax.set_title("TV Room Captivity vs Buffer Size (Tau=0.95)", fontsize=15, weight="bold")
for i, (bar, v) in enumerate(zip(bars, occupancies)):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.4f}%",
            ha="center", weight="bold", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "buffer_tv_occupancy.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved buffer_tv_occupancy.png in {plots_dir}")

# ─────────────────────────────────────────────
# PLOT 3: COSINE SIMILARITY CURVES
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
for key in prefixes.keys():
    run_path = paths.get(key)
    steps, vals = load_tb_scalar(run_path, "charts/avg_max_cos_sim")
    if steps and vals:
        smoothed = smooth_data(vals, weight=0.9)
        ax.plot(steps, smoothed, label=names[key], color=colors[key], linewidth=2.2, alpha=0.9)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Mean Cosine Similarity", fontsize=13)
ax.set_title("Embedding Cosine Similarity vs Buffer Size (Tau=0.95)\nMeasures how quickly the buffer saturates", fontsize=15, weight="bold")
ax.legend(frameon=True, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "buffer_cos_sim_curves.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved buffer_cos_sim_curves.png in {plots_dir}")

# ─────────────────────────────────────────────
# PLOT 4: PENALTY RATIO CURVES
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
for key in prefixes.keys():
    run_path = paths.get(key)
    steps, vals = load_tb_scalar(run_path, "charts/penalty_applied_ratio")
    if steps and vals:
        smoothed = smooth_data(vals, weight=0.9)
        ax.plot(steps, smoothed, label=names[key], color=colors[key], linewidth=2.2, alpha=0.9)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Penalty Ratio (fraction of steps penalized)", fontsize=13)
ax.set_title("Penalty Trigger Rate vs Buffer Size (Tau=0.95)\nHigher = more frames suppressed by boredom penalty", fontsize=15, weight="bold")
ax.legend(frameon=True, fontsize=11)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "buffer_penalty_ratio.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved buffer_penalty_ratio.png in {plots_dir}")

# ─────────────────────────────────────────────
# PLOT 5: EPISODE LENGTH CURVES
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
for key in prefixes.keys():
    run_path = paths.get(key)
    steps, vals = load_tb_scalar(run_path, "charts/episodic_length")
    if steps and vals:
        smoothed = smooth_data(vals, weight=0.9)
        ax.plot(steps, smoothed, label=names[key], color=colors[key], linewidth=2.2, alpha=0.9)

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Avg Episode Length (steps)", fontsize=13)
ax.set_title("Episode Length vs Buffer Size (Tau=0.95)\nShorter = agent solving the task more efficiently", fontsize=15, weight="bold")
ax.legend(frameon=True, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "buffer_episode_length.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved buffer_episode_length.png in {plots_dir}")

# ─────────────────────────────────────────────
# PLOT 6: SPATIAL HEATMAPS (Publication-Grade)
# ─────────────────────────────────────────────
def plot_heatmap(key, dest_dir, filename, grid_size=9):
    csv_path = csv_paths.get(key)
    if csv_path is None or not os.path.exists(csv_path):
        print(f"  No CSV for {key}, skipping heatmap.")
        return
    try:
        df = pd.read_csv(csv_path)
        if "x" not in df.columns or "y" not in df.columns:
            print(f"  CSV for {key} missing x/y columns.")
            return
            
        grid = np.zeros((grid_size, grid_size))
        x_coords = df["x"].astype(int).values
        y_coords = df["y"].astype(int).values
        
        # Bounding coordinates to protect against environment changes
        valid_indices = (x_coords >= 0) & (x_coords < grid_size) & (y_coords >= 0) & (y_coords < grid_size)
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        
        np.add.at(grid, (y_coords, x_coords), 1)
        grid_pct = (grid / len(df)) * 100 if len(df) > 0 else grid
        grid_plot = grid_pct[::-1, :]  # Flip to match visualization grid coordinates
        
        plt.figure(figsize=(10.5, 9.5), dpi=300)
        
        # Custom sequential colormap
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ["#FAFBFD", "#E0E7FF", "#A5B4FC", "#6366F1", "#4F46E5", "#3730A3", "#1E1B4B"]
        cmap = LinearSegmentedColormap.from_list("custom_indigo", colors_list, N=256)
        
        # Plot heatmap
        ax = sns.heatmap(grid_plot, cmap=cmap, vmax=2.5, cbar=True, square=True,
                    xticklabels=list(range(grid_size)), yticklabels=list(range(grid_size - 1, -1, -1)), 
                    linewidths=0.5, linecolor="#FFFFFF", 
                    cbar_kws={'shrink': 0.8, 'aspect': 30})
        
        # Modern colorbar styling
        cbar = ax.collections[0].colorbar
        cbar.set_label('Step Occupancy %', fontsize=12, weight='bold', labelpad=12, color="#1E293B")
        cbar.ax.tick_params(labelsize=10, labelcolor="#475569")
        cbar.outline.set_visible(False)
        
        # Format axes
        ax.set_facecolor("#FAFBFD")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#CBD5E1")
            spine.set_linewidth(1.0)
            
        ax.tick_params(axis='both', which='both', length=0, labelsize=10, labelcolor="#475569")
        
        # Helper to draw double-bordered landmarks
        def add_glowing_landmark(ax, x, y, label, border_color, text_color, bg_color, is_dashed=False):
            rect_glow = plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=border_color, lw=4, alpha=0.25)
            rect_core = plt.Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, fill=False, 
                                      edgecolor=border_color, lw=1.8, 
                                      ls="--" if is_dashed else "-", alpha=0.9)
            ax.add_patch(rect_glow)
            ax.add_patch(rect_core)
            
            ax.text(x + 0.5, y + 0.5, label, color=text_color, ha="center", va="center", weight="black", fontsize=7.5,
                     bbox=dict(facecolor=bg_color, edgecolor=border_color, boxstyle="round,pad=0.25", alpha=0.85, lw=1))
        
        # Draw game landmarks (Flipped layout coordinate mapping: col x, row 8-y)
        # 1. START (0,0) - bottom-left cell. row=8, col=0
        add_glowing_landmark(ax, 0, 8, "START\n(0,0)", "#10B981", "#047857", "#ECFDF5")
        
        # 2. KEY (0,8) - top-left cell. row=0, col=0
        add_glowing_landmark(ax, 0, 0, "KEY\n(0,8)", "#2563EB", "#1D4ED8", "#EFF6FF")
        
        # 3. DOOR (4,4) - center cell. row=4, col=4
        add_glowing_landmark(ax, 4, 4, "DOOR\n(4,4)", "#8B5CF6", "#6D28D9", "#F5F3FF", is_dashed=True)
        
        # 4. NOISY TV (3,0) - bottom cell. row=8, col=3
        add_glowing_landmark(ax, 3, 8, "NOISY TV\n(3,0)", "#EF4444", "#B91C1C", "#FEF2F2")
        
        # 5. GOAL (8,8) - top-right cell. row=0, col=8
        add_glowing_landmark(ax, 8, 0, "GOAL\n(8,8)", "#F59E0B", "#B45309", "#FFFBEB")
        
        # 6. Sleek, solid dark slate walls (col 4, rows 0..3 and 5..8 on screen)
        rect_wall_top = plt.Rectangle((4.02, 0.02), 0.96, 3.96, fill=True, facecolor="#1E293B", edgecolor="#475569", lw=1.2, alpha=0.85)
        plt.gca().add_patch(rect_wall_top)
        
        rect_wall_bottom = plt.Rectangle((4.02, 5.02), 0.96, 3.96, fill=True, facecolor="#1E293B", edgecolor="#475569", lw=1.2, alpha=0.85)
        plt.gca().add_patch(rect_wall_bottom)
        
        # 7. PERFORMANCE METRICS OVERLAY BOX (floating premium dashboard widget)
        stat = summary_stats[key]
        cos_sim_val = stat['mean_cos_sim']
        cos_sim_str = f"{cos_sim_val:.3f}" if cos_sim_val is not None else "N/A"
        
        pen_ratio_val = stat['mean_penalty_ratio']
        pen_ratio_str = f"{pen_ratio_val * 100:.1f}%" if pen_ratio_val is not None else "N/A"
        
        stats_text = (
            f"PERFORMANCE METRICS\n"
            f"-------------------\n"
            f"• Success Rate : {stat['final_return'] * 100:.1f}%\n"
            f"• TV Occupancy : {stat['noisy_tv_occupancy_pct']:.4f}%\n"
            f"• Avg Ep Length: {stat['mean_episode_length']:.1f} steps\n"
            f"• Cosine Sim   : {cos_sim_str}\n"
            f"• Boredom Pen. : {pen_ratio_str}"
        )
        plt.text(5.2, 7.8, stats_text, color="#0F172A", ha="left", va="top", fontsize=9.5, weight="medium",
                 bbox=dict(facecolor="#FCFDFE", edgecolor="#CBD5E1", boxstyle="round,pad=0.5", lw=1.0, alpha=0.97))
        
        # Main Title & Coordinates Labels
        plt.title(f"Spatial Occupancy: {names[key]}", fontsize=15, weight="bold", pad=18, color="#0F172A")
        plt.xlabel("X Coordinate (Grid Column)", fontsize=11, labelpad=10, color="#475569")
        plt.ylabel("Y Coordinate (Grid Row)", fontsize=11, labelpad=10, color="#475569")
        
        plt.tight_layout()
        out_path = os.path.join(dest_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved Heatmap: {filename} in {dest_dir}")
    except Exception as e:
        import traceback
        print(f"  Heatmap error for {key}: {e}")
        traceback.print_exc()

print("\nGenerating spatial heatmaps...")
for key in prefixes.keys():
    plot_heatmap(key, plots_dir, f"heatmap_{key}.png")

# ─────────────────────────────────────────────
# SUMMARY TABLE (printed to stdout)
# ─────────────────────────────────────────────
print("\n" + "=" * 90)
print("UPDATED BUFFER SIZE ABLATION SUMMARY (Tau=0.95)")
print("=" * 90)
print(f"{'Config':<32} {'Max Ret':>8} {'Mean Ret':>10} {'Final Ret':>10} {'TV Occ%':>9} {'Cos Sim':>9} {'Pen Ratio':>10}")
print("-" * 90)
for key in prefixes.keys():
    s = summary_stats[key]
    cs  = f"{s['mean_cos_sim']:.4f}"  if s['mean_cos_sim']  is not None else "   N/A"
    pr  = f"{s['mean_penalty_ratio']:.4f}" if s['mean_penalty_ratio'] is not None else "   N/A"
    print(f"{names[key]:<32} {s['max_return']:>8.3f} {s['mean_return']:>10.3f} "
          f"{s['final_return']:>10.3f} {s['noisy_tv_occupancy_pct']:>9.4f} {cs:>9} {pr:>10}")
print("=" * 90)

# Sync to brain artifacts directories (if they exist in the execution environment)
brain_dir_1 = "/Users/jonathandavid/.gemini/antigravity/brain/54789afb-b042-40ad-aa5f-eb4bf44bc7ed/results/updated_buffer_ablation"
brain_dir_2 = "/Users/jonathandavid/.gemini/antigravity/brain/54789afb-b042-40ad-aa5f-eb4bf44bc7ed/results_small_grid/updated_buffer_ablation"

for bdir in [brain_dir_1, brain_dir_2]:
    if os.path.exists(os.path.dirname(bdir)):
        try:
            if os.path.exists(bdir):
                shutil.rmtree(bdir)
            print(f"Syncing results to local environment directory: {bdir}")
            shutil.copytree(results_dir, bdir)
        except Exception as e:
            pass

print("\n--- UPDATED BUFFER ABLATION RESULTS COMPILED SUCCESSFULLY UNDER results/updated_buffer_ablation/! ---")

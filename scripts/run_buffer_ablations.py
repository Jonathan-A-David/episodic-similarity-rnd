import os
import shutil
import subprocess
import sys

# Output runs directory
target_runs_dir = "runs_updated_buffer_ablation"
source_runs_dir = "runs_small_grid"

# Reusable runs mapping: source name fragment -> target name fragment
reusable_runs = [
    {
        "src_folder": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k50__1__1779419445",
        "src_coords": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k50__1__1779419445_coords.csv",
        "dest_folder": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k50__1__1779419445",
        "dest_coords": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k50__1__1779419445_coords.csv"
    },
    {
        "src_folder": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k100__1__1779419643",
        "src_coords": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k100__1__1779419643_coords.csv",
        "dest_folder": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k100__1__1779419643",
        "dest_coords": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k100__1__1779419643_coords.csv"
    },
    {
        "src_folder": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k200__1__1779419891",
        "src_coords": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k200__1__1779419891_coords.csv",
        "dest_folder": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k200__1__1779419891",
        "dest_coords": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k200__1__1779419891_coords.csv"
    },
    {
        "src_folder": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_tau95__1__1779413819",
        "src_coords": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_tau95__1__1779413819_coords.csv",
        "dest_folder": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k500__1__1779413819",
        "dest_coords": "NoisyTVKeyDoorGridWorldSmall-v0__ablation_es_rnd_k500__1__1779413819_coords.csv"
    }
]

# Missing configurations to run
missing_configs = [
    {
        "name": "Buffer Ablation: ES-RND (K=20, Tau=0.95)",
        "args": [
            "--env-id", "NoisyTVKeyDoorGridWorldSmall-v0",
            "--use-penalty",
            "--int-coef", "1.0",
            "--sim-threshold", "0.95",
            "--penalty-multiplier", "0.2",
            "--buffer-size", "20",
            "--zero-buffer-on-done",
            "--noise-scale", "1.0",
            "--total-timesteps", "500000",
            "--num-envs", "4",
            "--num-steps", "128",
            "--exp-name", "ablation_es_rnd_k20",
            "--seed", "1",
            "--output-dir", target_runs_dir,
            "--no-cuda"
        ]
    },
    {
        "name": "Buffer Ablation: ES-RND (K=80, Tau=0.95)",
        "args": [
            "--env-id", "NoisyTVKeyDoorGridWorldSmall-v0",
            "--use-penalty",
            "--int-coef", "1.0",
            "--sim-threshold", "0.95",
            "--penalty-multiplier", "0.2",
            "--buffer-size", "80",
            "--zero-buffer-on-done",
            "--noise-scale", "1.0",
            "--total-timesteps", "500000",
            "--num-envs", "4",
            "--num-steps", "128",
            "--exp-name", "ablation_es_rnd_k80",
            "--seed", "1",
            "--output-dir", target_runs_dir,
            "--no-cuda"
        ]
    },
    {
        "name": "Buffer Ablation: ES-RND (K=150, Tau=0.95)",
        "args": [
            "--env-id", "NoisyTVKeyDoorGridWorldSmall-v0",
            "--use-penalty",
            "--int-coef", "1.0",
            "--sim-threshold", "0.95",
            "--penalty-multiplier", "0.2",
            "--buffer-size", "150",
            "--zero-buffer-on-done",
            "--noise-scale", "1.0",
            "--total-timesteps", "500000",
            "--num-envs", "4",
            "--num-steps", "128",
            "--exp-name", "ablation_es_rnd_k150",
            "--seed", "1",
            "--output-dir", target_runs_dir,
            "--no-cuda"
        ]
    }
]

def setup_workspace():
    print(f"Creating runs target directory: {target_runs_dir}")
    os.makedirs(target_runs_dir, exist_ok=True)
    
    print("\n--- Copying and symlinking compatible prior runs ---")
    for run in reusable_runs:
        src_fold_path = os.path.join(source_runs_dir, run["src_folder"])
        dest_fold_path = os.path.join(target_runs_dir, run["dest_folder"])
        
        src_csv_path = os.path.join(source_runs_dir, run["src_coords"])
        dest_csv_path = os.path.join(target_runs_dir, run["dest_coords"])
        
        # Copy folder
        if os.path.exists(src_fold_path):
            if os.path.exists(dest_fold_path):
                print(f"Destination folder already exists: {dest_fold_path}")
            else:
                print(f"Copying folder: {src_fold_path} -> {dest_fold_path}")
                shutil.copytree(src_fold_path, dest_fold_path)
        else:
            print(f"WARNING: Source folder not found: {src_fold_path}")
            
        # Copy Coords CSV
        if os.path.exists(src_csv_path):
            if os.path.exists(dest_csv_path):
                print(f"Destination CSV already exists: {dest_csv_path}")
            else:
                print(f"Copying CSV: {src_csv_path} -> {dest_csv_path}")
                shutil.copy2(src_csv_path, dest_csv_path)
        else:
            print(f"WARNING: Source CSV not found: {src_csv_path}")
            
    print("Prior runs setup completed.\n")

def run_experiment(config):
    print("=" * 60)
    print(f"Running Experiment: {config['name']}")
    print("=" * 60)
    
    cmd = [
        "venv/bin/python",
        "cleanrl/ppo_rnd_penalty.py"
    ] + config["args"]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Live logging print
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()
        
    process.wait()
    if process.returncode != 0:
        print(f"ERROR: Experiment '{config['name']}' failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    print(f"SUCCESS: Finished '{config['name']}'\n")

if __name__ == "__main__":
    setup_workspace()
    
    print(f"Starting execution of {len(missing_configs)} missing configurations...")
    for config in missing_configs:
        run_experiment(config)
        
    print("\n--- All training runs completed successfully and compiled under runs_updated_buffer_ablation/ ---")

import subprocess
import sys
import os

# Tau ablation at fixed buffer size K=50 on NoisyTVKeyDoorGridWorldSmall-v0
# 5 configurations x 500,000 steps each = 2,500,000 total steps
# All runs stored in runs_buffer50_tau/ for clean separation

configs = [
    {
        "name": "Buffer50 Tau Ablation: ES-RND (K=50, Tau=0.98)",
        "args": [
            "--env-id", "NoisyTVKeyDoorGridWorldSmall-v0",
            "--use-penalty",
            "--int-coef", "1.0",
            "--sim-threshold", "0.98",
            "--penalty-multiplier", "0.2",
            "--buffer-size", "50",
            "--zero-buffer-on-done",
            "--noise-scale", "1.0",
            "--total-timesteps", "500000",
            "--num-envs", "4",
            "--num-steps", "128",
            "--exp-name", "b50_tau98",
            "--seed", "1",
            "--output-dir", "runs_buffer50_tau",
            "--no-cuda"
        ]
    },
    {
        "name": "Buffer50 Tau Ablation: ES-RND (K=50, Tau=0.95)",
        "args": [
            "--env-id", "NoisyTVKeyDoorGridWorldSmall-v0",
            "--use-penalty",
            "--int-coef", "1.0",
            "--sim-threshold", "0.95",
            "--penalty-multiplier", "0.2",
            "--buffer-size", "50",
            "--zero-buffer-on-done",
            "--noise-scale", "1.0",
            "--total-timesteps", "500000",
            "--num-envs", "4",
            "--num-steps", "128",
            "--exp-name", "b50_tau95",
            "--seed", "1",
            "--output-dir", "runs_buffer50_tau",
            "--no-cuda"
        ]
    },
    {
        "name": "Buffer50 Tau Ablation: ES-RND (K=50, Tau=0.90)",
        "args": [
            "--env-id", "NoisyTVKeyDoorGridWorldSmall-v0",
            "--use-penalty",
            "--int-coef", "1.0",
            "--sim-threshold", "0.90",
            "--penalty-multiplier", "0.2",
            "--buffer-size", "50",
            "--zero-buffer-on-done",
            "--noise-scale", "1.0",
            "--total-timesteps", "500000",
            "--num-envs", "4",
            "--num-steps", "128",
            "--exp-name", "b50_tau90",
            "--seed", "1",
            "--output-dir", "runs_buffer50_tau",
            "--no-cuda"
        ]
    },
    {
        "name": "Buffer50 Tau Ablation: ES-RND (K=50, Tau=0.85)",
        "args": [
            "--env-id", "NoisyTVKeyDoorGridWorldSmall-v0",
            "--use-penalty",
            "--int-coef", "1.0",
            "--sim-threshold", "0.85",
            "--penalty-multiplier", "0.2",
            "--buffer-size", "50",
            "--zero-buffer-on-done",
            "--noise-scale", "1.0",
            "--total-timesteps", "500000",
            "--num-envs", "4",
            "--num-steps", "128",
            "--exp-name", "b50_tau85",
            "--seed", "1",
            "--output-dir", "runs_buffer50_tau",
            "--no-cuda"
        ]
    },
    {
        "name": "Buffer50 Tau Ablation: ES-RND (K=50, Tau=0.80)",
        "args": [
            "--env-id", "NoisyTVKeyDoorGridWorldSmall-v0",
            "--use-penalty",
            "--int-coef", "1.0",
            "--sim-threshold", "0.80",
            "--penalty-multiplier", "0.2",
            "--buffer-size", "50",
            "--zero-buffer-on-done",
            "--noise-scale", "1.0",
            "--total-timesteps", "500000",
            "--num-envs", "4",
            "--num-steps", "128",
            "--exp-name", "b50_tau80",
            "--seed", "1",
            "--output-dir", "runs_buffer50_tau",
            "--no-cuda"
        ]
    },
]


def run_experiment(config):
    print("=" * 60)
    print(f"Running: {config['name']}")
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

    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        print(f"ERROR: Experiment '{config['name']}' failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    print(f"SUCCESS: Finished '{config['name']}'\n")


if __name__ == "__main__":
    os.makedirs("runs_buffer50_tau", exist_ok=True)
    print(f"Starting Buffer-K50 Tau Ablation Suite ({len(configs)} configs x 500k steps each)")
    for config in configs:
        run_experiment(config)
    print("All Buffer50 Tau Ablation experiments completed successfully!")

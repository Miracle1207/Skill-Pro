# Skill-Pro — Skill-augmented MDP for LLM Agents

[![Paper](https://img.shields.io/badge/arXiv-2602.01869-b31b1b.svg)](https://arxiv.org/abs/2602.01869)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This is the official implementation of the paper
**[Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](https://arxiv.org/abs/2602.01869)**
(Qirui Mi, Zhijian Ma, Mengyue Yang, Haoxuan Li, Yisen Wang, Haifeng Zhang, Jun Wang).

`Skill-Pro` is a **Skill-augmented Markov Decision Process (Skill-MDP, SMDP)** framework for **LLM Agents**.
On top of a vanilla LLM-Agent decision loop, it introduces an evolvable **Skill Pool**, where each skill is described by a three-part *Initiation / Policy / Termination* schema, similar to the classical Options framework. At every step, the agent first selects an appropriate skill and injects it into the prompt, then lets the LLM produce an action. In parallel, a **Skill Evolution** module continuously generates, refines, and prunes skills based on interaction experience via **Non-Parametric PPO** (semantic-gradient candidate generation + a PPO Gate for verification), enabling online evolution of the agent's capabilities **without any parameter updates**.

---

## Features

- **Skill-MDP / vanilla MDP modes**: switch via `--MDP_type {SMDP, MDP}` to enable or disable skills.
- **Evolvable Skill Pool**: ships with a set of general reasoning skills (StructuredCoT, ReActDecision, HypothesisElimination, etc.), maintained online during training.
- **Multiple skill-selection strategies**: supports `llm_model` (LLM picks the skill) and `llm_topk_lcb` (Top-K retrieval + LCB), among others.
- **Multiple environments**: built on [TextArena](https://github.com/LeonGuertler/TextArena) and [ALFWorld](https://github.com/alfworld/alfworld); already adapted to `Mastermind-v0`, `FrozenLake`, `Hangman`, `alfworld`, and more.
- **Local and remote LLM backends**: local models are served via [vLLM](https://github.com/vllm-project/vllm); remote models are accessed through OpenRouter.
- **Rich experiment logging**: training curves are tracked with [SwanLab](https://swanlab.cn), and per-iteration skill snapshots, rewards, and maintenance logs are written to JSON.
- **Ablation switches**: `--ablation_type` supports `wo_sg / wo_ppo / wo_score / none` for paper-grade reproducibility.

---

## Project Structure

```
ProcMEM/
├── main.py                  # Entry point: parses CLI args and launches SkillMDP
├── run.py                   # Core training / evaluation loop (SkillMDP)
├── data_structures.py       # Skill / Experience dataclasses
├── pool_managers.py         # ExperiencePool / GoldenExperiencePool
├── Skills/
│   ├── skill_pool.py        # Skill Pool (selection, retrieval, maintenance)
│   ├── skill_evolution.py   # Skill evolution and verification
│   └── loss.py              # Log-prob and other training losses
├── utils/
│   ├── local_llm.py         # Local LLM wrapper (vLLM / HF)
│   ├── encode.py            # Text embedding (sentence-transformers)
│   ├── models.py            # Model-side utilities
│   └── utils.py             # Logging paths, random seeds, etc.
├── configs/
│   └── base_config.yaml     # Configuration for ALFWorld and other envs
├── requirements.txt
└── README.md
```

---

## Installation (Windows + Conda)

We recommend **Python 3.10**. Run the following in Anaconda Prompt or PowerShell:

```powershell
git clone https://github.com/<your-org>/ProcMEM.git
cd ProcMEM

conda create -n procmem python=3.10 -y
conda activate procmem

pip install -r requirements.txt
```

> Tip: if `conda activate` fails the first time you use it in PowerShell, run `conda init powershell` once and restart the terminal.

### ALFWorld Data Preparation (optional)

To run ALFWorld tasks, download the data and set the data path (PowerShell syntax):

```powershell
pip install alfworld
$env:ALFWORLD_DATA = "D:\path\to\alfworld_data"
alfworld-download
```

The `$ALFWORLD_DATA` placeholder in `configs/base_config.yaml` will be expanded automatically.

### LLM Backends

- **Local model**: pass a local path to `--agent_name`, e.g. `/mnt/.../gemma-2-9b-it`. The model is loaded via vLLM internally.
- **Remote model**: any OpenRouter-served model, e.g. `--agent_name meta-llama/llama-3.3-70b-instruct`. Set the `OPENROUTER_API_KEY` environment variable beforehand.

---

## Quick Start

### Training (with Skill Evolution)

```powershell
python main.py `
  --MDP_type SMDP `
  --env_names Mastermind-v0 `
  --agent_name meta-llama/llama-3.3-70b-instruct `
  --ge_model_name meta-llama/llama-3.3-70b-instruct `
  --max_iters 50 `
  --episodes_per_iter 5 `
  --pool_size 10 `
  --select_type llm_model
```

Each outer iteration will:
1. Run `episodes_per_iter` episodes in every environment.
2. Update each skill's `frequency / avg_gain / maturity` statistics from the collected trajectories.
3. Invoke `SkillEvolution` for skill evolution and verification.
4. Log to SwanLab and write `outputs/*_training_logs.json`.

### Evaluation Only (load an existing Skill Pool)

```powershell
python main.py `
  --MDP_type SMDP `
  --test `
  --env_names Mastermind-v0 `
  --agent_name meta-llama/llama-3.3-70b-instruct `
  --load_pool_path ./outputs/xxx_training_logs.json `
  --load_iteration 1
```

### Ablation Studies

```powershell
# Disable Skill Generation
python main.py --ablation_type wo_sg ...

# Disable score-based maintenance, use FIFO instead
python main.py --ablation_type wo_score ...
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--MDP_type` | `SMDP` | `SMDP` enables skills; `MDP` is pure LLM decision-making |
| `--ablation_type` | `none` | One of `wo_sg / wo_ppo / wo_score / none` |
| `--env_names` | `Mastermind-v0` | Comma-separated list of task names |
| `--agent_name` | gemma/llama, etc. | Decision LLM (local path or remote ID) |
| `--ge_model_name` | llama-3.3-70b | LLM used for Skill Evolution |
| `--select_type` | `llm_model` | Skill selection strategy |
| `--pool_size` | `10` | Maximum size of the Skill Pool |
| `--max_iters` | `50` | Number of outer iterations |
| `--episodes_per_iter` | `5` | Episodes per environment per iteration |
| `--epsilon_initial` | `0.3` | Initial ε for skill selection |
| `--skill_select_k` | `1` | Number of skill-selection steps |
| `--topk` | `5` | Top-K used during retrieval |
| `--output_dir` | `outputs` | Directory for training logs |

See `parse_arguments` in `main.py` for the full list.

---

## Outputs and Logging

- **JSON training logs**: `outputs/<env>_<MDP>_<select>_<agent>_<ge_model>_<timestamp>_training_logs.json`, containing:
  - Per-iteration `avg_return`, `task_baselines`, and `global_baseline`
  - `pool_snapshot`: snapshot of the Skill Pool (including initiation / policy / termination / freq / gain / maturity)
  - `evolution_details` and `maintenance_details`: evolution and maintenance traces
  - `delta_prompt_tokens_per_step`: extra prompt tokens introduced by skill injection
- **SwanLab**: the experiment name is auto-composed from CLI arguments and tracks per-environment episode return, baseline, ε, and more.

---

## Design Overview

```
            ┌─────────────────────┐
            │     SkillPool       │
            │  (Initiation/Policy │
            │   /Termination)     │
            └──────────┬──────────┘
                       │ select_skill(state)
                       ▼
state ─►  build_decision_prompt() ─►  LLM Policy ─► action ─► Env
                       ▲                                       │
                       │ trajectory + reward                   │
                       │                                       ▼
                ExperiencePool / GoldenExperiencePool ◄────────┘
                       │
                       ▼
                 SkillEvolution
        (generate / refine / verify / score / prune skills)
```

- **Skill selection**: `SkillPool.select_skill` decides whether to activate a skill given the current state, ε, and `select_type`.
- **Skill termination**: `SkillMDP._should_terminate` uses a meta-controller LLM to decide whether the active skill should stop.
- **Skill maintenance**: `maintain` / `maintain_fifo` prune the pool by `freq × avg_gain` score or by FIFO.
- **Skill evolution**: `SkillEvolution.run_skill_evolution_with_verification` combines recent experiences with the Golden Experience Pool to generate and verify new skills.

---

## FAQ

- **vLLM fails to load?** Check `tensor_parallel_size` (default is 4); set it to 1 on a single-GPU machine and make sure VRAM is sufficient.
- **OpenRouter returns 401?** Set the environment variable: in PowerShell, `$env:OPENROUTER_API_KEY = "..."`.
- **Running non-ALFWorld tasks without ALFWorld installed?** The code falls back gracefully and skips ALFWorld initialization automatically.

---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{mi2026skillpro,
  title   = {Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents},
  author  = {Mi, Qirui and Ma, Zhijian and Yang, Mengyue and Li, Haoxuan and Wang, Yisen and Zhang, Haifeng and Wang, Jun},
  journal = {arXiv preprint arXiv:2602.01869},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.01869}
}
```

---

## License

This project is released under the [MIT License](./LICENSE). See the `LICENSE` file at the repository root for details.

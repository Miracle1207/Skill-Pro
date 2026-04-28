"""
Guidance-Enhanced MDP - Main Entry
----------------------------------
This script:
1. Parses command-line arguments;
2. Passes configuration to runner.run_guidance_mdp();
3. Executes the main loop including interaction and Skills evolution.
"""

import argparse
import logging

from run import SkillMDP

# ======================
# Argument Parser
# ======================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Skill-augmented MDP Main")

    # Basic setup --env_names "FrozenLake-v0"
    parser.add_argument("--MDP_type", type=str, default="SMDP", choices=["SMDP", "MDP"], help="Choose between full Skill-MDP (SMDP) or vanilla MDP")
    parser.add_argument("--ablation_type", type=str, default="none", choices=["wo_sg", "wo_ppo", "wo_score", "none"], help="Choose between full Skill-MDP (SMDP) or vanilla MDP")
    parser.add_argument("--env_names", type=str, default="Mastermind-v0", help="Comma separated env names")   # "Mastermind-v0"
    parser.add_argument("--agent_name", type=str, default="/mnt/QTJC/qirui/LM/google/gemma-2-9b-it")  # meta-llama/llama-3.3-70b-instruct,  /mnt/QTJC/qirui/LM/google/gemma-3-4b-it   "/mnt/QTJC/qirui/LM/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    parser.add_argument("--ge_model_name", type=str, default="meta-llama/llama-3.3-70b-instruct")
    parser.add_argument("--select_type", type=str, default="llm_model")  # llm_model, llm_topk_lcb
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_iters", type=int, default=50, help="Maximum number of outer iterations.")
    parser.add_argument("--episodes_per_iter", type=int, default=5, help="Number of episodes per iteration.")
    parser.add_argument("--test", action="store_true", help="若开启，则 run SkillMDP 会进行优化，否则仅运行")
  
    parser.add_argument("--load_pool_path", type=str, default="./outputs/Mastermind-v0_llm_model_llama-3.3-70b-instruct_llama-3.3-70b-instruct_20260107-025951_training_logs.json", help="加载本地 Pool 的路径")   # 第一次训练好的 skill pool
    parser.add_argument("--load_iteration", type=int, default=1, help="加载指定轮次的 Pool")

    parser.add_argument("--pool_size", type=int, default=10, help="Number of Skills to keep (k=10 for single Skill).")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of interaction trajectories.")
    parser.add_argument("--skill_select_k", type=int, default=1, help="Steps of selecting skills.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K retrieval during Skill selection.")
    parser.add_argument("--eta_len", type=float, default=0.01,  help="Length penalty weight.")
    parser.add_argument("--gamma_div", type=float, default=0.10,  help="Diversity weight.")
    parser.add_argument("--beta_compat", type=float, default=0.00,  help="Compatibility weight.")
    parser.add_argument("--acceptance_margin", type=float, default=0.000,  help="Acceptance margin.")

    parser.add_argument("--data_dir", type=str, default="./collected_data",
                        help="Directory to save/load interaction data.")
    parser.add_argument("--save_data", action="store_true", help="If True, save the collected ge_pairs to disk.")
    parser.add_argument("--load_data", action="store_true",
                        help="If True, load ge_pairs from disk instead of running environment interaction.")
    

    # Critic / training
    parser.add_argument("--epsilon_initial", type=float, default=0.3, help="Initial exploration probability.")
    parser.add_argument("--ge_batch_size", type=int, default=8,  help="Number of batch size for Skill evolution model.")
    parser.add_argument("--ge_num_epochs", type=int, default=3,  help="Number of training episodes for Skill evolution model.")
    parser.add_argument("--critic_updates_per_iter", type=int, default=1,  help="Number of critic updates per training phase.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu/cuda).")
    parser.add_argument("--lr_critic", type=float, default=1e-4,  help="learning rate of critics.")


    # Logging / output
    parser.add_argument("--output_dir", type=str, default="outputs",  help="Directory to save outputs.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging verbosity.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    smdp = SkillMDP(args)
    smdp.run_skill_mdp()
        

if __name__ == "__main__":
    main()

tune_ckpt_path='llama3_instruct_0.6'
prune_ckpt_path='llama3_instruct_0.6'

# export HF_HOME=/data/share
# export HF_ENDPOINT=https://hf-mirror.com

echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=$1 python3 ./LLMPruner/llama3.py --base_model "/data/share/Meta-Llama-3-8B-Instruct" --prune_pth $prune_ckpt_path --pruning_ratio 0.5 --device cuda  --eval_device cuda --block_wise --anchor_layers 27 30 2 31 1 0 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=$1 python3 ./LLMPruner/post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama3_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python3 generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"
prefill_slo=(0.8 0.6 0.4 0.2 0.2)
decode_slo=(0.9 0.8 0.7 0.6 0.5)

for ((i=0; i<${#prefill_slo[@]}; i++)); do
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama" --mode "LLMPruner" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama" --mode "Lingua2+Contextual" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama" --mode "LayerReduction" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama" --mode "LaCo" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama" --mode "ShortGPT" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama" --mode "AttnDrop" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama" --mode "Ours" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama_ARC_E.txt"
done

for ((i=0; i<${#prefill_slo[@]}; i++)); do
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3" --mode "LLMPruner" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3" --mode "Lingua2+Contextual" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3" --mode "LayerReduction" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3" --mode "LaCo" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3" --mode "ShortGPT" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3" --mode "AttnDrop" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3" --mode "Ours" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_ARC_E.txt"
done

for ((i=0; i<${#prefill_slo[@]}; i++)); do
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3-instruct" --mode "LLMPruner" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_instruct_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3-instruct" --mode "Lingua2+Contextual" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_instruct_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3-instruct" --mode "LayerReduction" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_instruct_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3-instruct" --mode "LaCo" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_instruct_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3-instruct" --mode "ShortGPT" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_instruct_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3-instruct" --mode "AttnDrop" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_instruct_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama3-instruct" --mode "Ours" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/llama3_instruct_ARC_E.txt"
done

for ((i=0; i<${#prefill_slo[@]}; i++)); do
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "vicuna" --mode "LLMPruner" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/vicuna_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "vicuna" --mode "Lingua2+Contextual" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/vicuna_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "vicuna" --mode "LayerReduction" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/vicuna_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "vicuna" --mode "LaCo" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/vicuna_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "vicuna" --mode "ShortGPT" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/vicuna_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "vicuna" --mode "AttnDrop" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/vicuna_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "vicuna" --mode "Ours" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/vicuna_ARC_E.txt"
done

for ((i=0; i<${#prefill_slo[@]}; i++)); do
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "orca3b-mini" --mode "LLMPruner" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/orcamini_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "orca3b-mini" --mode "Lingua2+Contextual" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/orcamini_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "orca3b-mini" --mode "LayerReduction" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/orcamini_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "orca3b-mini" --mode "LaCo" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/orcamini_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "orca3b-mini" --mode "ShortGPT" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/orcamini_ARC_E.txt"
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "orca3b-mini" --mode "AttnDrop" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/orcamini_ARC_E.txt"
    # CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "orca3b-mini" --mode "Ours" --prefill_SLO ${prefill_slo[$i]} --decode_SLO ${decode_slo[$i]} --res_save_pth "ELASTICLLM/scripts/res/orcamini_ARC_E.txt"
done

prune_ratios=("2.7b" "1.3b" "350m")
for ((i=0; i<${#prune_ratios[@]}; i++)); do
    CUDA_VISIBLE_DEVICES=$1 python3 ELASTICLLM/ARC_E.py --model "llama" --mode "Off-The-Shelf" --prune_ratio ${prune_ratios[$i]}  --res_save_pth "ELASTICLLM/scripts/res/ARC_E.txt"
done
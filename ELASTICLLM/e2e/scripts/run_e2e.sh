if [ "$2" -eq 1 ]; then
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama.txt"

fi

if [ "$2" -eq 2 ]; then
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama3" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama3" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama3" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama3" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama3" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama3" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama3" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama3" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama3" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama3" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama3" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama3" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama3" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama3" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama3" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3.txt"

fi

if [ "$2" -eq 3 ]; then
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "orca3b-mini" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "orca3b-mini" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "orca3b-mini" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "orca3b-mini" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "orca3b-mini" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "orca3b-mini" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "orca3b-mini" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "orca3b-mini" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "orca3b-mini" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "orca3b-mini" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "orca3b-mini" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "orca3b-mini" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "orca3b-mini" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "orca3b-mini" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "orca3b-mini" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_orcamini.txt"

fi

if [ "$2" -eq 4 ]; then
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama3-instruct" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama3-instruct" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama3-instruct" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama3-instruct" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama3-instruct" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama3-instruct" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama3-instruct" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama3-instruct" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama3-instruct" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama3-instruct" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "llama3-instruct" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "llama3-instruct" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "llama3-instruct" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "llama3-instruct" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "llama3-instruct" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_llama3_instruct.txt"
fi

if [ "$2" -eq 5 ]; then
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "vicuna" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "vicuna" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "vicuna" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "vicuna" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "vicuna" --alpha 0 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "vicuna" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "vicuna" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "vicuna" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "vicuna" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "vicuna" --alpha 0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"

CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Ours" --model "vicuna" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LayerReduction" --model "vicuna" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Lingua2+Contextual" --model "vicuna" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "LLMPruner" --model "vicuna" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"
CUDA_VISIBLE_DEVICES=$1 python ELASTICLLM/e2e/exps/run_e2e_acc.py --mode "Off-The-Shelf" --model "vicuna" --alpha -0.25 --res_save_pth "ELASTICLLM/e2e/scripts/res/res_vicuna.txt"

fi
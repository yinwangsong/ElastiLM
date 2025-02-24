Code for \[TBD'25\] Elastic On-Device LLM Service.

## Project structure

```
ElastiLM
|–– ElastiLM/
|   |–– ... # datasets for on-device training
|–– deployment/
|   |–– mllm # C++/Assembly code for on-device deployment
|–– proxynetworks/ # original NN, proxy networks, etc. 
|   |–– models/
|   |   |––jit/
|   |   |   |––...
|   |––pth/
|   |   |––...
|–– res/ # training configs and logs
|   |––train_log/ 
|–– scripts/
|   |––run_e2e.sh
|   |––...
|–– src/
|   |––training.py
|   |––...
```

## Running ElastiLM

### Hardware/software environment

We mainly perform the expriments on a cloud linux server with 8x 45GB A40 GPUs.
The software environment is
```
python=3.10.12
torch==2.3.1
datasets==2.19.1
numpy==1.26.4
tqdm==4.66.4
wandb==0.17.0
```
. For detailed information, we recommend you directly using the [docker image](), or checking the software inside it and installing them manually.

ElastiLM is built atop several third-party libs, which are modified significantly by us from their original source code. You need to install some of them by
```
cd transformers/
pip install -e .
cd ../scores/
pip install -e .
cd ../LLMLingua/
pip install -e .
```
. 

### Elasticizing & LoRA Recovery

First, profile the importance of permutation-consistent units and recover each submodel with LoRA.

```
# '02.sh' means identifying the top 20% important permutation consistent units.
# '0' means the GPU rank.
bash LLMPruner/scripts/02.sh 0
bash LLMPruner/scripts/03.sh 0
bash LLMPruner/scripts/04.sh 0
...
bash LLMPruner/scripts/05.sh 0
```

After profiling, the importance scores will be generated in `ELASTICLLM/imp/`.
The corresponding LoRAs will be generated in `ELASTICLLM/tune_log/`.

This procedure will last for over 100 hours on a single A40 GPU for all the models in our experiment. 
For the efficiency of multi-GPU users, we also provide fine-grained scripts in `LLMPruner/scripts/<model_name>_prune_tune/`.

### Calibrating the anchor layers

ElastiLM does not elasticize several the most important layers (named anchor layers).
To calibrate the anchor layers, run

```
python3 ELASTICLLM/Anchor_layers/llama_layers.py
...
python3 ELASTICLLM/Anchor_layers/orca_mini_3b_layers.py
```

. The output importance/layer will be like the following.

```
[8.74935245513916, 3.9551780223846436, 6.804546356201172, 3.6290643215179443, 3.6744985580444336, 3.544438600540161, 3.488487482070923, 3.517648696899414, 3.483860731124878, 3.4955999851226807, 3.50988507270813, 3.5368385314941406, 3.496831178665161, 3.496894359588623, 3.5130739212036133, 3.4906234741210938, 3.4990649223327637, 3.5532522201538086, 3.5287230014801025, 3.532752752304077, 3.516073703765869, 3.4922406673431396, 3.462526798248291, 3.5130510330200195, 3.501737356185913, 3.4650626182556152, 3.449836015701294, 3.5024378299713135, 3.5504703521728516, 3.4913651943206787, 3.7376880645751953, 4.114096164703369]
[26, 22, 25, 8, 6, 15, 29, 21, 9, 12, 13, 16, 24, 27, 10, 23, 14, 20, 7, 18, 19, 11, 5, 28, 17, 3, 4, 30, 1, 31, 2, 0]
```

### The tiny language model

ElastiLM trains a tiny (or you can say small) language model for prompt elastification and prompt-model orchestration.

## On-device deployment

We provide an on-device deployment demo of ElastiLM.

### Hardware/software environment

The demo is runnable on ARM platforms with SIMD Extention (i.e., NEON) and half precision (i.e., FP16) support.
Currently we demonstrate the elasity by `TTFT (Time-To-First Token)` and `TPOT (Time-Per-Output-Token)`measured by `std::chrono::system_clock::now()`.
You can also root your device to monitor the advanced information like energy.

In the following parts, we use a MI14 smartphone by default. The specification is listed below.

```
$ free -h                                                                                                                                                                                            
                total        used        free      shared     buffers
Mem:              15G        7.5G        7.3G        213M        5.3M
-/+ buffers/cache:           7.5G        7.3G
Swap:             14G        2.4G         12G

$ getprop | grep -E 'cpu|hardware'
[dalvik.vm.background-dex2oat-cpu-set]: [0,1,5,6]
[dalvik.vm.boot-dex2oat-cpu-set]: [0,1,5,6]
[dalvik.vm.default-dex2oat-cpu-set]: [0,1,2,3,4,5,6,7]
[ro.boot.hardware]: [qcom]
[ro.product.cpu.abilist64]: [arm64-v8a]
[ro.product.cpu.pagesize.max]: [4096]
```

We cross-compile the C++ deployment code on linux servers. The software versions are

```
cmake version 3.25.1
GNU Make 4.3
Android NDK r26c
```

. For detailed information, we recommend you directly using the [docker image](), or checking the software inside it and installing them manually.

### Compiling

```
cd deployment/mllm/scripts
export $ANDROID_NDK=/path/to/your/NDK
bash ./build_android.sh
```

The compiled binary file `demo_elastic_llama_lora` will be located in `deplyment/mllm/bin-arm/`.

### Model preparation

We pre-uploaded an elasticized [`oraca_mini_3b` model](https://huggingface.co/pankajmathur/orca_mini_3b) with the corresponding fine-tuned LoRA weights and the tiny language model on [Google Drive](https://drive.google.com/drive/folders/1RAKabZHfubIXmpzFMqDaGv7ki5SjXjSi?usp=sharing).

Please download them and put them in `deployment/mllm/models/` by

```
gdown <file-id>
```
. For other models, you can use the script `deployment/mllm/tools/convertor/converter.py`.

### Running

We recommend you using the `adb` tool to connect to the device and run the demo. You can attach your device to the host machine by either the USB or the WiFI.

After connection, run

```
cd deployment/mllm/scripts
./run_elastic_llama_lora.sh
```

. Replace the `adb -H host.docker.internal` to `adb` when you are not using docker.

If you have performed the aforementioned correctly, you will see 

```
[Q] India is in the Northern Hemisphere and Australia is in the Southern Hemisphere. In June, it is summer in India and winter in Australia. What is the main reason the seasons are opposite in the two countries?
prefill SLO (20%, 30%, ..., 100%): 
```

. Set up the SLOs for each request by screen input.

### Demo video

Here is a demo video that is run on device via [Termux](https://play.google.com/store/apps/details?id=com.termux&pli=1).


<div align="center">
<table>
    <tr>
        <td>
            <video src="https://github.com/user-attachments/assets/33ce728b-c26f-42cd-bf2c-808d1ec94e6a">
        </td>
    </tr>
    <tr>
        <td align="center">Demo on MI14</td>
    </tr>    
</table>
</div>


## Artifact evaluation

## Coming soon

- System service binding APIs.
- NPU support.
- ...

## Acknowledgement

[`HF Transformers`](https://github.com/huggingface/transformers) [`LLMPruner`](https://github.com/horseee/LLM-Pruner) [`mllm`](https://github.com/UbiquitousLearning/mllm)

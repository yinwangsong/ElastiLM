Code for \[TBD'25\] Elastic On-Device LLM Service.

## Project structure

```
ElastiLM
|–– ElastiLM/
|   |–– ... # datasets for on-device training
|–– deployment/
|   |–– ... # documents
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

## Running

## On-device deployment

We provide an on-device deployment demo of ElastiLM.

### Hardware/software environment

The demo is runnable on ARM plantforms with SIMD Extention (i.e., NEON) and half precision (i.e., FP16) support.
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

```bash
cd deployment/mllm/scripts
export $ANDROID_NDK=/path/to/your/NDK
bash ./build_android.sh
```

The compiled binary file `demo_elastic_llama_lora` will be located in `deplyment/mllm/bin-arm/`.

### Model preparation

We pre-uploaded an elasticized [`oraca_mini_3b` model]() with the corresponding fine-tuned LoRA weights and the tiny language model on [Google Drive]().

Please download them and put them in `deplyment/mllm/models/` by

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

```

### Demo video

## Artifact evaluation

## Coming soon

## Acknowledgement
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
Currently we demonstrate the elasity by TTFT (Time-To-First Token) and TPOT (Time-Per-Output-Token).
You can also root your device to monitor the advanced information like energy.

In the following parts, we use a 

### Compiling

```bash
cd mllm/scripts
export $ANDROID_NDK=/path/to/your/NDK
bash ./build_android.sh
```

### Model preparation

### Running

### Demo video

## Comming soon

## Acknowledgement
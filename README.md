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

We pre-uploaded an elasticized [`oraca_mini_3b` model](https://huggingface.co/pankajmathur/orca_mini_3b) with the corresponding fine-tuned LoRA weights and the tiny language model on [Google Drive](https://drive.google.com/drive/folders/1RAKabZHfubIXmpzFMqDaGv7ki5SjXjSi?usp=sharing).

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
[Q] India is in the Northern Hemisphere and Australia is in the Southern Hemisphere. In June, it is summer in India and winter in Australia. What is the main reason the seasons are opposite in the two countries?
prefill SLO (20%, 30%, ..., 100%): 
```

. Set up the SLOs for each request by screen input.

### Demo video

Here is a demo video that is run on device via [Termux](https://play.google.com/store/apps/details?id=com.termux&pli=1).


<!DOCTYPE html>
<html>
<head>
<style>
    table {
        margin: auto; /* 表格水平居中 */
        width: 50%;   /* 根据需要调整宽度 */
        border-collapse: collapse; /* 可选：合并边框 */
    }
    td {
        text-align: center; /* 文本和内容居中 */
        vertical-align: middle; /* 垂直居中 */
    }
    video {
        max-width: 100%; /* 视频宽度适应单元格大小 */
        height: auto; /* 保持视频比例 */
    }
</style>
</head>
<body>

<table>
    <tr>
        <td>  
            <video controls src="https://github.com/user-attachments/assets/33ce728b-c26f-42cd-bf2c-808d1ec94e6a">
                您的浏览器不支持 video 标签。
            </video>
        </td>
    </tr>
    <tr>
        <td>On MI14</td>
    </tr>
</table>

</body>
</html>


## Artifact evaluation

## Coming soon

## Acknowledgement

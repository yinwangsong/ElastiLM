#!/bin/bash

adb shell mkdir /data/local/tmp/mllm
adb shell mkdir /data/local/tmp/mllm/bin
adb shell mkdir /data/local/tmp/mllm/models
adb shell mkdir /data/local/tmp/mllm/vocab
adb push ../vocab/gte_vocab.mllm /data/local/tmp/mllm/vocab/
adb push ../bin-arm/demo_mobilebert /data/local/tmp/mllm/bin/
adb push ../models/mobilebert-uncased-fp32.mllm /data/local/tmp/mllm/models/
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb push failed"
    exit 1
fi
#adb shell "cd /data/local/tmp/mllm/bin && ./main_llama"
adb shell "cd /data/local/tmp/mllm/bin && ./demo_mobilebert"
#!/bin/bash

adb -H host.docker.internal shell mkdir /data/local/tmp/mllm
adb -H host.docker.internal shell mkdir /data/local/tmp/mllm/bin
adb -H host.docker.internal shell mkdir /data/local/tmp/mllm/models
adb -H host.docker.internal shell mkdir /data/local/tmp/mllm/vocab
adb -H host.docker.internal push ../vocab/gte_vocab.mllm /data/local/tmp/mllm/vocab/
adb -H host.docker.internal push ../bin-arm/demo_mobilebert /data/local/tmp/mllm/bin/
adb -H host.docker.internal push ../models/mobilebert-uncased-fp16.mllm /data/local/tmp/mllm/models/
# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb -H host.docker.internal push failed"
    exit 1
fi
#adb -H host.docker.internal shell "cd /data/local/tmp/mllm/bin && ./main_llama"
adb -H host.docker.internal shell "cd /data/local/tmp/mllm/bin && ./demo_mobilebert -m ../models/mobilebert-uncased-fp16.mllm -t 1"
#!/bin/bash

adb -H host.docker.internal shell mkdir /data/local/tmp/mllm
adb -H host.docker.internal shell mkdir /data/local/tmp/mllm/bin
adb -H host.docker.internal shell mkdir /data/local/tmp/mllm/models
adb -H host.docker.internal shell mkdir /data/local/tmp/mllm/vocab
adb -H host.docker.internal push ../vocab/orca_vocab.mllm /data/local/tmp/mllm/vocab/
#adb -H host.docker.internal push ../bin-arm/main_llama /data/local/tmp/mllm/bin/
adb -H host.docker.internal push ../bin-arm/demo_llama /data/local/tmp/mllm/bin/
# adb -H host.docker.internal push ../models/orca_mini_3b-fp16.mllm /data/local/tmp/mllm/models/

if ! adb -H host.docker.internal shell [ -f "/data/local/tmp/mllm/models/orca_mini_3b-fp16.mllm" ]; then
    adb -H host.docker.internal push ../models/orca_mini_3b-fp16.mllm /data/local/tmp/mllm/models/
else
    echo "orca_mini_3b-fp16 file already exists"
fi


# if push failed, exit
if [ $? -ne 0 ]; then
    echo "adb -H host.docker.internal push failed"
    exit 1
fi
#adb -H host.docker.internal shell "cd /data/local/tmp/mllm/bin && ./main_llama"
adb -H host.docker.internal shell "cd /data/local/tmp/mllm/bin && ./demo_llama -m ../models/orca_mini_3b-fp16.mllm -v ../vocab/orca_vocab.mllm"
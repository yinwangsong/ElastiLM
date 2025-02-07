import numpy as np

LAYER_NUM = 26
HIDDEN_DIM = 3200
FFN_HIDDEN = 8640
SLO_LEVELS = 4
SLOS = [0.5, 0.6, 0.7, 0.8]
RANK = 8

LORA_PTH = "../models/orca_mini_3b-fp16-lora/"

for i in range(0, SLO_LEVELS):
    for j in range(0, 7*2*LAYER_NUM):
        if j % 14 < 6:
            if j %2 == 0:
                lora_qkv = np.random.rand(1, 1, int(HIDDEN_DIM), RANK).astype(np.float32)
            if j %2 == 1:
                lora_qkv = np.random.rand(1, 1, RANK, int(HIDDEN_DIM*SLOS[i])).astype(np.float32)

            flat_tensor = lora_qkv.flatten()
            with open(LORA_PTH + 'lora_{}_{}.raw'.format(i, j), 'wb') as f:
                np.array(lora_qkv.shape, dtype=np.int32).tofile(f)
                f.write(flat_tensor.tobytes())

        if j % 14 < 8 and j % 14 >= 6:
            if j %2 == 0:
                lora_o = np.random.rand(1, 1, int(HIDDEN_DIM*SLOS[i]), RANK).astype(np.float32)
            if j %2 == 1:
                lora_o = np.random.rand(1, 1, RANK, int(HIDDEN_DIM)).astype(np.float32)

            flat_tensor = lora_o.flatten()
            with open(LORA_PTH + 'lora_{}_{}.raw'.format(i, j), 'wb') as f:
                np.array(lora_o.shape, dtype=np.int32).tofile(f)
                f.write(flat_tensor.tobytes())

        if j % 14 >= 8 and j % 14 < 12:
            if j %2 == 0:
                lora_gateup = np.random.rand(1, 1, int(HIDDEN_DIM), RANK).astype(np.float32)
            if j %2 == 1:
                lora_gateup = np.random.rand(1, 1, RANK, int(FFN_HIDDEN*SLOS[i])).astype(np.float32)

            flat_tensor = lora_gateup.flatten()
            with open(LORA_PTH + 'lora_{}_{}.raw'.format(i, j), 'wb') as f:
                np.array(lora_gateup.shape, dtype=np.int32).tofile(f)
                f.write(flat_tensor.tobytes())


        if j % 14 >= 12:
            if j %2 == 0:
                lora_down = np.random.rand(1, 1, int(FFN_HIDDEN*SLOS[i]), RANK).astype(np.float32)
            if j %2 == 1:
                lora_down = np.random.rand(1, 1, RANK, int(HIDDEN_DIM)).astype(np.float32)

            flat_tensor = lora_down.flatten()
            with open(LORA_PTH + 'lora_{}_{}.raw'.format(i, j), 'wb') as f:
                np.array(lora_down.shape, dtype=np.int32).tofile(f)
                f.write(flat_tensor.tobytes())

for i in range(0, SLO_LEVELS):
    for j in range(0, LAYER_NUM):
        lora_scales = np.random.rand(1, 1, 1, 7).astype(np.float32)

        flat_tensor = lora_scales.flatten()
        with open(LORA_PTH + 'lora_scales_{}_{}.raw'.format(i, j), 'wb') as f:
            np.array(lora_scales.shape, dtype=np.int32).tofile(f)
            f.write(flat_tensor.tobytes())
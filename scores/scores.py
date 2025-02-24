# The naming is really none sense here, but let it be.
class scores:
    features = {}

class sparse:
    SPARSE = False
    PRUNE = False
    LORA = False
    CONTEXTUAL_TRAINING = False
    CONTEXTUAL_INFERENCE = False
    imp_attn = []
    imp_mlp = []

class contextual:
    predictor = []
    inputs = []
    labels = []
    ratio = 0.0

class mask:
    head_mask = []
    mlp_mask = [] 

class Passer:
    prune_strategy_save_path = ""

class LoRAs:
    lora_a_q = []
    lora_b_q = []
    lora_scale_q = 2.0

    lora_a_k = []
    lora_b_k = []
    lora_scale_k = 2.0

    lora_a_v = []
    lora_b_v = []
    lora_scale_v = 2.0

    lora_a_o = []
    lora_b_o = []
    lora_scale_o = 2.0

    lora_a_gate = []
    lora_b_gate = []
    lora_scale_gate = 2.0

    lora_a_down = []
    lora_b_down = []
    lora_scale_down = 2.0

    lora_a_up = []
    lora_b_up = []
    lora_scale_up = 2.0

class LoRAs_all_in_Dict:
    lora_a_q = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_b_q = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_scale_q = 2.0

    lora_a_k = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_b_k = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_scale_k = 2.0

    lora_a_v = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_b_v = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_scale_v = 2.0

    lora_a_o = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_b_o = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_scale_o = 2.0

    lora_a_gate = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_b_gate = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_scale_gate = 2.0

    lora_a_down = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_b_down = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_scale_down = 2.0

    lora_a_up = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_b_up = {0.2:[], 0.3:[], 0.4:[], 0.5:[], 0.6:[], 0.7:[], 0.8:[], 0.9:[]}
    lora_scale_up = 2.0

class tester:
    outputs = None
    outputs_pruner = None
#include <iostream>
#include "cmdline.h"
#include "models/llama/modeling_elastic_llama_with_lora.hpp"
#include "models/llama/tokenization_llama.hpp"
#include "models/mobilebert/modeling_dual_head_TLM.hpp"
#include "models/mobilebert/tokenization_mobilebert.hpp"
#include "processor/PostProcess.hpp"
#include <memory/MemInspect.hpp>
#include <map>
#include <algorithm> // std::sort, std::iota

#include <fmt/core.h>

#include <cfloat>
#include <cstring>
#include <cmath>

inline int64_t time_ms(void) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return ms.count();
}

using namespace mllm;

float get_prob(int k, const float *A, int pos) {
    assert(pos < k);
    float max_value = -FLT_MIN;
    for (int i = 0; i < k; i++) {
        if (A[i] > max_value) {
            max_value = A[i];
        }
    }
    float sum = 0.f;
    for (int i = 0; i < k; i++) {
        sum += std::exp(A[i] - max_value);
    }
    return std::exp(A[pos] - max_value) / sum;
}


bool compareByValue(const std::pair<int, float>& a, const std::pair<int, float>& b) {
    return a.second > b.second;
}

std::vector<int> argsort(const std::vector<float>& vec) {
    std::vector<std::pair<int, float>> indexedVec(vec.size());
    
    for (size_t i = 0; i < vec.size(); ++i) {
        indexedVec[i] = std::make_pair(i, vec[i]);
    }
    
    std::sort(indexedVec.begin(), indexedVec.end(), compareByValue);
    
    std::vector<int> sortedIndices(vec.size());
    for (size_t i = 0; i < indexedVec.size(); ++i) {
        sortedIndices[i] = indexedVec[i].first;
    }
    
    return sortedIndices;
}


float percentStringToFloat(const std::string& percentStr) {
    if (percentStr.back() != '%') {
        throw std::invalid_argument("The string does not end with a '%' symbol.");
    }
    
    std::string numberPart = percentStr.substr(0, percentStr.size() - 1);
    
    float value;
    try {
        value = std::stof(numberPart);
    } catch (const std::exception& e) {
        throw std::invalid_argument("Conversion of the numeric part failed.");
    }
    
    return value / 100;
}

int main(int argc, char **argv) {


    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama2_vocab.mllm");
    cmdParser.add<string>("vocab_slm", 's', "specify mllm slm tokenizer model path", false, "../vocab/llama2_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-2-7b-chat-q4_0_4_4.mllm");
    cmdParser.add<string>("model_slm", 'k', "specify mllm slm model path", false, "../models/llama-2-7b-chat-q4_0_4_4.mllm");
    cmdParser.add<string>("lora", 'a', "specify mllm lora path", false, "../models/llama-2-7b-chat-q4_0_4_4.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add<float>("ratio", 'r', "elasticize ratio", false, 1.0);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string vocab_path_slm = cmdParser.get<string>("vocab_slm");
    string model_path_slm = cmdParser.get<string>("model_slm");
    string lora_path = cmdParser.get<string>("lora");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");
    float ratio = cmdParser.get<float>("ratio");

    std::vector<std::string> PREFILL_SLO = {
        "[02]",
        "[03]",
        "[04]",
        "[05]",
        "[06]",
        "[07]",
        "[08]",
        "[09]",
        // "[10]"
    };

    std::vector<std::string> DECODE_SLO = {
        "<05>",
        "<06>",
        "<07>",
        "<08>",
        "<09>",
        "<10>"
    };

    std::map<float, string> prefill_dict = {
        {0.2, "[02]"}, 
        {0.3, "[03]"}, 
        {0.4, "[04]"}, 
        {0.5, "[05]"}, 
        {0.6, "[06]"}, 
        {0.7, "[07]"}, 
        {0.8, "[08]"}, 
        {0.9, "[09]"},
        {1.0, "[10]"},
    };

    std::map<float, string> decode_dict = {
        {0.5, "<05>"}, 
        {0.6, "<06>"}, 
        {0.7, "<07>"}, 
        {0.8, "<08>"}, 
        {0.9, "<09>"},
        {1.0, "<10>"}
    };

    std::vector<float> prompt_ratios = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
    std::vector<float> model_ratios = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    MobileBertTokenizer tlm_tokenizer = MobileBertTokenizer(vocab_path_slm);
    tlm_tokenizer.add_special_tokens_into_vocab(PREFILL_SLO);
    tlm_tokenizer.add_special_tokens_into_vocab(DECODE_SLO);

    MobileBertConfig config_slm = MobileBertConfig();
    MobileBertModel tlm = MobileBertModel(config_slm);

    LLaMATokenizer tokenizer = LLaMATokenizer(vocab_path);

    LLaMAConfig config(tokens_limit, "3B", HFHUBROPE);
    Elastilm::submodel_attn_hidden_dims = {
        {3200, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 0 level
        {2800, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 1
        {2400, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 2
        {2000, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 3
        {1600, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 4
        {1200, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 5
        {800, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 6
        {400, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 7
        {400, (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 8
        // {(int)(config.head_size * 0.1) * (int)(config.hidden_dim/config.head_size), (int)(config.head_size * 1.0) * (int)(config.hidden_dim/config.head_size)}, // 9
    };

    // std::cout<<Elastilm::submodel_attn_hidden_dims

    Elastilm::submodel_mlp_hidden_dims = {
        {8640, (int)(config.ffn_hidden * 1.0)}, // 0 level
        {7560, (int)(config.ffn_hidden * 1.0)}, // 1
        {6480, (int)(config.ffn_hidden * 1.0)}, // 2
        {5400, (int)(config.ffn_hidden * 1.0)}, // 3
        {4320, (int)(config.ffn_hidden * 1.0)}, // 4
        {3240, (int)(config.ffn_hidden * 1.0)}, // 5
        {2160, (int)(config.ffn_hidden * 1.0)}, // 6
        {1080, (int)(config.ffn_hidden * 1.0)}, // 7
        {863, (int)(config.ffn_hidden * 1.0)}, // 8
        // {(int)(config.ffn_hidden * 0.1), (int)(config.ffn_hidden * 1.0)}, // 9
    };

    Elastilm::submodel_lora_scale = 2.0;

    Elastilm::anchor_layers = {13, 6, 1, 25, 2, 0};
    Elastilm::layers_order = {};
    for(int id = 0; id < config.block_num; id++) {
        if (Elastilm::anchor_layers.find(id) != Elastilm::anchor_layers.end()) {
            continue;
        } else {
            Elastilm::layers_order.push_back(id);
        }
    }
    Elastilm::layers_order.insert(Elastilm::layers_order.end(), Elastilm::anchor_layers.begin(), Elastilm::anchor_layers.end());

    auto model = ElasticLLaMAModelWithLoRA(config);
    model.load_from_multifiles({model_path, lora_path});
    tlm.load(model_path_slm);


    string SYS_PROMPT = "";
    SYS_PROMPT += "You are a smart assistant that helps human sloving problems. You help them by answering questions.\n";
    SYS_PROMPT += "Examples: \nQuestion: Which factor will most likely cause a person to develop a fever? Answer: a bacterial population in the bloodstream.";
    SYS_PROMPT += "\nQuestion: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship? Answer: food.";
    SYS_PROMPT += "\nQuestion: When a switch is used in an electrical circuit, the switch can Answer: stop and start the flow of current.";
    SYS_PROMPT += "\nQuestion: Which of the following is an example of an assistive device? Answer: contact lens.";
    SYS_PROMPT += "\nQuestion: Rocks are classified as igneous, metamorphic, or sedimentary according to Answer: how they formed.";


    // string QUERY = "\nQuestion: ";
    // QUERY += "Answer:";
    vector<string> LABELS = {"A", "B", "C", "D", "E", "F", "G", "H"};
    vector<string> LABELS_NUMBER = {"1", "2", "3", "4", "5", "6", "7", "8"};

    vector<string> in_strs = {
        "India is in the Northern Hemisphere and Australia is in the Southern Hemisphere. In June, it is summer in India and winter in Australia. What is the main reason the seasons are opposite in the two countries?",
        "A green plant absorbs light. A frog eats flies. These are both examples of how organisms",
        "Soccer players use their muscle systems to kick a ball into a goal. What organ system coordinates the muscles?",
        "Organisms contain DNA. What makes prokaryotic DNA different from eukaryotic DNA?",
    };

    vector<vector<string>> out_strs_choices = {
        {
            "because Earth is tilted on its axis",
            "because the Sun rotates on its axis",
            "because Earth revolves around the Sun",
            "because of the distance between the countries"
        },
        {
            "obtain energy", 
            "escape predators", 
            "produce offspring", 
            "excrete waste"
        },
        {
            "The nervous system",
            "The endocrine system",
            "The respiratory system",
            "The circulatory system"
        },
        {
            "the molecular shape", 
            "the types of bases", 
            "the sugar composition", 
            "the presence of phosphates"
        }
    };

    vector<string> out_strs_labels = {
        "A",
        "B"
    };

    for (int i = 0; i < in_strs.size(); ++i) {

        string in_str = in_strs[i];
        // auto input_tensor = tokenizer.tokenize(in_str);
        // std::cout << mllm::physical_memory_used_by_process()/1024 << "MB" <<std::endl;
        // std::cout << mllm::virtual_memory_used_by_process()/1024 << "MB" <<std::endl;
        string prefill_slo;
        string decode_slo;        
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "prefill SLO (20%, 30%, ..., 100%): ";
        std::getline(std::cin, prefill_slo);
        std::cout << "decode  SLO (50%, 60%, ..., 100%): ";
        std::getline(std::cin, decode_slo);
        std::cout << "\n[A] " << std::flush;

        string tlm_text = prefill_dict[percentStringToFloat(prefill_slo)] + " " + decode_dict[percentStringToFloat(decode_slo)] + "\n" + SYS_PROMPT + "\nQuestion: ";
        // std::cout<<tlm_text<<std::endl;
        vector<Tensor> tlm_input = tlm_tokenizer.tokenizes(tlm_text);
        // tlm_input[0].printDataTorchLike<float>();
        clock_t start = time_ms();
        vector<Tensor> tlm_output = tlm(tlm_input);
        clock_t end = time_ms();
        Tensor token_scores_tensor = tlm_output[0];
        Tensor prompt_ratio_tensor = tlm_output[1];
        Tensor model_ratio_tensor = tlm_output[2];

        
        // prompt_ratio_tensor.printDataTorchLike<float>();
        // model_ratio_tensor.printDataTorchLike<float>();
        // token_scores_tensor.printDataTorchLike<float>();

        // exit(-1);
        int prompt_ratio_idx = 0;
        vector<float> tmp = {};
        for (int d = 0; d < prompt_ratio_tensor.dimension(); d++) {
            tmp.push_back(prompt_ratio_tensor.dataAt<float>(0, 0, 0, d));
        }
        prompt_ratio_idx = std::distance(tmp.begin(), std::max_element(tmp.begin(), tmp.end()));


        int model_ratio_idx = 0;
        tmp = {};
        for (int d = 0; d < model_ratio_tensor.dimension(); d++) {
            tmp.push_back(model_ratio_tensor.dataAt<float>(0, 0, 0, d));
        }
        model_ratio_idx = std::distance(tmp.begin(), std::max_element(tmp.begin(), tmp.end()));

        float prompt_ratio = prompt_ratios[prompt_ratio_idx];
        float model_ratio = model_ratios[model_ratio_idx];

        // std::cout<<tmp.size()<<" "<<model_ratio_idx<<" "<<model_ratio<<std::endl;

        if (prompt_ratio * model_ratio > percentStringToFloat(prefill_slo) || model_ratio > percentStringToFloat(decode_slo)) {
            model_ratio = 1.0;
            for (float r : model_ratios) {
                if (prompt_ratio * r <= percentStringToFloat(prefill_slo) && r <= percentStringToFloat(decode_slo)) {
                    model_ratio = r;
                }
            }
        }

        if (percentStringToFloat(prefill_slo) == 1.0) {
            prompt_ratio = 1.0;
        }

        // std::cout<<model_ratio<<std::endl;
        float prompt_compress_ratio = prompt_ratio;

        int max_query_len = 0;
        for (int j = 0; j < out_strs_choices.size(); j++) {
            string NEW_QUERY = "\nQuestion: " + in_str + "Answer: " + out_strs_choices[i][j];
            Tensor query = tokenizer.tokenize(NEW_QUERY);
            if (query.sequence() > max_query_len) {
                max_query_len = query.sequence();
            }
        }
        Tensor sys_prompt_ids = tokenizer.tokenize(SYS_PROMPT);
        int sys_prompt_len = sys_prompt_ids.sequence();

        float sys_prompt_compress_ratio = ((sys_prompt_len + max_query_len) * prompt_compress_ratio - max_query_len) / sys_prompt_len;
        sys_prompt_compress_ratio = sys_prompt_compress_ratio > 0 ? sys_prompt_compress_ratio : 0;


        tmp = {};
        for (int s = 0; s < token_scores_tensor.sequence(); s++) {
            tmp.push_back(token_scores_tensor.dataAt<float>(0, 0, s, 1));
        }
        vector<int> pred_indices_ = argsort(tmp);
        vector<int> pred_indices(pred_indices_.begin(), std::next(pred_indices_.begin(), int(sys_prompt_compress_ratio*sys_prompt_len)+1));

        vector<token_id_t> sys_prompt_compressed = {};
        for (int idx = 0; idx < sys_prompt_len; idx++) {
            if (std::find(pred_indices.begin(), pred_indices.end(), idx) != pred_indices.end()) {
                sys_prompt_compressed.push_back((token_id_t)sys_prompt_ids.dataAt<float>(0, 0, idx, 0));
            }
        }

        string sys_prompt_compressed_text = tlm_tokenizer.detokenize(sys_prompt_compressed);
        string TEMPLATE = sys_prompt_compressed_text + "\nQuestion: ";

        // std::cout<<model_ratio<<" "<<prompt_ratio<<std::endl;

        Elastilm::LEVEL = 10 - int(model_ratio*10);

        // std::cout<<model_ratio<<std::endl;

        // std::cout<<TEMPLATE<<"\n";
        string prompt = TEMPLATE + in_str + "Answer: ";
        // vector<token_id_t> x = {1,  3820,  9630, 11285,  1246,   287, 18123,   291, 10277};
        // for (int i = 0;i < 512; i++){
        //     x.push_back(1);
        // }
        // std::cout<<TEMPLATE<<std::endl;
        // std::cout<<prompt<<std::endl;
        Tensor input_tensor = tokenizer.tokenize(prompt);
        // std::cout<<SYS_PROMPT+"\nQuestion"<<std::endl;
        // Tensor input_tensor = tokenizer.tokenize(prompt);
        int input_seq_len = input_tensor.sequence();

        // Tensor input_tensor = LLaMATokenizer::tokens2Input(x);
        // input_tensor.printData<float>();

        Elastilm::inner_rank_buffer = Tensor(Backend::global_backends[MLLM_CPU]);
        Elastilm::inner_rank_buffer.setName("lora_inner_rank_buffer");
        Elastilm::inner_rank_buffer.reshape(1, 1, input_tensor.sequence(), Elastilm::RANK);
        Elastilm::inner_rank_buffer.setDtype(MLLM_TYPE_F32);
        Elastilm::inner_rank_buffer.alloc();

        float ratio = float(10 - Elastilm::LEVEL) / 10.0f;
        vector<vector<int>> activate_dims = {};
        for (int layer_id = 0; layer_id < config.block_num; layer_id++) {
            if (Elastilm::anchor_layers.find(layer_id) == Elastilm::anchor_layers.end()) {
                // std::cout<<ratio<<"\n";
                activate_dims.push_back({(int)(Elastilm::submodel_attn_hidden_dims[Elastilm::LEVEL][0]/100), (int)(Elastilm::submodel_mlp_hidden_dims[Elastilm::LEVEL][0])});
            } else {
                activate_dims.push_back({(int)(config.head_size * 1.0), (int)(config.ffn_hidden * 1.0)});
            }
        }

        // for (int layer_id = 0; layer_id < config.block_num; layer_id++) {
        //     std::cout<<activate_dims[layer_id][0]<<" "<< activate_dims[layer_id][1]<<std::endl;
        // }

        // exit(-1);

        float pred = 0.0f;
        double prefill_time_spent = 0.0;
        prefill_time_spent += end - start;
        double decode_time_spent = 0.0;
        int decode_token_nums = 0;
        std::cout << "prefilling...";

        // do prefill
        start = time_ms();
        auto result = model({input_tensor}, activate_dims);
        end = time_ms();

        prefill_time_spent += (double)(end - start);
        // auto [out_string, out_token] = tokenizer.detokenize(result[0]);
        // std::cout<<out_string<<std::endl;
        // auto [not_end, output_string] = tokenizer.postprocess(out_string);
        // if (!not_end) { continue; }
        // printf("\033[%dG", (int) strlen("[A]  "));
        std::cout << "             ";
        printf("\033[%dG", (int) strlen("[A]  "));
        std::cout.flush();
        // std::cout << output_string << std::flush;
        // chatPostProcessing(out_token, input_tensor, {});

        // float first_decoded_token = input_tensor.dataAt<float>(0, 0, 0, 0);
        // Tensor input_tensor_raw = input_tensor;

        Tensor output_tensor_prefill(result[0].batch(), result[0].head(), 1, result[0].dimension(), Backend::global_backends[MLLM_CPU], true);
        output_tensor_prefill.setName("output_tensor_prefill");
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        output_tensor_prefill.setTtype(INPUT_TENSOR);
        // result[0].printDataTorchLike<float>();
        // std::cout<<output_tensor_prefill.ctype()<<"\n";
        memcpy(output_tensor_prefill.ptrAt<float>(0, 0, 0, 0), result[0].ptrAt<float>(0, 0, result[0].sequence()-1, 0), result[0].dimension()*sizeof(float));
        // for (int b=0; b < result[0].batch(); b++){
        //     for (int h=0; b < result[0].head(); h++){
        //         for (int s=0; s < 1; s++){
        //             for (int d=0; d < result[0].dimension(); d++){
        //                 output_tensor_prefill.setDataAt<float>(b, h, s, d, result[0].dataAt<float>(b, h, s, d));
        //             }
        //         }
        //     }
        // }

        Tensor input_tensor_decode(1, 1, 1, 1, Backend::global_backends[MLLM_CPU], true);
        input_tensor_decode.setName("input_tensor_decode");
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        input_tensor_decode.setTtype(INPUT_TENSOR);
        input_tensor_decode.setDataAt<float>(0, 0, 0, 0, 0);
        
        int best_choice = 0;
        double max_log_prob_sum = 0;

        // do decode
        std::cout << "decoding...";
        for (int j = 0; j < out_strs_choices[i].size(); j++) {
            Tensor outputs_ = tokenizer.tokenize(out_strs_choices[i][j]);
            double log_prob_sum = 0;
            
            // std::cout<<input_tensor.dataAt<float>(0, 0, 0, 0)<<std::endl;
            // input_tensor_raw.setDataAt<float>(0, 0, 0, 0, first_decoded_token);
            Elastilm::inner_rank_buffer = Tensor(Backend::global_backends[MLLM_CPU]);
            Elastilm::inner_rank_buffer.setName("lora_inner_rank_buffer");
            Elastilm::inner_rank_buffer.reshape(1, 1, 1, Elastilm::RANK);
            Elastilm::inner_rank_buffer.setDtype(MLLM_TYPE_F32);
            Elastilm::inner_rank_buffer.alloc();

            // output_tensor_prefill.printDataTorchLike<float>();
            // std::cout<<(int) outputs_.dataAt<float>(0, 0, 0, 0)<<"\n";

            float prob = get_prob(output_tensor_prefill.dimension(), output_tensor_prefill.ptrAt<float>(0, 0, output_tensor_prefill.sequence()-1, 0), (int) outputs_.dataAt<float>(0, 0, 0, 0));
            // std::cout<<log(prob)<<std::endl;
            log_prob_sum += (double) log(prob);

            int decode_steps = outputs_.sequence() - 1;
            for (int step = 0; step < decode_steps; step++) {
                // input_tensor.printDataTorchLike<float>();
                // input_tensor_raw.printData<float>();

                input_tensor_decode.setDataAt<float>(0, 0, 0, 0, outputs_.dataAt<float>({0, 0, step, 0}));
                // input_tensor_decode.printDataTorchLike<float>();
                clock_t start = time_ms();
                auto result_raw = model({input_tensor_decode}, activate_dims);
                clock_t end = time_ms();

                decode_time_spent += end - start;
                decode_token_nums += 1;
                // result_raw[0].printDataTorchLike<float>();
                // std::cout<<(int) outputs_.dataAt<float>(0, 0, step+1, 0)<<"\n";

                float prob = get_prob(result_raw[0].dimension(), result_raw[0].ptrAt<float>(0, 0, result_raw[0].sequence()-1, 0), (int) outputs_.dataAt<float>(0, 0, step+1, 0));
                // std::cout<<log(prob)<<" "<<(int) outputs_.dataAt<float>(0, 0, step+1, 0)<<std::endl;
                log_prob_sum += (double) log(prob);


                // result[0].printDataTorchLike<float>();

                // auto [out_string, out_token] = tokenizer.detokenize(result[0]);
                // auto [not_end, output_string] = tokenizer.postprocess(out_string);
                // // std::cout<<"??"<<not_end<<std::endl;
                // if (!not_end) { break; }

                // std::cout << output_string << std::flush;
                // chatPostProcessing(out_token, input_tensor_raw, {});

            }
            // std::cout<<"==========\n";
            model.revert_kvcache_to_pos(input_seq_len);

            if (log_prob_sum > max_log_prob_sum) {
                max_log_prob_sum = log_prob_sum;
                best_choice = j;
            }
        }

        // printf("\033[%dG", (int) strlen("[A]  "));
        std::cout << "           ";
        printf("\033[%dG", (int) strlen("[A]  "));
        std::cout.flush();
        std::cout<<out_strs_choices[i][best_choice];

        std::cout<<"\nTTFT: "<<prefill_time_spent<<"ms, TPOT: "<<decode_time_spent/decode_token_nums<<"ms/token";
        printf("\n\n");
        model.clear_kvcache();
    }

    return 0;
}
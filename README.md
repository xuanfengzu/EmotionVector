# Emotion Vectors
The code and data for the paper "Controllable Emotion Generation with Emotion Vectors".

arxiv: https://arxiv.org/abs/2502.04075

## Datasets
| Dataset       | #Samples | #Contain Emotions | #Contain daily scenarios |
| ------------- | -------- | ----------------- | ------------------------ |
| EmotionQuery  | 500      | ✅                 | ❎                        |
| EmotionQuery+ | 400      | ✅                 | ✅                        |

## Emotion Vectors
For the sake of the file size, we only provide the EVs extracted from the following models:
| Model                      | #Num Layers |
| -------------------------- | ----------- |
| Meta-Llama-3.1-8B-Instruct | 32          |
| Qwen2.5-7B-Instruct        | 28          |
| MiniCPM3-4B                | 62          |

## Results on EmotionQuery+ dataset
Model results on EmotionQuery+ datasets after applying $ \text{EV}^{base} $ are listed in `/results/base`, and for each $ \text{EV}^{emotion} $ condition, results are listed in `/results/emotion`.


## How to use EmotionVector
The sample code of how to use the emotion vectors extracted is provided in `/code` folder. Here, we provide the sample code for applying vector to the model `Meta-Llama-3.1-8B-Instruct`. If you want to use other models, you can refer to the code and modify the code according to your needs.

### Usage
First, configure the environment for the corresponding model. 

Then, run:
```bash
cd code
python apply.py --model_path model_name_or_path --times 1 0 0 0 0 --vector_path path_to_emotion_vector
```

`--times` arg means the weight of each emotion vector, and the order of the weight is `anger, disgust, fear, joy, sadness`. For example, `--times 1 0 0 0 0` means applying the emotion vector of `anger` to the model, and the other emotions are not applied. You can configure any combination of emotions you want freely, even "0.5 * anger + 0.5 * joy"😆.

> A very "emotional" example on `Meta-Llama-3.1-8B-Instruct` is to apply 1*anger and then ask it "who are you", just try it!🥳

### Apply to other models
if you want to apply emotion vectors to other models, you can refer to the code and modify the code according to your needs.

Refer to `code/apply.py`, in `LlamaDetector.get_parameter_activation()`, change the following name:
```python
if "LlamaSdpaAttention" in str(module) and "LlamaDecoderLayer" not in str(
    module
):
    if "model.layers" in name:
```
into the name of the model you want to apply emotion vectors to.

You can use `print(model)` to decide the name you have to use. Take `Meta-Llama-3.1-8B-Instruct` as an example, `print(model)` will output:
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
```

Remember to change the num of layers in `LlamaDetector.get_parameter_activation()` and the way to generate results in `LlamaDetector.model_chat()`.


## Citation
If you find this project useful in your research, please cite:
```
@misc{dong2025controllableemotiongenerationemotion,
      title={Controllable Emotion Generation with Emotion Vectors}, 
      author={Yurui Dong and Luozhijie Jin and Yao Yang and Bingjie Lu and Jiaxi Yang and Zhi Liu},
      year={2025},
      eprint={2502.04075},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.04075}, 
}
```



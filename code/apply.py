import json
import os
import math
import argparse

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import logging

from log_config import configure_logger, set_log_level
from api import BaseDetector


logging.set_verbosity_error()

# Configure logging
logger = configure_logger(__name__)
set_log_level("info")

ORDER = ["anger", "disgust", "fear", "joy", "sadness"]


class LlamaDetector(BaseDetector):
    def __init__(
        self,
        model,
        tokenizer,
        times: list,
        get_activation: bool = True,
        clamp_min=float("-inf"),
        clamp_max=float("inf"),
    ):
        logger.info("Initializing the ActivationProcessDetector.")
        self.times = times
        super().__init__(model, tokenizer, get_activation, clamp_min, clamp_max)

    def _register_emotion_vector(self, vector):
        if isinstance(vector, str) and os.path.exists(vector):
            print("vector is a file path")
            all_data = {}
            for filename in os.listdir(vector):
                filepath = os.path.join(vector, filename)
                if filename.endswith(".json") and os.path.isfile(filepath):
                    with open(filepath, "r") as json_file:
                        try:
                            file_data = json.load(json_file)
                            key = os.path.splitext(filename)[0]
                            all_data[key] = file_data
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON from file: {filename}")
            self.emotion_vector = all_data
        else:
            print("Invalid vector type")

    def get_parameter_activation(self, name):
        def hook(module, input, output):
            emotion_vector = self.emotion_vector
            if "LlamaSdpaAttention" in str(module) and "LlamaDecoderLayer" not in str(
                module
            ):
                if "model.layers" in name:
                    emotion_vectors = [
                        torch.Tensor(
                            emotion_vector[f"Llama_{emotion}"][str(self.idx)]
                        ).to(output[0].device)
                        for emotion in ORDER
                    ]
                    for idx, vec in enumerate(emotion_vectors):
                        output[0][:, -1:, :] = (
                            output[0][:, -1:, :] + self.times[idx] * vec
                        )

                    self.idx += 1
                    if self.idx >= 32:
                        self.idx = 0
                    return output
            return output

        return hook

    def __register_hooks(self):
        logger.debug("Registering the hook")
        # print("hhhhhhhhh")
        self.idx = 0
        self.disturb = 0
        self.hooks = []
        for name, module in self.model.named_modules():
            # module = self.model.get_submodule(name.rsplit('.', 1)[0])
            self.hooks.append(
                module.register_forward_hook(self.get_parameter_activation(name))
            )

    def __remove_hooks(self):
        logger.debug("Removing the hook")
        self.idx = 0
        self.disturb = 0
        for hook in self.hooks:
            hook.remove()

    def model_chat(self, text, history=None):
        if self.get_activation:
            self.__register_hooks()

        logger.debug("Generating the answer")
        messages = [
            {"role": "user", "content": text},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1] :]
        self.output = self.tokenizer.decode(response, skip_special_tokens=True)

        if self.get_activation:
            self.__remove_hooks()

        return self.output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Model name or path to model. Model used here is Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--times",
        type=float,
        nargs=5,
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        help='Times to apply the emotion vector. Default order for emotions are ["anger", "disgust", "fear", "joy", "sadness"].',
    )
    parser.add_argument("--vector_path", type=str, help="Path to emotion vectors")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    detector = LlamaDetector(model, tokenizer, args.times)

    detector._register_emotion_vector(args.vector_path)

    while True:
        text = input("Enter text: ")
        if text == "exit":
            break
        print(detector.model_chat(text))

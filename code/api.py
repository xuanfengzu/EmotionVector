import json

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from log_config import configure_logger

logger = configure_logger(__name__)


def load_model(path):
    logger.info(f"Loading model from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", trust_remote_code=True
    ).eval()
    return tokenizer, model


class BaseDetector:
    def __init__(
        self,
        model,
        tokenizer,
        get_activation=True,
        clamp_min=float("-inf"),
        clamp_max=float("inf"),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.get_activation = get_activation
        self._parameter_activations_heatmap = {}
        self._parameter_activations_save = {}
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        logger.info(
            f"Initialing the dector with the get_activation={self.get_activation}"
        )
        if self.get_activation:
            self.__parameter_activation = {}

    def clear_activation(self):
        logger.debug("Clearing the activation map")
        self._parameter_activations_heatmap = {}

    def get_parameter_activation(self, name):
        raise NotImplementedError

    def __register_hooks(self):
        logger.debug("Registering the hook")
        self.hooks = []
        for name, module in self.model.named_modules():
            # module = self.model.get_submodule(name.rsplit('.', 1)[0])
            self.hooks.append(
                module.register_forward_hook(self.get_parameter_activation(name))
            )

    def __remove_hooks(self):
        logger.debug("Removing the hook")
        for hook in self.hooks:
            hook.remove()

    def model_chat(self, text, history=None):
        if self.get_activation:
            self.__register_hooks()

        logger.debug("Generating the answer")
        self.output = self.model.chat(self.tokenizer, text, history=history)

        if self.get_activation:
            self.__remove_hooks()

        return self.output[0]

    def __pop_last(self):
        logger.debug("Poping the last layer's params")
        self.__parameter_activation.pop(list(self.__parameter_activation.keys())[-1])

    def save_activation(self, save_path, pop_last=False):
        result = {"output": self.output}

        for name, activate_list in tqdm(self._parameter_activations_heatmap.items()):
            if self.get_activation:
                self.__parameter_activation[name] = torch.stack(activate_list).mean(
                    dim=0
                )

            result[name] = torch.stack(activate_list).mean(dim=0).tolist()

        if self.get_activation:
            self.__pop_last()

        logger.debug(f"Saving the activation to {save_path}")
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=4)

    def get_activation_map(self):
        if not self.get_activation:
            raise ValueError(
                "You need to set get_activation=True to get the activation map."
            )

        return self.__parameter_activation

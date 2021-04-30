import pprint
import json

from .transformer import Transformer
from .bert import BERT
from .roberta import RoBERTa
from .albert import ALBERT
from .albert import UnsharedALBERT
from .nezha import NEZHA
from .gpt import GPT
from .gpt2 import GPT2
from .gpt2ml import GPT2ML
from .mask import LMMaskMixIn
from .mask import UniLMMaskMixIn
from .mask import install_attention_mask
from .base import ModelBuilder
from .base import CheckpointLoader
from .training import AdversarialTraining
from .training import GradientPenalty

# model + mask
# 如：bert+lm, bert+encoder
transformers = {
    "bert": BERT,
    "roberta": RoBERTa,
    "albert": ALBERT,
    "unshared-albert": UnsharedALBERT,
    "nezha": NEZHA,
    "gpt": GPT,
    "gpt2": GPT2,
    "gpt2ml": GPT2ML
}

def load_transformer_configs(config_path, **kwargs):
    configs = {}
    with open(config_path, "r", encoding="utf-8") as fp:
        configs.update(json.load(fp))

    for key, value in kwargs.items():
        configs[key] = value

    # 添加一些默认参数或别名
    if "config_path" not in configs:
        configs["config_path"] = config_path
    if "segment_size" not in configs:
        configs["segment_size"] = configs.get("type_vocab_size", 2)
    if "max_position" not in configs:
        configs["max_position"] = configs.get("max_position_embeddings", 512)
    if "dropout_rate" not in configs:
        configs["dropout_rate"] = configs.get("hidden_dropout_prob")
    if "attention_dropout_rate" not in configs:
        configs["attention_dropout_rate"] = configs.get("attention_probs_dropout_prob")
    return configs

def save_configs(config_path, configs, **kwargs):
    for key, value in kwargs.items():
        configs[key] = value

    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(configs, fp, sort_keys=True)

def build_transformer(
    model="bert+encoder",
    config_path=None,
    checkpoint_path=None,
    verbose=True,
    **kwargs):
    configs = load_transformer_configs(config_path, **kwargs)
    if "+" in model:
        name, app = model.lower().split("+")
    else:
        name, app = model, None

    transformer = transformers[name]
    if app in ("lm", "unilm"):
        mask = app + "-" + "mask"
        transformer = install_attention_mask(transformer, mask)

    model = transformer(**configs).build()
    if verbose:
        model.model.summary(line_length=160)
        pprint.pprint(configs)
        model.show_inputs_outputs()

    if checkpoint_path is not None:
        model.load_checkpoint(checkpoint_path, verbose)
    return model.model

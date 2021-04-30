import itertools
from tf2bert.models import build_transformer
from tf2bert.models import transformers
from tf2bert.models import list_transformers

print(list_transformers())

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
for model, app in itertools.product(transformers.keys(), ["encoder", "lm", "unilm"]):
    if model.startswith("gpt") and app in ("lm", "unilm"):
        continue
    model = model + "+" + app
    print("="*20 + model + "="*20)
    model = build_transformer(
        model=model, 
        config_path=config_path, 
        checkpoint_path=None,
        with_mlm=False,
        verbose=True
    )

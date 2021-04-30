from tf2bert.models import build_transformer

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"

model = build_transformer(
    model="bert+encoder", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    with_mlm=True,
    verbose=True
)

from tf2bert.models import build_transformer

# 测试mask是否完整往下传递

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"

model = build_transformer(
    model="bert", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    verbose=False
)

for layer in model.layers:
    try:
        print(layer.name, layer.output._keras_mask)
    except:
        print("ERR:", layer.name)

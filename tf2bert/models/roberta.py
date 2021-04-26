from .bert import BERT

class RoBERTa(BERT):
    """Facebook的RoBERTa并没有修改BERT模型架构，而是进行更充分的探索，
    包括对BERT的模型架构、训练目标等细节上进行充分的实验。不同之处：
    - 更充分的训练：更大batch_size、更丰富数据、更长预训练时间
    - 移除NSP任务
    - 使用动态的mask
    - 支持更长的序列
    - 使用Byte Pair Encoding
    """

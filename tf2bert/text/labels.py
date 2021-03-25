# BIOES、BIO、BMES
class TaggingTransformer:
    """标签映射，标签的转换和逆转换"""

    def fit(self, batch_tags):
        self.labels = set(itertools.chain(*batch_tags))
        self.id2label = {i:j for i,j in enumerate(self.labels)}
        self.label2id = {j:i for i,j in self.id2label.items()}

    def transform(self, batch_tags):
        batch_ids = []
        for tags in batch_tags:
            ids = []
            for tag in tags:
                ids.append(self.label2id[tag])
            batch_ids.append(ids)
        return batch_ids

    def inverse_transform(self, batch_ids):
        batch_tags = []
        for ids in batch_ids:
            tags = []
            for i in ids:
                tags.append(self.id2label[i])
            batch_tags.append(tags)
        return batch_tags

    @property
    def num_classes(self):
        return len(self.labels)

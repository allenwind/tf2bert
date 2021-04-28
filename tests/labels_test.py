from tf2bert.text.labels import find_entities_chunking
from tf2bert.text.labels import find_entities
from tf2bert.text.labels import bio2iobes
from tf2bert.text.labels import batch_bio2iobes
from tf2bert.text.labels import batch_iobes2bio
from tf2bert.text.labels import TaggingTokenizer
from tf2bert.text.utils import load_ner_sentences

# 测试完整性

texts, labels = load_ner_sentences()
for text, label in zip(texts, labels):
    a_entities = find_entities(text, label)
    b_entities = []
    for label, i, j in find_entities_chunking(label):
        b_entities.append((text[i:j], label))

    assert a_entities == b_entities
    print(a_entities)
    print(b_entities)

texts, labels = load_ner_sentences()
for text, label in zip(texts, labels):
    label = bio2iobes(label)
    a_entities = find_entities(text, label)
    b_entities = []
    for label, i, j in find_entities_chunking(label):
        b_entities.append((text[i:j], label))

    assert a_entities == b_entities
    print(a_entities)
    print(b_entities)

iobes_labels = batch_bio2iobes(labels)
bio_labels = batch_iobes2bio(iobes_labels)

for label1, label2 in zip(bio_labels, labels):
    assert label1 == label2

tagger = TaggingTokenizer()
tagger.fit(labels)

ids = tagger.batch_encode(labels)
rlabels = tagger.batch_decode(ids)

for label1, label2 in zip(labels, rlabels):
    assert label1, label2

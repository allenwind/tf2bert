def words2bmes(words):
    tags = []
    for w in words:
        if not w:
            raise ValueError('{} contains None or zero-length word {}'.format(str(words), w))
        if len(w) == 1:
            tags.append('S')
        else:
            tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
    return tags


def bmes2words(chars, tags):
    result = []
    if len(chars) == 0:
        return result
    word = chars[0]

    for c, t in zip(chars[1:], tags[1:]):
        if t == 'B' or t == 'S':
            result.append(word)
            word = ''
        word += c
    if len(word) != 0:
        result.append(word)

    return result

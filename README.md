# Chinese text auto-completion

Word Hammer is an implementation of N-Gram model for auto completion.

Usage

```
from hammer import WordHammer

text = "今天天气真好"
hm = WordHammer()

# train
hm.loadCorpus(corpus=text)

# prediction
hm.doPrediction(sen="今天天气")

# save model
hm.save()

# load model
hm.load()

```
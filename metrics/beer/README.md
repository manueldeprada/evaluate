---
title: BEER
emoji: 🤗 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
   BEER 2.0 (BEtter Evaluation as Ranking) is a trained machine translation evaluation metric with high correlation with human judgment both on sentence and corpus level. It is a linear model-based metric for sentence-level evaluation in machine translation (MT) that combines 33 relatively dense features, including character n-grams and reordering features.
It employs a learning-to-rank framework to differentiate between function and non-function words and weighs each word type according to its importance for evaluation.
The model is trained on ranking similar translations using a vector of feature values for each system output.
BEER outperforms the strong baseline metric METEOR in five out of eight language pairs, showing that less sparse features at the sentence level can lead to state-of-the-art results.
Features on character n-grams are crucial, and higher-order character n-grams are less prone to sparse counts than word n-grams.
---

# Metric Card for BEER

## Metric description

BEER 2.0 (BEtter Evaluation as Ranking) is a trained machine translation evaluation metric with high correlation with human judgment both on sentence and corpus level. It is a linear model-based metric for sentence-level evaluation in machine translation (MT) that combines 33 relatively dense features, including character n-grams and reordering features.
It employs a learning-to-rank framework to differentiate between function and non-function words and weighs each word type according to its importance for evaluation.
The model is trained on ranking similar translations using a vector of feature values for each system output.
BEER outperforms the strong baseline metric METEOR in five out of eight language pairs, showing that less sparse features at the sentence level can lead to state-of-the-art results.
Features on character n-grams are crucial, and higher-order character n-grams are less prone to sparse counts than word n-grams.

## How to use 

BEER has two mandatory arguments:

`predictions`: a `list` of predictions to score. Each prediction should be a string with tokens separated by spaces.

`references`: a `list` of references (multiple `references` per `prediction` are not allowed). Each reference should be a string with tokens separated by spaces.

## Prerequisites
This module downloads and executes the original authors' BEER package. You must have Java installed to run it, and it will fail to load otherwise. 

Since it is not Python code and calls the BEER executable, it is much faster to pass a batch of predicitions and references to evaluate in a single call than to iteratively call the metric with one prediction and reference at a time.

```python
>>> meteor = evaluate.load('beer')
>>> predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
>>> references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
>>> results = meteor.compute(predictions=predictions, references=references)
```

## Output values

The metric outputs a dictionary containing the METEOR score. Its values range from 0 to 1, e.g.:
```
{'meteor': 0.9999142661179699}
```



## Citation

```bibtex
@inproceedings{stanojevic-simaan-2014-fitting,
    title = "Fitting Sentence Level Translation Evaluation with Many Dense Features",
    author = "Stanojevi{\'c}, Milo{\v{s}}  and
      Sima{'}an, Khalil",
    booktitle = "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})",
    month = oct,
    year = "2014",
    address = "Doha, Qatar",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D14-1025",
    doi = "10.3115/v1/D14-1025",
    pages = "202--206",
}
```
    
## Further References 
- [BEER -- Official GitHub](https://github.com/stanojevic/beer)


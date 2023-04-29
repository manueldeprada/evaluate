# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BEER metric. """
import os
import re
import datasets
import evaluate
import subprocess
import tempfile

_CITATION = """\
@inproceedings{banarjee2005,
  title     = {Fitting Sentence Level Translation Evaluation with Many Dense Features},
  author    = {Stanojevi{\'c}, Milo{\v{s}}  and Sima{'}an, Khalil},
  booktitle = "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})",
  month = oct,
  year = "2014",
  address = "Doha, Qatar",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/D14-1025",
  doi = "10.3115/v1/D14-1025",
  pages = "202--206",
}
"""

_DESCRIPTION = """\
BEER is a linear model-based metric for sentence-level evaluation in machine translation (MT) that combines 33 relatively dense features, including character n-grams and reordering features.

It employs a learning-to-rank framework to differentiate between function and non-function words and weighs each word type according to its importance for evaluation.

The model is trained on ranking similar translations using a vector of feature values for each system output.

BEER outperforms the strong baseline metric METEOR in five out of eight language pairs, showing that less sparse features at the sentence level can lead to state-of-the-art results.

Features on character n-grams are crucial, and higher-order character n-grams are less prone to sparse counts than word n-grams.
"""

_KWARGS_DESCRIPTION = """
Computes BEER score of translated segments against one or more references.
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    'beer': beer score.
    'scores': list of scores for each sentence.
Examples:

    >>> beer = evaluate.load('beer')
    >>> predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    >>> references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
    >>> results = beer.compute(predictions=predictions, references=references)
    >>> print(round(results["beer"], 4))
    0.3190
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Beer(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/stanojevic/beer"],
            reference_urls=[
                "http://aclweb.org/anthology/D14-1025",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        try:
            subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        except Exception as e:
            raise Exception("Java is not installed. Please install java and try again.")
        dl_manager = datasets.download.DownloadManager()
        _BEER_URL = "https://raw.githubusercontent.com/stanojevic/beer/master/packaged/beer_2.0.tar.gz"
        paths = dl_manager.download_and_extract(_BEER_URL)
        self.beer_path = os.path.join(paths, "beer_2.0/beer")
        self.float_pattern = re.compile(r"\d+\.\d+")

    def _compute(self, predictions, references):
        if isinstance(references[0], list):
            raise ValueError("Beer metric does not support multiple references")
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as pred_file:
                pred_file.write("\n".join(predictions))
                pred_file.flush()
                pred_file.close()
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as ref_file:
                    ref_file.write("\n".join(references))
                    ref_file.flush()
                    ref_file.close()
                    cmd = [self.beer_path, "-r", ref_file.name, "-s",pred_file.name, "--printSentScores"]
                    output = subprocess.check_output(cmd).decode("utf-8")
                    assert output.startswith("sent 1 score is "), "Unexpected output: {}".format(output)
                    output = output.strip().split("\n")
                    total_score = float(output[-1][11:])
                    scores = [float(self.float_pattern.findall(s)[0]) for s in output[:-1]]
                    return {"beer": total_score, "beer_scores": scores}
        except Exception as e:
            raise Exception("Error while computing beer score: {}".format(e))

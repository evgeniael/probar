This Github Repository contains all code and generations to reproduce results found in the paper:

[Variability Need Not Imply Error: The Case of Adequate but Semantically Distinct Responses]([https://arxiv.org/abs/2402.17527](https://arxiv.org/pdf/2412.15683))

The raw data used, [Provo Corpus](https://link.springer.com/article/10.3758/s13428-017-0908-4), [AbgCOQA](https://openreview.net/pdf?id=SlDZ1o8FsJU) and [AmbigQA](https://aclanthology.org/2020.emnlp-main.466.pdf) can be found in the `raw_datasets` folder, and their processed counterparts which we used in `processed_datasets`.

The scripts to generate samples from models (`get_generations.py`), obtain Probar (`proxy_probar.py`), other baselines (P(Adequate), a version of P(True) in `prob_adequate.py` and Entropy and Semantic entropy in `get_clusters_entail.py` and `get_entropies.py`; the former clustering generations semantically using NLI and the latter computing them) and the correctness judgements for a model decoding (`get_rougeL.py` and `greedy_correctness_llm_as_a_judge.py` for different criteria) can be found in the `scripts` folder, separately for the `NWP_task` and `QA_tasks`.

The resulting generations along with the various computed metrics can be found in the `generations` folder for all datasets, which can be used to exactly reproduce the results from the paper. For the subset of samples for which we obtained manual annotations, the relevant generations, uncertainty metrics and annotations can be found in the `manual_annotations` folder.

import itertools
import _pickle
import json
import random

import requests
from io import BytesIO
import zipfile

from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import BartTokenizer, BartModel

random.seed(20221219)

model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
num_cluster = 336
num_layer = 5
data_source = "SemEval2017-task10.csv"


def create_dataframe(filename):
  """
  :param filename:
  :return: pandas dataframe with separated columns for variables cluster id, preferred name, and term
  """
  with zf.open(filename) as file:
    data = pd.read_csv(file, sep=";", encoding="utf-8")
  ids = list()
  syns = list()
  pns = list()
  for item in data["0"]:
    syn_id, syn = item.split("||")
    if "|pn" in syn_id:
      pref_name = True
    else:
      pref_name = False
    syn_id = syn_id.split("|")[0]
    ids.append(syn_id)
    syns.append(syn)
    pns.append(pref_name)
  data["id"] = ids
  data["term"] = syns
  data["pref_name"] = pns
  return data


def calc_embed(data):
  new_data = pd.DataFrame(columns=["id", "term", "pref-name", "embed-layer", "embedding"])
  for term, syn_id, pref_name in zip(data["term"], data["id"], data["pref_name"]):
    embeds = []
    k_layers = []
    # embed = model.encode(term)
    # embeds.append(embed)
    token = tokenizer(term, return_tensors="pt")
    model.eval()
    with torch.no_grad():
      outputs = model(**token, output_hidden_states=True)
      layers = outputs.hidden_states[1:]
      for k in range(0, len(layers)):
        embed = layers[k].numpy().squeeze()
        n = list(embed.shape)[0]
        embed = embed[1:n-1]    # remove [CLS] and [SEP] tokens from calculation
        embed = np.mean(embed, axis=0).flatten()    # mean embedding over complete term
        k_layers.append(k+1)
        embeds.append(embed)
    sub_data = pd.DataFrame({"id": itertools.repeat(syn_id, len(k_layers)), "term": itertools.repeat(term, len(k_layers)), "pref-name": itertools.repeat(pref_name, len(k_layers)), "embed-layer": k_layers, "embedding": embeds})
    # new_data, sub_data = new_data.align(sub_data)
    new_data = pd.concat([new_data, sub_data], ignore_index=True)
  return new_data


def clustering(data, alg, **kwargs):
  X = np.array(data["embedding"].to_list())
  if alg == "kmeans":
    kmeans = KMeans(n_clusters=kwargs["n_clusters"])
    cluster = kmeans.fit_predict(X)
  data["cluster"] = cluster
  return data


def scoring(data):
  """

  :param data: one layer embeddings and associated cluster
  :return:
  """
  concept_scores = []
  for concept_id in data["id"].unique():
    concept = data.where(data["id"] == concept_id).dropna()
    # highest number of same cluster instances divided by total number of instances in concept should be recall?
    recall = concept.cluster.value_counts().max() / len(concept)
    precision = -1
    for c in concept.cluster.unique():
      cluster = data.where(data["cluster"] == c).dropna()
      # ratio of cluster instances in same concept compared to all instances in cluster
      precision_c = cluster.id.value_counts().max() / len(cluster)
      if precision_c > precision:
        precision = precision_c
    # f1-score is harmonic mean of precision and recall
    score = 2*((precision*recall)/(precision+recall))
    concept_scores.append(score)
  return np.mean(concept_scores)


if __name__ == '__main__':
  # get data:
  r = requests.get("https://zenodo.org/records/7971572/files/synonym_sets.zip?download=1")
  vf = BytesIO(r.content)
  zf = zipfile.ZipFile(vf)

  # transform text data to embeddings
  data = create_dataframe(data_source)
  data = calc_embed(data)

  ## replace two lines above with following comment to save embedding results after calculation:
  # try:
    # with open("embeds", mode="rb") as file:
      # data = _pickle.load(file)
  # except FileNotFoundError:
    # data = create_dataframe(data_source)
    # data = calc_embed(data)
    # with open("embeds", mode="wb") as file:
      # _pickle.dump(data, file)

  # apply clustering
  partition = data.where(data["embed-layer"] == num_layer).dropna()
  # uses known number of concepts in text data as k for kMeans
  partition = clustering(partition, "kmeans", n_clusters=num_cluster)
  f1 = scoring(partition)
  print(f"f1 score layer {num_layer}: {f1}")

  ## for all layers iteratively use the following:
  # f1s = {}
  # for i in tqdm(range(1, num_layer + 1)):
    # partition = data.where(data["embed-layer"] == i).dropna()
    ## uses known number of concepts in text data as k for kMeans
    # partition = clustering(partition, "kmeans", n_clusters=num_cluster)
    # f1 = scoring(partition)
    # print(f"f1 score layer {i}: {f1}")
    # f1s[str(i)] = f1
  # f1s[str(num_layer)] = f1

  ## saves results to file
  # with open('results.txt', 'w') as save_file:
    # save_file.write(json.dumps(f1s))

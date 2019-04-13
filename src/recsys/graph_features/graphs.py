import gc
import glob
import os
import matplotlib.pyplot as plt
import pathlib
import time

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.feather as pf
import pyarrow.parquet as pq

from collections import Counter, OrderedDict
from datetime import datetime
from recsys.utils import group_lengths, reduce_mem_usage, timer
from recsys.df_utils import split_by_timestamp
from recsys.metric import mrr_fast
from recsys.utils import group_lengths
from recsys.vectorizers import make_vectorizer_1
from tqdm import tqdm


LOAD_TEST = True
DEBUG = True
REMOVE_ORIG = False
N_ROWS = int(1e5)
SAVE = True


def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out


def create_node_interactions(comb, name):
    comb['min_'+name] = comb[['node1_'+name, 'node2_'+name]].min(1)
    comb['max_'+name] = comb[['node1_'+name, 'node2_'+name]].max(1)
    comb['mean_'+name] = comb[['node1_'+name, 'node2_'+name]].mean(1)
    comb['sum_'+name] = comb['node1_'+name] + comb['node2_'+name]
    comb['diff_'+name] = abs(comb['node1_'+name] - comb['node2_'+name])
    return comb


with timer('reading data...'):
    train = pf.read_feather('../data/train.feather')
    print('train shape: {}'.format(train.shape))
    train['is_test'] = int(False)
    if LOAD_TEST:
        test = pf.read_feather('../data/test.feather')
        print('test shape: {}'.format(test.shape))
        test["is_test"] = (
            test["reference"].isnull() & (test["action_type"] == "clickout item")).astype(np.int)
        df_all = pd.concat([train, test], ignore_index=True, sort=False)
    else:
        df_all = train

    if DEBUG:
        print('debug mode, use {} rows.'.format(N_ROWS))
        df_all = df_all.iloc[:N_ROWS, :].reset_index(drop=True)
    if REMOVE_ORIG:
        print('remove orig train and test')
        del train, test
        gc.collect()
    
    item_metadata = pf.read_feather('../data/item_metadata.feather')
    
    df_all = reduce_mem_usage(df_all)
    test = get_submission_target(df_all)
    train = df_all.loc[~df_all.index.isin(test.index), :]
    assert train.shape[0] + test.shape[0] == df_all.shape[0]
    
    
ref_set = df_all[['impressions', 'reference']]
ref_set = ref_set.assign(reference=pd.to_numeric(ref_set['reference'], errors='coerce'))
ref_set = ref_set.loc[np.isfinite(ref_set['reference']), :]
ref_set = ref_set.loc[~pd.isnull(ref_set.impressions), :]
ref_set['impressions'] = ref_set['impressions'].apply(lambda x: x.split('|'))


start_time = time.time()
impr_unique = list(set(np.concatenate(ref_set.impressions.values)))
impr_edges = ref_set.impressions.values
edge_records = []
for edge_ in tqdm(impr_edges):
    for i in range(0, len(edge_) - 1):
        for j in range(1, len(edge_)):
            if edge_[i] != edge_[j]:
                # edge_pair = (edge_[i], edge_[j])
                # if edge_pair not in edge_records:
                edge_records.append((edge_[i], edge_[j]))
                    
edge_records_unique = list(set(edge_records))
print('unique combinations len: {}'.format(len(edge_records_unique)))
print('edge preparation: {:.2f}s'.format(time.time() - start_time))


g = nx.Graph()
g.add_nodes_from(impr_unique)
g.add_edges_from(edge_records)
print('graph preparation: {:.2f}s'.format(time.time() - start_time))

d = g.degree()
mean_deg = np.mean(list(dict(d).values()))
print('number of edges:', g.number_of_edges())
print('mean degree: {:.3f}'.format(mean_deg))


edges_records_df = pd.DataFrame(edge_records_unique, columns=['node1', 'node2'])
edges_records_df['node1_neighbor_count'] = edges_records_df['node1'].map(g.neighbors).map(lambda x: len(list(x)))
edges_records_df['node2_neighbor_count'] = edges_records_df['node2'].map(g.neighbors).map(lambda x: len(list(x)))
edges_records_df = create_node_interactions(edges_records_df, 'neighbor_count')
if SAVE:
    edges_records_df.to_csv('output/edges_records_df.csv', index=False)

    
from networkx.algorithms import *

# Dicts with features
cluster_dict = nx.cluster.clustering(g)
print('clustering done {}.'.format(str(datetime.now())))
if SAVE:
    joblib.dump(cluster_dict, 'output/cluster_dict.joblib')
# cluster_square_dict = nx.cluster.square_clustering(g)
# print('square clustering done.')
cluster_triangles_dict = nx.triangles(g)
print('triangle clustering done {}.'.format(str(datetime.now())))
if SAVE:
    joblib.dump(cluster_triangles_dict, 'output/cluster_triangles_dict.joblib')

    
avg_neighbor_deg_dict = average_neighbor_degree(g)
if SAVE:
    joblib.dump(avg_neighbor_deg_dict, 'output/avg_neighbor_deg_dict.joblib')
avg_deg_connect_dict = average_degree_connectivity(g)
if SAVE:
    joblib.dump(avg_deg_connect_dict, 'output/avg_deg_connect_dict.joblib')
print('neighbors and degrees done {}.'.format(str(datetime.now())))


graph_matched = list(map(lambda x: matching.is_matching(g, x), edge_records_unique))
graph_matched_dict = {k: v for k, v in zip(edge_records_unique, graph_matched)}
pagerank_dict = link_analysis.pagerank_alg.pagerank(g)
if SAVE:
    joblib.dump(graph_matched_dict, 'output/graph_matched_dict.joblib')
    joblib.dump(pagerank_dict, 'output/pagerank_dict.joblib')
print('features preparation: {:.2f}s'.format(time.time() - start_time))
print('ranks done {}.'.format(str(datetime.now())))
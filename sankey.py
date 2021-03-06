#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:25:15 2020

@author: subhajit
"""

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd



''' Compute how many predecessor nodes are "before" this node '''
def computeNodeDepths(adj):    
    node_depth = np.zeros(adj.shape[0])
    adjpower = (adj.to_numpy() > 0)
    sumadj = adjpower.sum()
    while sumadj > 0:
        node_depth += (adjpower.sum(axis=1) > 0)
        adjpower = (np.dot(adjpower, adj) > 0)
        sumadj = adjpower.sum()
        if np.max(node_depth) == adj.shape[0]:
            raise Exception('Circular Link!')
    
    return node_depth



''' Compute the nodes & their details from dataframe of links '''
def computeNodePositions(df):
    # calculate adjacency matrix
    adj = pd.crosstab(df.target, df.source)
    idx = adj.columns.union(adj.index)
    adj = adj.reindex(index = idx, columns=idx, fill_value=0)
    
    vertical_gap_quotient = 0.05
    nodes = pd.DataFrame({'name': adj.index})
    nodes['depth'] = computeNodeDepths(adj)
    nodes.sort_values(['depth', 'name'], inplace=True)
    max_layer_height = df.merge(nodes, how='left', left_on='source', right_on='name').groupby('depth')['value'].sum().max()
    frame_height = (1 + len(nodes) * vertical_gap_quotient) * max_layer_height
    frame_width = frame_height * 4/3
    
    nodes['inflow'] = nodes['name'].map(df.groupby('target')['value'].sum().to_dict()).fillna(0)
    nodes['outflow'] = nodes['name'].map(df.groupby('source')['value'].sum().to_dict()).fillna(0)
    nodes['width'] = vertical_gap_quotient * frame_width
    nodes['height'] = nodes[['inflow', 'outflow']].max(axis=1).astype(df.value.dtype)
    nodes['x'] = nodes.depth * frame_width / nodes.depth.max()
    nodes['y'] = 0
    for d in range(int(nodes.depth.max())+1):
        num_nodes = np.sum(nodes.depth == d)
        nodes.loc[nodes.depth == d, 'y'] += (nodes[nodes.depth == d].shift(1)['height'].cumsum() + np.arange(num_nodes) * vertical_gap_quotient * max_layer_height).fillna(0)
    #nodes['y'] = -nodes['y']-nodes['height']
    
    return nodes



''' Get node & link details from dataframe of links '''
def getNodesAndLinks(df):
    links = df.copy()
    nodes = computeNodePositions(df)
    links['source_depth'] = links['source'].map(dict(zip(nodes['name'], nodes['depth'])))
    links['target_depth'] = links['target'].map(dict(zip(nodes['name'], nodes['depth'])))
    links['depth'] = links['target_depth'] - links['source_depth']
    links.sort_values(['depth', 'source', 'target'], inplace=True)
    nodes['in_y'] = nodes['out_y'] = nodes['y']
    
    return nodes, links



''' Plot the sankey diagram '''
def sankey(df, labels=True, labelsize=5, cmap=None, edgecolor='white'):
    nodes, links = getNodesAndLinks(df)
    fig, ax = plt.subplots()
    for i, link in links.iterrows():
        startx = (nodes[nodes.name==link.source]['x'] + nodes[nodes.name==link.source]['width']).values[0]
        endx = (nodes[nodes.name==link.target]['x']).values[0]
        starty = (nodes[nodes.name==link.source]['out_y']).values[0]
        endy = (nodes[nodes.name==link.target]['in_y']).values[0]
        nodes.loc[nodes.name==link.source, 'out_y'] = starty + link['value']
        nodes.loc[nodes.name==link.target, 'in_y'] = endy + link['value']
        linkstretchx = endx - startx
        linkstretchy = endy - starty
        x = np.array([startx, startx+linkstretchx/4, endx-linkstretchx/4, endx])
        y = np.array([starty, starty+linkstretchy/5, endy-linkstretchy/5, endy])
        f = interp1d(x, y, kind='cubic')
        points = [[ix, f(ix)] for ix in np.linspace(startx, endx, 100)]
        points += [(coord[0], coord[1]+link['value']) for coord in points[::-1]]
        connector = Polygon(points, fc='gray', alpha=0.5)
        ax.add_patch(connector)
    
    
    nplots = [Rectangle((row['x'], row['y']), row['width'], row['height']) for i, row in nodes.iterrows()]
    pc = PatchCollection(nplots, cmap='viridis', array=nodes.depth, ec=edgecolor, alpha=0.5)
    ax.add_collection(pc)
    
    if labels:
        for i, row in nodes.iterrows():
            ax.text(row['x'] + row['width'] * 1.2, row['y'] + row['height'] / 2, row['name'] + ' ' + str(row['height']), fontsize=labelsize, va='center')
    
    plt.axis('scaled')
    plt.axis('off')
    
    return fig, ax



df = pd.read_csv('data.csv')
sankey(df, labels=True, labelsize=5, cmap=None, edgecolor='white')
plt.savefig('sankey.png', dpi=1200)
plt.close()
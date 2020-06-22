#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 09:28:31 2020

@author: subhajit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def buildTree(df, prev_path, curr_col, curr_set, depth, max_depth=5, typ='left'):
    global branches
    
    if depth >= max_depth:
        return
    
    newdf = df.drop(curr_col, axis=1)
    if newdf.shape[1] > 0:
        nxt = newdf[curr_set==1].sum().idxmax()
    else:
        return
    
    if typ=='right':
        curr_col = 'no '+curr_col
    prev_path = ', '.join([prev_path, curr_col]) if prev_path != '' else curr_col
    
    left_child = curr_set & df[nxt]
    branches.append((prev_path, prev_path+', '+nxt, left_child.sum()))
    buildTree(newdf, prev_path, nxt, left_child, depth+1, max_depth)
    
    right_child = curr_set & (1-df[nxt])
    branches.append((prev_path, prev_path+', '+'no '+nxt, right_child.sum()))
    buildTree(newdf, prev_path, nxt, right_child, depth+1, max_depth, typ='right')


df = pd.DataFrame(np.random.choice([0,1], size = (10000, 5)), columns=list('ABCDE'))
start_col = 'A'
start_set = df[start_col]
branches = []
buildTree(df, '', start_col, start_set, depth=1, max_depth=4)

links = pd.DataFrame(branches, columns=['source', 'target', 'value'])
labeldict = {}

for i, row in links.iterrows():
    labeldict[row['target']] = {'label': ', '.join([p for p in row['target'].split(', ') if 'no' not in p])}

for i, row in links.iterrows():
    labeldict[row['source']] = {'label': row['source'].split(', ')[-1]}
    #labeldict[row['target']] = {'label': row['target'].split(', ')[-1]}
    
from psankey.sankey import sankey
fig, ax = sankey(links, nodecolorby='size', nodecmap='viridis', nodemodifier=labeldict)
plt.savefig('test.png', dpi=1200)
plt.close()

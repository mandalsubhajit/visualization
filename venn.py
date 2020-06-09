#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:49:19 2020

@author: Subhajit Mandal
"""

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
from functools import reduce
from itertools import product, compress
from scipy.optimize import bisect, minimize
from sklearn.manifold import MDS
import pandas as pd

''' Calculates the approximate intersection areas based on a projection
of circles onto a 200x200 pixel matrix '''
def calc_overlap_area(circles):
    left = min([c[0][0]-c[1] for c in circles])
    right = max([c[0][0]+c[1] for c in circles])
    bottom = min([c[0][1]-c[1] for c in circles])
    top = max([c[0][1]+c[1] for c in circles])
    
    rmin = min([c[1] for c in circles])
    scale_min = min(left, right, bottom, top)
    scale_max = max(left, right, bottom, top)
    granularity = max(200, int(10*(scale_max-scale_min)/rmin))
    scale = np.linspace(scale_min, scale_max, granularity)
    
    x = np.array([scale,]*granularity)
    y = x.transpose()
    
    unit = granularity/(scale_max-scale_min)
    
    cp = list(map(np.vectorize(lambda b: '1' if b else '0'), [(x-c[0][0])**2 + (y-c[0][1])**2 < c[1]**2 for c in circles]))
    intersectionIds = reduce(np.char.add, cp)
    
    unique, counts = np.unique(intersectionIds, return_counts=True)
    counts = counts.astype(float)/unit**2
    intersectionAreas = dict(zip(unique, counts))
    del intersectionAreas['0'*len(cp)]
    
    return x, y, intersectionIds, intersectionAreas


# Circular segment area calculation. See http://mathworld.wolfram.com/CircularSegment.html
def circleArea(r, width):
    return r * r * np.arccos(1 - width/r) - (r - width) * np.sqrt(width * (2 * r - width));



''' Returns the overlap area of two circles of radius r1 and r2 - that
have their centers separated by distance d. Simpler faster
circle intersection for only two circles '''
def circleOverlap(r1, r2, d):
    # no overlap
    if (d >= r1 + r2):
        return 0
    
    # completely overlapped
    if (d <= np.abs(r1 - r2)):
        return np.pi * np.min([r1, r2]) * np.min([r1, r2])
    
    w1 = r1 - (d * d - r2 * r2 + r1 * r1) / (2 * d)
    w2 = r2 - (d * d - r1 * r1 + r2 * r2) / (2 * d)
    return circleArea(r1, w1) + circleArea(r2, w2)


''' Returns the distance necessary for two circles of radius r1 + r2 to
have the overlap area 'overlap' '''
def distanceFromIntersectArea(r1, r2, overlap):
    # handle complete overlapped circles
    if (np.min([r1, r2]) * np.min([r1, r2]) * np.pi <= overlap + 1e-10):
        return np.abs(r1 - r2)
    
    return bisect(lambda d: circleOverlap(r1, r2, d) - overlap, 0, r1 + r2)



''' Given a bunch of sets, and the desired overlaps between these sets - computes
the distance from the actual overlaps to the desired overlaps. Note that
this method ignores overlaps of more than 2 circles '''
def lossFunction(centers, radii, overlaps):
    output = 0;
    
    assert len(centers)%2 == 0, 'number parameters should be a multiple of 2 (2 xy co-ordinates for center of each circle)'
    assert len(centers)/2 == len(radii), 'number of centers & number of radii do not match'
    circles =  []
    for i in range(int(len(centers)/2)):
        circles.append([(centers[2*i+0], centers[2*i+1]), radii[i]])
    
    x, y, intersectionIds, curr_overlap = calc_overlap_area(circles)
    sst = len(overlaps)*np.var(list(overlaps.values()))
    for k in overlaps:
        error = overlaps[k] - curr_overlap[k] if k in curr_overlap else overlaps[k]
        output += error * error / sst
    
    return output




''' Calculates the intersection between columns of a dataframe '''
def df2areas(df):
    radii = np.sqrt(df.sum()/np.pi).tolist()
    labels = df.columns
    
    # intersection of two sets - may be overlapped with other sets - A int B
    actualOverlaps = {}
    for comb in product(*[[True,False]]*len(radii)):
        if comb != (False, )*len(radii):
            olap = np.sum(reduce(lambda x, y: x & y, [df[col] for col in compress(df.columns, comb)]))
            actualOverlaps[''.join([str(int(b)) for b in comb])] = olap
    
    # intersection of two sets only - not overlapped with any other set - A int B int (not C)
    disjointOverlaps = {}
    for comb in product(*[[True,False]]*len(radii)):
        if comb != (False, )*len(radii):
            temp = []
            for i, b in enumerate(comb):
                if b:
                    temp.append(df.iloc[:,i])
                else:
                    temp.append(1-df.iloc[:,i])
            olap = np.sum(reduce(lambda x, y: x & y, temp))
            disjointOverlaps[''.join([str(int(b)) for b in comb])] = olap
    
    return labels, radii, actualOverlaps, disjointOverlaps


''' Computes the circles from radius and overlap data
If fineTune=False, returns initial estimates - faster & fairly acurate - should be fit for most cases
If fineTune=True, returns optimized estimates - slower but more accurate '''
def getCircles(radii, actualOverlaps, disjointOverlaps, fineTune=False):
    distances = np.zeros((len(radii), (len(radii))))
    for i in range(len(radii)):
        for j in range(len(radii)):
            if i == j:
                distances[i, j] = 0
            else:
                idxmin, idxmax = np.min([i, j]), np.max([i, j])
                combstr = '0'*idxmin+'1'+'0'*(idxmax-idxmin-1)+'1'+'0'*(len(radii)-idxmax-1)
                distances[i, j] = distanceFromIntersectArea(radii[i], radii[j], actualOverlaps[combstr])
    
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=42,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(distances).embedding_
    circles = [[(pos[i,0], pos[i,1]), radii[i]] for i in range(len(radii))]
    
    if fineTune:
        centers = []
        for i in range(pos.shape[0]):
            centers += [pos[i, 0], pos[i, 1]]
        
        res = minimize(lambda p: lossFunction(p, radii, disjointOverlaps), centers, method='Nelder-Mead', options={'maxiter': 500, 'disp': True})
        
        centers = list(res['x'])
        circles =  []
        for i in range(int(len(centers)/2)):
            circles.append([(centers[2*i+0], centers[2*i+1]), radii[i]])
    
    return circles


''' Get label positions for each circle avoiding the overlap areas '''
def getLabelPositions(circles, labels):
    x, y, intersectionIds, curr_overlap = calc_overlap_area(circles)
    olapByNset = [dict() for x in range(len(circles))]
    for k in curr_overlap:
        n = k.count('1')
        olapByNset[n-1][k] = curr_overlap[k]
    
    areaTol = 0.01*(np.max(x)-np.min(x))*(np.max(y)-np.min(y))
    for i, (l, c) in enumerate(zip(labels, circles)):
        olapC = [filterTheDict(ol, lambda elem: (elem[0][i] == '1') and (elem[1] > areaTol)) for ol in olapByNset]
        for ol in olapC:
            if len(ol) > 0:
                break
        areaId = max(ol, key=lambda x: ol[x])
        indices = np.where(intersectionIds == areaId)
        rndx, rndy = int(np.median(indices[0])), int(np.median(indices[1][np.where(indices[0]==int(np.median(indices[0])))]))
        lx, ly = x[rndx, rndy], y[rndx, rndy]
        yield l, lx, ly

''' Dictionary filter utility function - filter a dictionary basedon criteria'''
def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict


''' Plots the Venn diagrams from radius and overlap data '''
def venn(radii, actualOverlaps, disjointOverlaps, labels=None, cmap=None, fineTune=False):
    circles = getCircles(radii, actualOverlaps, disjointOverlaps, fineTune)
    fig, ax = plt.subplots()
    cplots = [plt.Circle(circles[i][0], circles[i][1]) for i in range(len(circles))]
    arr = np.array(radii)
    col = PatchCollection(cplots, cmap=cmap, array=arr, alpha=0.5)
    ax.add_collection(col)
    
    if labels is not None:
        for l, lx, ly in getLabelPositions(circles, labels):
            ax.annotate(l, xy=(lx, ly), fontsize=15, ha='center', va='center')
    
    ax.axis('off')
    ax.set_aspect('equal')
    ax.autoscale()
    
    return fig, ax



''' Usage Example '''
if __name__ == '__main__':
    df = pd.DataFrame(np.random.choice([0,1], size = (1000, 5)), columns=list('ABCDE'))
    labels, radii, actualOverlaps, disjointOverlaps = df2areas(df)
    fig, ax = venn(radii, actualOverlaps, disjointOverlaps, labels=labels, cmap=None, fineTune=False)
    plt.savefig('venn.png', dpi=300, transparent=True)
    plt.close()

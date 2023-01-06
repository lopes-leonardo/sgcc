# distutils: language = c++

from libcpp.map cimport map
from libcpp.vector cimport vector
from libc.stdlib cimport malloc
from sklearn.neighbors import BallTree
import cython
import numpy as np


cdef class LhrrWrapper:
    cdef Lhrr lhrr
    cdef int k, t, l, n
    cdef vector[vector[int]] D
    cdef float *imatrix

    def __init__(self, 
                  k=20,
                  t=2):
        self.k = k
        self.l = 4*k
        self.t = t
        self.lhrr = Lhrr(self.k, self.l, self.t)
        
    def run(self,
            data,
            metric:str = 'euclidean'):
        np_data = np.array(data)
        x, y = np_data.shape
        if x == y:
            print("WARNING -> Input features present symmetric shape. Using it as a Distance Matrix.")
            self._init_with_distance_matrix(np_data)
        else:
            self._init_with_features(np_data, metric)
        self.lhrr.runMethod()

    def get_confid_scores(self):
        return self.lhrr.getConfidScores()
    
    def get_hyper_edges(self):
        return self.lhrr.getHyperEdges()

    def get_ranked_lists(self):
        vector_rk = np.asarray(self.lhrr.getRankedLists())
        ranked_lists = []
        for i in range(self.n):
            start = self.l * i
            end = start + self.l
            ranked_lists.append(vector_rk[start:end])
        return np.asarray(ranked_lists)

    def _init_input_matrix(self, n):
        self.imatrix = <float *>malloc(n*n*cython.sizeof(float))

    def _init_with_features(self, points, metric):
        dataSize = len(points)
        self.n = dataSize
        self.lhrr.setN(dataSize)
        self._init_input_matrix(dataSize)
        ranked_lists = []
        balltree = BallTree(points, leaf_size=50, metric=metric)
        for index, point in enumerate(points):
            rank, rank_map = balltree.query([point], k=self.l)
            ranked_lists.extend(rank_map[0])
        self.lhrr.setComputedRanks(ranked_lists)

    def _init_with_distance_matrix(self, points):
        dataSize = len(points)
        self.n = dataSize
        self.lhrr.setN(dataSize)
        self._init_input_matrix(dataSize)

        i = 0
        for index, ranking in enumerate(points):
            for value in ranking:
                self.imatrix[i] = value
                i += 1
        self.lhrr.setMatrix(self.imatrix)
        self.lhrr.computeRanksFromMatrix()
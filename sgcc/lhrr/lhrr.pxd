cdef extern from "source.cpp":
    pass
cdef extern from "Effectiveness.cpp":
    pass

from libcpp.map cimport map as mapcpp
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp cimport bool as bool_t

cdef extern from "source.hpp" namespace "udl":
    cdef cppclass Lhrr:
        Lhrr() except +
        Lhrr(int, int, int) except +
        void validateParameters();
        void setMatrix(float*);
        void printMatrix();
        void setN(int);
        void computeRanksFromMatrix();
        void setComputedRanks(vector[int]);
        void computeMap();
        void runMethod();
        vector[vector[pair[int, double]]] getHyperEdges() except +
        vector[int] getRankedLists() except +
        vector[double] getConfidScores() except +

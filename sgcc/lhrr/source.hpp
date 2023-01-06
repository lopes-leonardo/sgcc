/* <source.hpp>
 *
 *
 * This file contains main data structures for lhc clusterization method.
 *
 *
 * @author: Leonardo Tadeu Lopes
 */

#ifndef DATA_H
#define DATA_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <list>
#include <stdlib.h>
#include <iostream> 
#include <cstdlib> 
#include "Effectiveness.hpp"

namespace udl {
	class Lhrr {
		public:
			// Initialization
			Lhrr();
			Lhrr(int, int, int);
			void validateParameters();
			void setMatrix(float*);
			void setN(int);
			void initSparseMatrix();
			// Compute Ranks
			void setComputedRanks(std::vector<int> tmpRkList);
			void computeRanksFromMatrix();
			void heapsort(std::vector<float>&, std::vector<int>&, int);
			void exchange(std::vector<float>&, std::vector<int>&, int, int);
			void downheap(std::vector<float>&, std::vector<int>&, int, int);
			void buildheap(std::vector<float>&, std::vector<int>&, int);
			// Method Execution
			void runMethod();
			void execFillPosMatrix();
			void kernelFillPosMatrix(int);
			void execSortRankedLists();
			void kernelSortRankedLists(int);
			void hyperGraphIteration();
			void initializeDataStructures();
			void loadHyperEdges();
			void createHyperEdge(int);
			void includeHyperEdgeValue(int, int, double);
			int searchPairByKey(int, std::vector<std::pair<int, double>>&);
			void compressHE();
			void loadRevHyperEdges();
			void resetDB(int);
			void computeCartesianProductHyperEdges();
			void computeHyperEdgesSimilarities();
			void computeReciprocalHyperEdgesSimilarities();
			void computeDBBySimilarities();
			void sortTmpList(int);
			void joinRks(int);
			double weightPosition(int);
			// Extract information
			std::vector<std::vector<std::pair<int, double>>> getHyperEdges();
			std::vector<int> getRankedLists();
			std::vector<double> getConfidScores();
			
			int k, l, t, n = 0;
			float soma = 0.0;
			std::vector<int> rkLists;
			std::vector<std::vector<int>> imgHyperEdges;
			std::vector<std::vector<std::pair<int, double>>> hyperEdges;
            std::vector<std::vector<std::pair<int, double>>> revHyperEdges;
            std::vector<double> confid;
            std::vector<std::vector<int>> imgList;
            std::vector<std::vector<int>> imgListRev;
            std::vector<std::vector<float_t>> valList;
            std::vector<std::vector<float_t>> valListRev;
			float* matrix = NULL;
			std::vector<std::vector<int>> tmpList;

		protected:

		private:
	};
};

#endif
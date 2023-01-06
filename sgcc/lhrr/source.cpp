/* <source.cpp>
 *
 * Lhc clustering method implementation file
 *
 * @Author: Leonardo Tadeu Lopes
 *
 *
 *****************************************************************************************************************
 *
 *
 * This file contains the methods implementation for lhc object class
 */

#include "source.hpp"
#include <iostream>

namespace udl {

    // Initialization
    Lhrr::Lhrr() {
    }
    Lhrr::Lhrr(int k, int l, int t) {
        if (k > l) {
            std::cerr << "Error - Parameter k can't be grater than parameter l!" << std::endl;
            exit(1);
        }

        this->k = k;
        this->l = l;
        this->t = t;
    }
    void Lhrr::setMatrix(float* tmpMatrix) {
        matrix = tmpMatrix;
    }
    void Lhrr::setN(int n) {
        if(l > n) {
            std::cerr << "Error - Parameter l can't be greater than dataset size 'n'!" << std::endl;
            exit(1); 
        }
        this->n = n;
    }
    void Lhrr::setComputedRanks(std::vector<int> tmpRkList) {
        if((int)tmpRkList.size() != n * l) {
            std::cerr << "Error - Ranked lists must be of size n * l!" << std::endl;
            exit(1);
        }

        rkLists = tmpRkList;
    }
    void Lhrr::initSparseMatrix() {
        delete [] matrix;
        matrix = new float[n*n];

        for (long int l = 0; l < n*n; l++) {
            if (matrix[l] != 0) {
                matrix[l] = 0;
            }
        }
    }

    // Heap to calculate ranks 
    void Lhrr::computeRanksFromMatrix() {
        if(matrix == NULL) {
            std::cerr << "Error - Trying to compute rankings without distance matrix!" << std::endl;
            exit(1);
        }
        if(n == -1) {
            std::cerr << "Error - Trying to compute rankings without setting the dataset size 'n'!" << std::endl;
            exit(1);
        }

        rkLists.clear();
        rkLists.resize(n*l);

        for (int rk = 0; rk < this->n; rk++) {
            std::vector<float> distances(this->n);
            std::vector<int> curRk(this->n);
            for (int j = 0; j < this->n; j++) {
                curRk[j] = j;
                distances[j] = this->matrix[n*rk + j];
            }
            heapsort(distances, curRk, this->n);
            int l = rkLists.size()/n;
            for (int j = 0; j < l; j++) {
                rkLists[l*rk + j] = curRk[j];
            }
        }
    }
    void Lhrr::heapsort(std::vector<float>& distances, std::vector<int>& curRk, int n) {
        buildheap(distances, curRk, n);
        while (n > 1) {
            n--;
            exchange(distances, curRk, 0, n);
            downheap(distances, curRk, n, 0);
        }
    }
    void Lhrr::exchange(std::vector<float>& distances, std::vector<int>& curRk, int i, int j) {
        //Distances
        float t = distances[i];
        distances[i] = distances[j];
        distances[j] = t;
        //Ranked Lists
        int trk = curRk[i];
        curRk[i] = curRk[j];
        curRk[j] = trk;
    }
    void Lhrr::downheap(std::vector<float>& distances, std::vector<int>& curRk, int n, int v) {
        int w = 2 * v + 1; //first descendant of v
        while (w < n) {
            if (w + 1 < n) {
                if (distances[w + 1] > distances[w]) {
                    w++;
                }
            }
            if (distances[v] >= distances[w]) {
                return;
            }
            exchange(distances, curRk, v, w);
            v = w;
            w = 2 * v + 1;
        }
    }
    void Lhrr::buildheap(std::vector<float>& distances, std::vector<int>& curRk, int n) {
        for (int v = n / 2 - 1; v >= 0; v--) {
            downheap(distances, curRk, n, v);
        }
    }

    // Method execution
    void Lhrr::runMethod() {
        initSparseMatrix();
        execFillPosMatrix();
        execSortRankedLists();

        int iteration = 1;
        while (iteration <= t) {
            hyperGraphIteration();
            iteration++;
        }
    }
    void Lhrr::execFillPosMatrix() {
        for (int i = 0; i < n; i++) {
            kernelFillPosMatrix(i);
        }
    }
    void Lhrr::kernelFillPosMatrix(int rk) {
        long int lrk = ((long int) l)*rk;
        long int nrk = ((long int) n)*rk;
        long int img;

        for (int pos = 0; pos < l; pos++) {
            img = rkLists[lrk + pos];
            matrix[nrk + img] += l - pos;
            matrix[n*img + rk] += l - pos;
        }
    }
    void Lhrr::execSortRankedLists() {
        for (int i = 0; i < n; i++) {
            kernelSortRankedLists(i);
        }
    }
    void Lhrr::kernelSortRankedLists(int rk) {
        long int LcurRL = ((long int) l)*rk;
        long int cNcurRL = ((long int) n)*rk;
        float a[l];

        long int index;
        for (int j = 0; j < l; j++) {
            index = cNcurRL + rkLists[LcurRL + j];
            a[j] = matrix[index];
        }

        //---------------------- INSERTION SORT --------------------------
        int i, j, keyR;
        float keyA;

        for (j = 2; j < l; j++) {
            keyA = a[j];
            keyR = rkLists[LcurRL + j];
            i = j - 1;
            while (i > 0 && (a[i] < keyA)) {
                a[i + 1] = a[i];
                rkLists[LcurRL + i + 1] = rkLists[LcurRL + i];
                i--;
            }
            a[i + 1] = keyA;
            rkLists[LcurRL + i + 1] = keyR;
        }

        //----------------------------------------------------------------

        //Setting query image at first position
        i = 0;
        while ((rkLists[LcurRL + i] != rk)&&(i < l)) {
            i++;
        }
        if (i > 0) {
            int aux = rkLists[LcurRL + 0];
            rkLists[LcurRL + 0] = rkLists[LcurRL + i];
            rkLists[LcurRL + i] = aux;

            float auxA = a[0];
            a[0] = a[i];
            a[i] = auxA;
        }
    }
    void Lhrr::hyperGraphIteration() {
        initializeDataStructures();

        loadHyperEdges();
        compressHE();

        loadRevHyperEdges();

        initSparseMatrix();
        resetDB(1);

        computeCartesianProductHyperEdges();

        computeHyperEdgesSimilarities();

        computeReciprocalHyperEdgesSimilarities();

        computeDBBySimilarities();
    }
    void Lhrr::initializeDataStructures() {
        hyperEdges.clear();
        revHyperEdges.clear();
        confid.clear();
        imgHyperEdges.clear();

        hyperEdges.resize(n);
        revHyperEdges.resize(n);
        confid.resize(n);
        imgHyperEdges.resize(n);

        tmpList.clear();
        tmpList.resize(n);

        imgList.clear();
        imgListRev.clear();
        valList.clear();
        valListRev.clear();

        imgList.resize(n);
        imgListRev.resize(n);
        valList.resize(n);
        valListRev.resize(n);
    }
    void Lhrr::loadHyperEdges() {
        for (int i = 0; i < n; i++) {
            createHyperEdge(i);
        }
    }
    void Lhrr::createHyperEdge(int img) {
        includeHyperEdgeValue(img, img, 1);
        for (int o = 0; o < k; o++) {
            int imgo = rkLists[l*img + o];
            int poso = o + 1;
            for (int j = 0; j < k; j++) {
                int imgj = rkLists[l*imgo + j];
                int posj = j + 1;
                double w = weightPosition(poso) * weightPosition(posj);
                includeHyperEdgeValue(img, imgj, w);
            }
        }

        //sort
        std::sort(hyperEdges[img].begin(), hyperEdges[img].end(), [](const std::pair<int, double> &x, const std::pair<int, double> &y) { //sorting
            return x.second > y.second;
        });

        //compute confidence
        for (int i = 0; i < k; i++) {
            confid[img] += hyperEdges[img][i].second;
        }
    }
    void Lhrr::includeHyperEdgeValue(int i, int j, double value) {
        double curValue = 0;
        int pos = searchPairByKey(j,hyperEdges[i]);
        if (pos != -1) {
            curValue = hyperEdges[i][pos].second;
            hyperEdges[i][pos].second = curValue + value;
        } else {
            hyperEdges[i].push_back(std::make_pair(j,value));
            imgHyperEdges[j].push_back(i);
        }

        //add elements
        imgList[i].push_back(j);
        valList[i].push_back(value);
    }
    double Lhrr::weightPosition(int pos) {
        double logValue = log(pos) / log(k);
        return (1.0 - logValue);
    }
    int Lhrr::searchPairByKey(int key, std::vector<std::pair<int, double>>& hyperEdge) {
        int i = 0;
        for (auto it : hyperEdge) {
            if (it.first == key) {
                return i;
            }
            i++;
        }
        return -1;
    }
    void Lhrr::compressHE() {
        for (int i = 0; i < n; i++) {
            //init structures
            std::vector<int> tmpImgList;
            std::vector<float_t> tmpValList;
            float tmpArr[n];
            for (int j = 0; j < n; j++) {
                tmpArr[j] = 0;
            }

            //fill array
            int j = 0;
            for (const int& img : imgList[i]) {
                if ((tmpArr[img] == 0) && (valList[i][j] != 0)){
                    tmpImgList.push_back(img);
                }
                tmpArr[img] += valList[i][j];
                j++;
            }

            //fill values
            for (const int& img : tmpImgList) {
                tmpValList.push_back(tmpArr[img]);
            }

            //bind new lists
            imgList[i] = tmpImgList;
            valList[i] = tmpValList;
        }
    }
    void Lhrr::loadRevHyperEdges() {
        for (int img = 0; img < n; img++) {
            for (auto entry : hyperEdges[img]) {
                int curImg =  entry.first;
                double curValue = entry.second;
                //add elements
                imgListRev[curImg].push_back(img);
                valListRev[curImg].push_back(curValue);
            }
        }
    }
    void Lhrr::resetDB(int value) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                long int index = ((long int) n)*i + rkLists[l*i + j];
                matrix[index] = value;
            }
        }
    }
    void Lhrr::computeCartesianProductHyperEdges() {
        for (int img = 0; img < n; img++) {
            double conf = confid[img];
            auto curHEdge = hyperEdges[img];
            for (auto entry1 : curHEdge) { //for the first k hyper edges or for all of them?
                int img1 = entry1.first;
                double value1 = entry1.second;
                for (auto entry2 : curHEdge) { //for the first k hyper edges or for all of them?
                    int img2 = entry2.first;
                    double value2 = entry2.second;
                    long int index = ((long int) n)*img1 + img2;
                    double inc = conf * value1 * value2;
                    if (matrix[index] == 0) {
                        tmpList[img1].push_back(img2);
                        matrix[index] = 1;
                    }
                    matrix[index] += inc; //it is an increment, so the similarity increases
                }
            }
        }
    }
    void Lhrr::computeHyperEdgesSimilarities() {
        for (int qimg = 0; qimg < n; qimg++) {

            int j = 0;
            float tmpArr[n];
            for (int i = 0; i < n; i++) {
                tmpArr[i] = 0;
            }
            for (const int& img : imgList[qimg]) {
                tmpArr[img] = valList[qimg][j];
                j++;
            }


            for (int j = 1; j < l; j++) {
                int img = rkLists[l*qimg + j];

                double simValue = 0;
                int o = 0;
                for (const int& e : imgList[img]) {
                    simValue += tmpArr[e]*valList[img][o];
                    o++;
                }

                long int index = ((long int) n)*qimg + img;
                double curValue = matrix[index];
                double newValue = curValue * simValue;

                matrix[index] = newValue;
            }
        }
    }
    void Lhrr::computeReciprocalHyperEdgesSimilarities() {
        for (int qimg = 0; qimg < n; qimg++) {
            int j = 0;
            float tmpArr[n];
            for (int i = 0; i < n; i++) {
                tmpArr[i] = 0;
            }
            for (const int& img : imgListRev[qimg]) {
                tmpArr[img] = valListRev[qimg][j];
                j++;
            }

            for (int j = 1; j < l; j++) {
                int img = rkLists[l*qimg + j];

                double simValue = 0;
                int o = 0;
                for (const int& e : imgList[img]) {
                    simValue += tmpArr[e]*valList[img][o];
                    o++;
                }

                long int index = ((long int) n)*qimg + img;
                double curValue = matrix[index];
                double newValue = curValue * simValue;
                matrix[index] = newValue;
            }
        }
    }
    void Lhrr::computeDBBySimilarities() {

        for (int i = 0; i < n; i++) {
            sortTmpList(i);
        }

        execSortRankedLists();

        for (int i = 0; i < n; i++) {
            joinRks(i);
        }
    }
    void Lhrr::sortTmpList(int qimg) {
        std::vector<std::pair<int, float>> rkTmp;

        for (int i = 0; i < (int)tmpList[qimg].size(); i++) {
            int img = tmpList[qimg][i];
            long int index = ((long int) n)*qimg + img;
            rkTmp.push_back(std::make_pair(img, matrix[index]));
        }

        std::sort(rkTmp.begin(), rkTmp.end(), [](const std::pair<int, float> &x, const std::pair<int, float> &y) { //sorting
            return x.second > y.second;
        });

        for (int i = 0; i < (int)tmpList[qimg].size(); i++) {
            tmpList[qimg][i] = rkTmp[i].first;
        }
    }
    void Lhrr::joinRks(int qimg) {
        if ((int)tmpList[qimg].size() == 0) {
            return;
        }


        float a1[l];
        long int LcurRL = ((long int) l)*qimg;
        long int cNcurRL = ((long int) n)*qimg;

        float a2[(int)tmpList[qimg].size()];

        long int index;
        for (int j = 0; j < l; j++) {
            index = cNcurRL + rkLists[LcurRL + j];
            a1[j] = matrix[index];
        }

        std::vector<int> rkTmp;
        rkTmp.clear();

        for (int j = 0; j < (int)tmpList[qimg].size(); j++) {
            index = cNcurRL + tmpList[qimg][j];
            a2[j] = matrix[index];
            rkTmp.push_back(tmpList[qimg][j]);
        }

        int j = 0;
        for (int i = 1; i < l; i++) {
            if (a2[j] > a1[i]) {
                for (int o = l-1; o > i; o--) {
                    rkLists[LcurRL + o] = rkLists[LcurRL + o - 1];
                }
                rkLists[LcurRL + i] = tmpList[qimg][j];
                j++;
                if (j >= (int)tmpList[qimg].size()) {
                    break;
                }
            }
        }
    }
    std::vector<double> Lhrr::getConfidScores() {
        return this->confid;
    }
    std::vector<int> Lhrr::getRankedLists() {
        return this->rkLists;
    }
    std::vector<std::vector<std::pair<int, double>>> Lhrr::getHyperEdges() {
        return this->hyperEdges;
    }
}
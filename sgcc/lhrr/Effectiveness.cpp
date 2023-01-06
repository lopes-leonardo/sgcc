/* <Effectiveness.cpp>
 *
 * Effectiveness class implementantion file
 *
 * @Authors: Lucas Pascotti Valem <lucasvalem@rc.unesp.br>
 *           Daniel Carlos Guimarães Pedronette <daniel@rc.unesp.br>
 *
 ***********************************************************************************
 *
 * This file is part of Unsupervised Distance Learning Framework (UDLF).
 *
 * UDLF is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * UDLF is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with UDLF.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "Effectiveness.hpp"

/* Constructor */
Effectiveness::Effectiveness(int& n_in, std::vector<int>& rkLists_in)
: n(n_in)
, rkLists(rkLists_in)
{

}

void Effectiveness::readImagesList(std::string listsFile) {
    // std::cout << "\n Starting reading images list ... [" << listsFile << "]  \n";

    imgList.clear();
    std::ifstream inFile;
    inFile.open(listsFile.c_str());
    if (!inFile) {
        std::cerr << " Unable to open image list file [" << listsFile << "].\n";
        exit(1); //terminate with error
    }
    std::string line;
    int i = 0;
    while (inFile >> line) {
        imgList.push_back(line);
        i++;
    }
    inFile.close();

    if ((int)imgList.size() != n) {
        // std::cout << " Your image list file is invalid for a dataset of " << n <<
        // " images! There are " << imgList.size() << " images in it! "<< std::endl;
        exit(1); //terminate with error
    }

    // std::cout << " Done! \n";
}

void Effectiveness::readClassesFile(std::string classFile) {
    // std::cout << "\n Starting reading classes file ... [" << classFile << "]  \n";

    std::ifstream inFile;
    inFile.open(classFile);
    if (!inFile) {
        std::cerr << " Unable to open classes file [" << classFile << "].";
        exit(1); // terminate with error
    }

    classes.clear();
    classesSize.clear();

    std::string line;
    while (inFile >> line) {
        int pos = line.find(":", 0);
        std::string imgName   = line.substr(0, pos);
        std::string className = line.substr(pos+1, line.length());
        classes[imgName] = className;
        classesSize[className]++;
    }
    inFile.close();
}

float Effectiveness::computeMAPMeasure() {
    // std::cout << "\n Computing MAP Measure ...";

    mapByClass.clear();
    int paramNK = rkLists.size()/n; // get the l value
    float map = computeMAP(paramNK);

    return map;
}

float Effectiveness::computeMAP(int paramNK) {
    float acumAP = 0;
    startMAPByClass();
    for (int i = 0; i < n; i++) {
        acumAP += computeAveragePrecision(i,paramNK,paramNK);
    }
    endMAPByClass();
    float Nq = n;
    float map = acumAP / Nq;
    return map;
}

float Effectiveness::computeAveragePrecision(int qId, int d, int offset) {
    float sumrj = 0;
    float curPrecision = 0;
    float sumPrecision = 0;
    std::string qClass = getClass(qId);
    for (int i = 0; i < d; i++) {
        int imgi = rkLists[offset*qId + i];
        std::string imgiClass = getClass(imgi);
        if (qClass == imgiClass) {
            sumrj = sumrj + 1;
            float posi = i + 1;
            curPrecision = sumrj / posi;
            sumPrecision += curPrecision;
        }
    }
    float nRel = getClassSize(getClass(qId));
    float l = rkLists.size()/n;
    float avgPrecision = sumPrecision / std::min(l, nRel);
    mapByClass[qClass] += avgPrecision;
    return avgPrecision;
}

void Effectiveness::startMAPByClass() {
    for (auto const& it : classesSize) {
        mapByClass[it.first] = 0;
    }
}

void Effectiveness::endMAPByClass() {
    for (auto const& it : mapByClass) {
        std::string i = it.first;
        mapByClass[i] = mapByClass[i]/getClassSize(i);
    }
}

std::string Effectiveness::getClass(int x) {
    std::string img = imgList[x];
    auto it = classes.find(img);
    if (it != classes.end()) {
        return it->second;
    }
    std::cerr << "WARNING: Requesting class of an unknown image!" << std::endl;
    return "";
}

int Effectiveness::getClassSize(std::string classname) {
    auto it = classesSize.find(classname);
    if (it != classesSize.end()) {
        return it->second;
    }
    std::cerr << "WARNING: Requesting size of an unknown class!" << std::endl;
    return -1;
}

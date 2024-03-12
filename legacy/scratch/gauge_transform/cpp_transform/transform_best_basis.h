#ifndef BEST_BASIS_H
#define BEST_BASIS_H

#include <vector>
#include <unordered_map>
#include <iostream>

class BestBasis {
public:
    BestBasis(const std::vector<std::vector<int>>& m);

    void findBest();

private:
    void scoreBases(const std::vector<std::vector<int>>& m);
    void generateInvertible(int rank);
    int scoringSum(const std::vector<std::vector<int>>& m2);
    std::vector<std::vector<int>> gaugeTransformXOR(const std::vector<std::vector<int>>& m, const std::vector<std::vector<int>>& g);
    std::vector<std::vector<int>> generateBinaryMatrices(const std::pair<int, int>& shape);
    std::vector<std::vector<int>> filterInvertibleMatrices(const std::vector<std::vector<int>>& matrices);

private:
    std::vector<std::vector<int>> m;
    size_t len;
    std::unordered_map<int, std::vector<std::vector<int>>> scores;
    std::vector<int> bestG; // we make each matrix a 1d vector for convinience
    int bestGScore;
    std::vector<std::vector<int>> allBest;
};

#endif // BEST_BASIS_H

#include "transform_best_basis.h"
#include <cassert>
#include <numeric>

BestBasis::BestBasis(const std::vector<std::vector<int>>& m) : m(m), len(m.size()), bestGScore(0) {}

void BestBasis::findBest() {
    scoreBases(m);
    bestGScore = std::min_element(scores.begin(), scores.end())->first;
    allBest = scores[bestGScore];
    // Assuming allBest is not empty
    bestG = allBest[0];
}




// void BestBasis::scoreBases(const std::vector<std::vector<int>>& m) {
//     //
// }

// void BestBasis::generateInvertible(int rank) {
//     // Implementation of __generate_invertible
// }

// int BestBasis::scoringSum(const std::vector<std::vector<int>>& m2) {
//     // Implementation of __scoring_sum
//     int score = 0;
//     for(std::vector<int>::iterator it = m2.begin(); it != m2.end(); ++it){
//         score += *it;
//     }
//     return score;
// }

// std::vector<std::vector<int>> BestBasis::gaugeTransformXOR(const std::vector<std::vector<int>>& m, const std::vector<std::vector<int>>& g) {
//     // Implementation of __gauge_transform_xor
// }

// std::vector<std::vector<int>> BestBasis::generateBinaryMatrices(const std::pair<int, int>& shape) {
//     // Implementation of __generate_binary_matrices
// }

// std::vector<std::vector<int>> BestBasis::filterInvertibleMatrices(const std::vector<std::vector<int>>& matrices) {
//     // Implementation of __filter_invertible_matrices
//     inv = std::vector<int>;
//     for (std::vector<int> m : matrices){

//     }
// }

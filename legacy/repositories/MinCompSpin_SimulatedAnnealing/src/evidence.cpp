#include "header.h"

map<__uint128_t, unsigned int> build_pdata(vector<pair<__uint128_t, unsigned int>> &data, __uint128_t community) {

    map<__uint128_t, unsigned int> pdata;
    __uint128_t mask_state;

    for (auto const &it : data) {
        mask_state = ((it).first) & community;
        pdata[mask_state] += ((it).second);
    }

    return pdata;
}

double icc_evidence(__uint128_t community, Partition &p_struct){

    double logE = 0;
    double pf;

    map<__uint128_t, unsigned int> pdata = build_pdata(p_struct.data, community);
    unsigned int rank = bit_count(community);
    
    if (rank > 32) {
        pf = -((double) rank - 1.) * p_struct.N * log(2);
    } else {
        double rank_pow = (double) (ONE << (rank - 1));
        pf = lgamma(rank_pow) - lgamma(p_struct.N + rank_pow);
    }

    logE += pf;
    for (auto const &pstate : pdata) {
        logE += lgamma((pstate).second + 0.5) - SQRT_PI;
    }

    return logE;
}

double get_evidence(__uint128_t community, Partition &p_struct) {

    double evidence;

    auto check = p_struct.evidence_memo.find(community);
    if (check != p_struct.evidence_memo.end()) {evidence = check->second;}
    else {
        evidence = icc_evidence(community, p_struct);
        p_struct.evidence_memo[community] = evidence;
        }
    return evidence;
}
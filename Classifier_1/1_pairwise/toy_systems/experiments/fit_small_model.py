import sys
sys.path.append("../../")
from src.pairwise_fitter import Pairwise_fitter


# test if the labels change anythings



if __name__ == "__main__":
    fname = "4spin_conv12"
    mod = Pairwise_fitter(50,"../gen",fname, "../output_small")
    mod.setup(42,input_spaced=True)
    mod.fit("ace","../../ace_utils/ace")
    # mod.fit("qls", "./utils/")
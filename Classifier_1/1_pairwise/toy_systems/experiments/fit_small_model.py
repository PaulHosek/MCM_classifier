import sys
sys.path.append("../../")
from src.pairwise_classify import Pairwise_model


if __name__ == "__main__":
    fname = "4spin"
    mod = Pairwise_model(10,"../gen",fname, "../output_small")
    mod.setup(42,input_spaced=True)
    mod.fit("ace","../../ace_utils/ace")
    # mod.fit("qls", "./utils/")


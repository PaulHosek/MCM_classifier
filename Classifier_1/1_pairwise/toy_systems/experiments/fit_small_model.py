import sys
sys.path.append("../../")
from src.pairwise_classify import Pairwise_model


if __name__ == "__main__":
    fname = "4_spin"
    mod = Pairwise_model(10,"../output_small",fname, "../gen")
    mod.setup(42)
    mod.fit("ace","../../ace_utils/ace")
    # mod.fit("qls", "./utils/")


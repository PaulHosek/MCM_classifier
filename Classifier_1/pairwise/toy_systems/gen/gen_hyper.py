import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt



if __name__ == "__main__":
    scenes = {
    0: ('FN', 'TH'),
    1: ('TH', 'JV'),
    2: ('BM', 'FN', 'JA'),
    3: ('JV', 'JU', 'CH', 'BM'),
    4: ('JU', 'CH', 'BR', 'CN', 'CC', 'JV', 'BM'),
    5: ('TH', 'GP'),
    6: ('GP', 'MP'),
    7: ('MA', 'GP')
}

    H = hnx.Hypergraph(scenes)
    plt.subplots(figsize=(5,5))
    hnx.draw(H)
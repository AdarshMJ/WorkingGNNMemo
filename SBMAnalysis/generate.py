from tqdm import tqdm
import numpy as np
import networkx as nx


def H(*args):
    x = np.array(args)
    return - (x * np.log2(x)).sum()


if __name__ == '__main__':
    degree = 10
    K = degree / 250
    sizes = [250, 250, 250, 250]

    mods = []
    lis = []
    graphs = []

    with tqdm(total=208) as progress_bar:
        for p_0 in np.linspace(0.01, 0.99, 30):
            for p_1 in np.linspace(0.01, 0.99, 30):
                p_2 = (1 - p_0 - p_1) / 2
                if p_2 <= 0:
                    continue

                mod = p_0 - 0.25
                li = 1 - H(p_0, p_1, p_2, p_2) / np.log2(4)

                too_close = False
                for prev_mod, prev_li in zip(mods, lis):
                    if np.sqrt((mod - prev_mod) ** 2 + (li - prev_li) ** 2) < 0.03:
                        too_close = True
                        break

                if too_close:
                    continue

                mods.append(mod)
                lis.append(li)
                graphs.append([])

                probs = np.array([
                    [p_0, p_1, p_2, p_2],
                    [p_1, p_0, p_2, p_2],
                    [p_2, p_2, p_0, p_1],
                    [p_2, p_2, p_1, p_0],
                ]) * K

                for i in range(10):
                    graph = nx.stochastic_block_model(sizes, probs, seed=i, directed=False, selfloops=False,
                                                      sparse=False)
                    graphs[-1].append(graph)

                progress_bar.update()

    for i in range(len(graphs)):
        for j in range(10):
            graph = graphs[i][j]
            nx.write_multiline_adjlist(graph, f'data/SBM/degree_10/{i}_{j}.adjlist')

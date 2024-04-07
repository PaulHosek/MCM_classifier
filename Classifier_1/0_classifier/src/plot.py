import os.path
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import numpy as npu
import matplotlib.colors as mcolors
import scipy.cluster.hierarchy as sch
from src.loaders import load_data
from sklearn.metrics import normalized_mutual_info_score
import scipy.ndimage as ndi


## ---- Co-occurance Matrix ---- ##
def create_cooccurance_matrix(mcm):
    """
    Create a co-occurrence matrix based on the given MCMs.

    :param MCMs: A list or nparray with strings of nr_pixel elements that are either 0 or 1.
    :type MCMs: list or np.ndarray
    :return: The ordered co-occurrence matrix.
    :rtype: np.ndarray
    """
    mcm = np.genfromtxt(mcm, delimiter=1, dtype=int)
    pairs = np.argwhere(mcm == 1).T
    nr_pixels = len(pairs[0])
    matrix = np.zeros((nr_pixels, nr_pixels))

    # fill the groupings into the co-occurrence matrix
    for icc in np.unique(pairs[0]):
        pixels = pairs[1][pairs[0] == icc]
        matrix[np.ix_(pixels, pixels)] = 1
        
    return matrix


def do_cluster(matrix, via_matrix=None):
    """
    Perform hierarchical clustering on a given matrix.
    Can provide via_matrix to base the clustering of "matrix" on the dendrogram of "via_matrix". 

    :param matrix: The input matrix for clustering.
    :type matrix: numpy.ndarray
    :param via_matrix: Optional matrix to base the clustering of "matrix" on. Clustering will be performed on "via_matrix" and applied to "matrix".
    :type return_dendro: numpy.ndarray
    :return: The clustered matrix
    :rtype: numpy.ndarray
    """
    
    if via_matrix is None:
        via_matrix = matrix

    linkage = sch.linkage(via_matrix, method='average')
    dendrogram = sch.dendrogram(linkage, no_plot=True)
    return matrix[:, dendrogram['leaves']][dendrogram['leaves']]




## ----- Partition Map ----- ## 
@np.vectorize
def int_to_letters(i,first_ascii = 65):
    """Converts an integer to a string representation using letters of the alphabet.

    :param i: The integer to be converted.
    :type i: int
    :return: The string representation of the integer using letters of the alphabet.
    :rtype: str
    """
    base26 = ''
    while i >= 0:
        i, idx = divmod(i, 26)
        base26 = chr(idx+first_ascii) + base26
        i -= 1
    return base26

@np.vectorize
def letters_to_int(s):
    """Converts a string representation using letters of the alphabet to an integer.

    :param s: The string to be converted.
    :type s: str
    :return: The integer representation of the string.
    :rtype: int
    """
    base26 = 0
    for i, char in enumerate(reversed(s)):
        base26 += (ord(char) - 65) * (26 ** i)
    return base26



def label_communities(comms, ax, labels, fix_to_cellcenter=True, **kwargs):
    """
    Label largest connected communities with centered text.

    Parameters:
    - comms (numpy.ndarray): An array containing the community assignments for each cell.
    - ax (matplotlib.axes.Axes): The axes object on which the labels will be plotted.
    - labels (list): A list of labels corresponding to each community.
    - fix_to_cellcenter (bool, optional): Whether to fix the label position to the center of the cell. Defaults to True.
    - **kwargs: Additional keyword arguments to be passed to the `ax.text()` function.

    Returns:
    None
    """
    communities = np.unique(comms)
    average_size = np.mean([np.sum(comms == community) for community in communities])

    for community in communities:
        mask = comms == community
        labeled_mask, num_labels = ndi.label(mask)

        sizes = ndi.sum(mask, labeled_mask, range(num_labels + 1)) # size largest connected component
        max_size = sizes.max()

        # only communities with 70% of the cells adjacent and community size larger than average
        if max_size / mask.sum() > 0.7 and mask.sum() > average_size:
            max_mask = labeled_mask == np.argmax(sizes)

            y, x = ndi.center_of_mass(max_mask)

            if fix_to_cellcenter:
                # map to closest cell
                y_indices, x_indices = np.where(max_mask)
                distances = (y_indices - y)**2 + (x_indices - x)**2
                y, x = y_indices[np.argmin(distances)], x_indices[np.argmin(distances)]

            # circle = patches.Circle((x, y), radius=0.5, facecolor='black',alpha=.4, zorder=2)
            # circle = patches.Rectangle((x-.5, y-.5),1,1, facecolor='black',alpha=.4, zorder=2)
            # ax.add_patch(circle)
                
            ax.text(x, y, labels[community], ha='center', va='center', **kwargs)



def compare_mcm_mutual_info_avg_vote(selected, all_P_icc, mcm_comms_map,mcm_idx,drawing_cond=lambda x: x!=0 ):
    """Draw two Parition maps in a row for a single MCM but when seeing different classes.
    Note: 2 maps are currently supported.

    :param selected: list of class labels e.g., [3,5]
    :param all_P_icc: return value of src.plot.calculate_P_icc()
    :param mcm_comms_map: mcm_communities for that mcm
    :param mcm_idx: Label of the mcm e.g., 3. Needs to match the data in all_P_icc. Only for title needed.
    :param drawing_cond: when to draw the values into the cells, defaults to lambdax:x!=0
    :return mutual_information: array of normalized mutual information score.
    """
    nr_comms = np.max(mcm_comms_map)+1
    borders = find_borders(mcm_comms_map)
    fig, axs = plt.subplots(1, len(selected), figsize=(12, 6))
    for idx, cls in enumerate(selected):
        P_mcm_cls = all_P_icc[:,:,cls]

        # Average vs. Individual predicted label
        out = np.zeros(nr_comms)
        for i in range(nr_comms):
            out[i]=normalized_mutual_info_score(np.where(P_mcm_cls.mean(axis=0)>=0.5,1,0),np.where(P_mcm_cls[i,:]>=0.5,1,0))
        mi_matrix = (out[mcm_comms_map]*100).round(1)

        axs[idx].set_title(f"MutInfo*100 for MCM {mcm_idx} ICC vs. average vote for seeing a {cls}")
        partition_map(axs[idx],mi_matrix,mi_matrix,borders,drawing_cond=drawing_cond)    
        axs[idx].text(5,11, f"Average Vote: {P_mcm_cls.mean().round(1)}", ha="center", va="bottom") # note vote not adjusted for number of pixels
        axs[idx].text(5,11.4, f"High = single icc votes predict average votes well. ", ha="center", va="bottom") # note vote not adjusted for number of pixels

    plt.tight_layout()
    plt.show()
    return out

def plot_communities(ax, comms_map, comms_labels = None, title="Communities in MCM ?",cmap=None):
    """Generate Partitionmap of the docstings

    :param ax: matplotlib.axis to use.
    :param comms_map: communities. Output of generate_icc_comms_map
    :param cmap: matplotlib.colormap
    :param title: _description_, defaults to "Communities in MCM ?"
    """
    cmap = create_pastel_cmap(121) if cmap is None else cmap
    borders = find_borders(comms_map)
    ax.set_title(title)
    draw_all_borders(borders,ax=ax)
    if comms_labels is None:
        comms_labels = comms_map
    draw_all_values(comms_labels,ax=ax)        
    ax.imshow(comms_map, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def calc_p_icc_single(data, P_MCM, n_variables,MCM,icc_idx):
    """Get probability distribution of single icc for some set of input images.
    E.g., what probabilities do we get for the first ICC in the MCM for the image of a 5 
    if we show it data of a 3. 
    ==> How good does this ICC differentiate between classes?
    P_MCM: is the probability distribution for the selected MCM
    """
    
    icc_Ps = np.zeros(len(data))
    for k, img in enumerate(data):
        idx = [i for i in range(n_variables) if MCM[icc_idx][i] == "1"]
        sm = int("".join([str(s) for s in img[idx]]), 2)
        icc_Ps[k] = P_MCM[icc_idx][sm] 
    return icc_Ps

def calculate_P_icc(P, all_MCM,mcm_index,n_variables, data_path, data_filename_format):
    """
    Calculate probability distribution for all ICC of one MCM (mcm_index) for all images in all classes.
    


    :param P: Probability distributions for each category. E.g., __P in the classifier.
    :type P: np.array
    :param all_MCM: List of all MCM. E.g., __MCM in the classifier
    :param mcm_index: which MCM to construct the probability distributions for. Legal values are 0-9.
    :param data_path: input data path
    :param data_filename_format: name of the input data files. All files should be named the same except for their category index.
    :type data_filename_format: str
    :return: np.array of shape [iccs, samples, category]
    """

    MCM = all_MCM[mcm_index]
    P_MCM = P[mcm_index]
    nr_icc = len(MCM)
    nr_images = len(load_data(os.path.join(data_path, data_filename_format.format(0))))
    all_P_icc = np.zeros((nr_icc,nr_images,10))

    for icc_idx in range(nr_icc):
        icc_Ps = np.zeros((nr_images,10)) # all samples same size
        for cat_idx in range(10):
            data = load_data(os.path.join(data_path, data_filename_format.format(cat_idx)))
            icc_Ps[:,cat_idx] = calc_p_icc_single(data,P_MCM,n_variables,MCM,icc_idx)
        all_P_icc[icc_idx,:,:] = icc_Ps
    return all_P_icc


def generate_icc_comms_map(single_mcm):
    """Generate pixelwise integer icc labels.
      Returns same shape as mcm/image.

    :param single_mcm: single mcm out of __MCM
    :type single_mcm: np.ndarray
    """
    icc_arr = np.array([list(icc) for icc in single_mcm],dtype=int)
    sorted_data = np.argwhere(icc_arr == 1)
    grouped_data = np.split(sorted_data[:, 1], np.unique(sorted_data[:, 0], return_index=True)[1][1:])
    out = np.empty(121)
    for com_i, comm in enumerate(grouped_data):
        out[comm] = com_i
    return out.reshape((11,11)).astype(int)

## -- Partition Map: Drawing --

def partition_map(ax, colors_vals=None, text_vals=None, borders=None, cmap="coolwarm",drawing_cond=lambda x:True,linewidth=2,cbar=True,normalise=True,global_vbounds=(None,None)):

    if colors_vals is None and text_vals is None: 
        raise ValueError("No color or text values provided. Dimensions of data unknown.")
    if colors_vals is None: 
        colors_vals = text_vals
        cmap = create_white_cmap()

    absmax = np.abs(colors_vals.flat[np.abs(colors_vals).argmax()])
    draw_all_borders(borders, ax=ax,linewidth=linewidth) if borders is not None else None
    draw_all_values(text_vals, color="black", cond=drawing_cond, ax=ax) if text_vals is not None else None
    if normalise:
        im = ax.imshow(colors_vals, cmap=cmap, vmin=-absmax, vmax=absmax)
    elif global_vbounds[0] != None:
        im = ax.imshow(colors_vals, cmap=cmap, vmin=global_vbounds[0],vmax=global_vbounds[1])
    else: 
        im = ax.imshow(colors_vals, cmap=cmap)
    
    if cbar:
        colorbar = plt.colorbar(im, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    return im

def cmap_to_gray(color, reverse=False):
    """
    Create a colormap from the specified color to grey.

    Parameters:
    - color (str): The color to start the colormap from. This can be any color name recognized by Matplotlib.
    - reverse (bool, optional): Whether to reverse the colormap. Defaults to False.

    Returns:
    - A Matplotlib colormap.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list("", [color, "whitesmoke"])
    if reverse:
        cmap = cmap.reversed()
    return cmap

def create_white_cmap():
    """
    Create a white colormap.

    :return: The white colormap.
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    white_cmap = mcolors.LinearSegmentedColormap.from_list("white_cmap", [(1, 1, 1), (1, 1, 1)])
    return white_cmap


def create_pastel_cmap(num_hues, saturation=.4, value=.8):
    """
    Create a pastel colormap.

    :param num_hues: The number of hues in the colormap.
    :type num_hues: int
    :param saturation: The saturation value in the HSV color space.
    :type saturation: float
    :param value: The value (brightness) value in the HSV color space.
    :type value: float
    :return: The pastel colormap.
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    hues = np.linspace(0, 1, num_hues)
    colors = [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]
    pastel_cmap = mcolors.LinearSegmentedColormap.from_list("pastel_cmap", colors)
    return pastel_cmap



 
def draw_all_borders(borders,ax=None, linewidth=2, color="black",offset=(0,0), **kwargs):
    """
    Draw borders for all elements in the border list.

    :param linewidth: The width of the border lines.
    :type linewidth: int
    :param color: The color of the border lines.
    :type color: str
    :param **kwargs: Additional keyword arguments to be passed to plt.Rectangle.
    """
    for i in range(len(borders)):
        for j in range(len(borders[i])):
            for side in borders[i][j]:
                draw_border(j, i, side,ax=ax,linewidth=linewidth,color="black",offset=offset, **kwargs) # TODO Need to do -1/2 linewidth offset the in the direction of the border

def draw_all_values(vals, ax=None, color="white",cond=lambda x: True, **kwargs):
    """
    Draw all values in a grid.

    :param vals: The values to be drawn.
    :type vals: numpy.ndarray
    :param ax: The axes object to draw on, defaults to None.
    :type ax: matplotlib.axes.Axes, optional
    :param cond: Condition when a number should be printed.
    :type cond: Function taking in the value of the cell, returning a bool.
    :param color: The color of the text, defaults to "white".
    :type color: str, optional
    """
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if cond(vals[i,j]):
                ax = ax or plt.gca()
                txt = ax.text(j, i, vals[i, j], ha='center', va='center', color=color, **kwargs)


def draw_border(x, y, side, ax=None, offset = (0,0), **kwargs):
    """
    Draw a border on a specific side of a rectangle (or cell in imshow).

    :param x: The x-coordinate of the rectangle.
    :type x: float
    :param y: The y-coordinate of the rectangle.
    :type y: float
    :param side: The side of the rectangle to draw the border on. Valid values are 't', 'r', 'b', 'l'.
    :type side: str
    :param ax: The matplotlib axes object to draw on, defaults to None.
    :type ax: matplotlib.axes.Axes, optional
    :raises ValueError: If an invalid side value is provided.
    :return: The matplotlib Rectangle object representing the border.
    :rtype: matplotlib.patches.Rectangle
    """
    side_to_offset = {'t': (-.5, -.5, 1, 0), 'r': (.5, -.5, 0, 1), 'b': (-.5, .5, 1, 0), 'l': (-.5, -.5, 0, 1)}

    if side not in side_to_offset:
        raise ValueError("Invalid side value. Valid values are 't', 'r', 'b', 'l'.")
    
    dx, dy, width, height = side_to_offset[side]
    rect = plt.Rectangle((x+dx+offset[0], y+dy+offset[1]), width, height, fill=False, **kwargs)
    
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def find_borders(arr):
    """
    Find the borders of different regions in a 2D array.

    :param arr: The input 2D array.
    :type arr: numpy.ndarray
    :return: A 2D list representing the borders of different regions.
             Each element in the list is a list of border directions ('t', 'b', 'l', 'r').
    :rtype: list[list[str]]
    """
    rows, cols = arr.shape
    borders = [[[] for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if i > 0 and arr[i, j] != arr[i-1, j]:  # Check top
                borders[i][j].append('t')

            if i < rows-1 and arr[i, j] != arr[i+1, j]:  # Check bottom
                borders[i][j].append('b')
            if j > 0 and arr[i, j] != arr[i, j-1]:  # Check left
                borders[i][j].append('l')
            if j < cols-1 and arr[i, j] != arr[i, j+1]:  # Check right
                borders[i][j].append('r')
    return borders

# co-occurance
def interesting_pix_map(mcms_ss, interesting_pix, nr_runs, digit, ax, map_kwargs={},show_letters=True):
    # borders around selected pixels
    b = np.zeros(121)
    b[interesting_pix] = 1
    b = b.reshape((11,11))
    b = find_borders(b)

    icc_loc = np.zeros((11,11))
    icc_sum = np.zeros((11,11))

    for pixel_idx in interesting_pix:
        row_i = pixel_idx // 11
        col_i = pixel_idx % 11
        for mcms in mcms_ss:
            comm = generate_icc_comms_map(mcms[digit])
            icc_sum += np.where(comm==comm[row_i,col_i], 1,0)
        icc_loc += np.where(icc_sum>0,1,0).astype(int)

        

    letters = int_to_letters(icc_loc.astype(int),first_ascii = 64)
    letters[letters == "@"] = "."
    if show_letters:
        partition_map(ax, icc_sum/nr_runs,letters, b,cmap="viridis", normalise=False, **map_kwargs)
    else:         
        partition_map(ax, icc_sum/nr_runs,None, b,cmap="viridis", normalise=False, **map_kwargs)










































































# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
#                        CLASSIFIER EVAL CODE PEPIJN
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_confusion_matrix(confusion_matrix, n_categories: int, title="Confusion matrix", cmap="Blues", logScale: bool = False):
    """
    This function prints and plots the confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        n_categories (int): Number of categories
        title (str, optional): Title of the plot. Defaults to "Confusion matrix".
        cmap (str, optional): Color map. Defaults to "Blues".
    """
    if logScale:
        plt.matshow(confusion_matrix, interpolation="nearest", cmap=cmap, norm=mcolors.LogNorm(vmin=1, vmax=confusion_matrix.max()))
    else:
        plt.matshow(confusion_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_categories)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    
def plot_label_prob_diff(label1, label2, test_labels, probs, predicted_classes, title="Label probability difference"):
    correctly_classified_as_label1 = []
    correctly_classified_as_label2 = []
    incorrectly_classified_as_label1 = []
    incorrectly_classified_as_label2 = []
    for i in range(len(probs)):
        # Correctly classified as category 'label1'
        if test_labels[i] == label1 and predicted_classes[i] == label1:
            correctly_classified_as_label1.append(probs[i])
        # Correctly classified as category 'label2'
        if test_labels[i] == label2 and predicted_classes[i] == label2:
            correctly_classified_as_label2.append(probs[i])
        # Incorrectly classified as category 'label1'
        if test_labels[i] == label2 and predicted_classes[i] == label1:
            incorrectly_classified_as_label1.append(probs[i])
        # Incorrectly classified as category 'label2'
        if test_labels[i] == label1 and predicted_classes[i] == label2:
            incorrectly_classified_as_label2.append(probs[i])
            
    # Plot probabilities and correctness for categories 0 and 1
    plt.figure()
    # Correctly classified as category 3
    plt.scatter(
        np.array(correctly_classified_as_label1)[:, 3],
        np.array(correctly_classified_as_label1)[:, 5],
        color="green",
        marker="o", # type: ignore
        alpha=0.5,
        label=f"Correctly classified as {label1}",
    )
    # Correctly classified as category 5
    plt.scatter(
        np.array(correctly_classified_as_label2)[:, 3],
        np.array(correctly_classified_as_label2)[:, 5],
        color="green",
        marker="^", # type: ignore
        alpha=0.5,
        label=f"Correctly classified as {label2}",
    )
    # Incorrectly classified as category 3
    plt.scatter(
        np.array(incorrectly_classified_as_label1)[:, 3],
        np.array(incorrectly_classified_as_label1)[:, 5],
        color="red",
        marker="o", # type: ignore
        alpha=0.5,
        label=f"Incorrectly classified as {label1}",
    )
    # Incorrectly classified as category 5
    plt.scatter(
        np.array(incorrectly_classified_as_label2)[:, 3],
        np.array(incorrectly_classified_as_label2)[:, 5],
        color="red",
        marker="^", # type: ignore
        alpha=0.5,
        label=f"Incorrectly classified as {label2}",
    )
    plt.plot([0, 1], [0, 1], color="black", label="Perfect classifier")
    plt.plot([1, 0], [1, 0], color="black")
    plt.title(title)
    plt.xlabel(f"Probability of category {label1}")
    plt.ylabel(f"Probability of category {label2}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_results(test_data, test_labels, predicted_classes, probs, classifier, output_path,n_categories):
    # Count amount of -1 labels
    n_no_labels = 0
    no_labels_labels = []
    images = []
    for i in range(len(test_labels)):
        if predicted_classes[i] == -1:
            n_no_labels += 1
            no_labels_labels.append(test_labels[i])
            images.append(test_data[i])
    if n_no_labels != 0:
        # Plot all images with no labels so that they are in the same figure in a square grid
        dim_1 = int(np.sqrt(n_no_labels))
        dim_2 = int(np.sqrt(n_no_labels))
        fig, axs = plt.subplots(dim_1, dim_2, figsize=(10, 10))
        for i in range(dim_1):
            for j in range(dim_2):
                axs[i, j].imshow(images[i * dim_2 + j].reshape(11, 11), cmap="gray")
                axs[i, j].set_title(no_labels_labels[i * dim_2 + j])
                axs[i, j].axis("off")

        plt.savefig(os.path.join(output_path, "test.png"))

    print("Number of datapoints for which the classifier didn't have any probability for any category: {}".format(
        n_no_labels))
    print("Labels of these datapoints: {}".format(no_labels_labels))

    # Find amount of datapoints with 2 or more categories with probability > 0
    n_multiple_probs = 0
    multiple_probs_labels = []
    for i in range(len(probs)):
        if np.sum(probs[i] > 0) > 1:
            n_multiple_probs += 1
            multiple_probs_labels.append(test_labels[i])
    print("Number of datapoints with 2 or more categories with probability > 0: {}".format(n_multiple_probs))

    plot_label_prob_diff(3, 5, test_labels, probs, predicted_classes)
    plt.savefig(os.path.join(output_path, "probs_and_correctness.png"))

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(classifier.stats["confusion_matrix"], n_categories)
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))

# ----- INDICATIVE ICC -----
def mirror_hist(ax, series1, series2, kwargs_up = {}, kwargs_down = {}, n_bins = 20, color="blue", label=None):
    # update default args
    kw_up = {"color":color,"alpha":.8}
    kw_up.update(kwargs_up)
    kw_down = {"color":color,"alpha":.5, "label":label}
    kw_down.update(kwargs_down)

    heights, bins = np.histogram(series1*-1, weights=np.ones(len(series1)) / len(series1), bins=n_bins) 
    bin_width = np.diff(bins)[0]
    bin_pos =( bins[:-1] + bin_width / 2) * -1
    ax.bar(bin_pos, heights, width=bin_width, **kw_up)
    ax.bar( bin_pos, heights, width=bin_width, color='none', edgecolor='black')


    # upside down plot
    heights, bins = np.histogram(series2*-1, weights=np.ones(len(series2)) / len(series2), bins=n_bins) 
    heights *= -1
    bin_width = np.diff(bins)[0]
    bin_pos =( bins[:-1] + bin_width / 2) * -1
    ax.bar(bin_pos, heights, width=bin_width, **kw_down)
    ax.bar( bin_pos, heights, width=bin_width, color='none', edgecolor='black')

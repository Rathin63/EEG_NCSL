#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-29T17:29:10.865Z
"""

import numpy as np
import dsp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import os
import pandas as pd
from scipy import stats
from fastcomputeA import fastcomputeA
from matplotlib.patches import Polygon
import networkx as nx
import cv2
from sklearn.cluster import KMeans
from matplotlib import colormaps
import edfio
#import precompute_windows as pcw



# # Finding Files


mainfolder = "C:\\Users\\Ultimateo\\OneDrive\\Desktop\\ChronicPainPractice\\"

mask = 3

def find_files_by_extension(root_folder, extension):
    matching_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(extension):
                matching_files.append(os.path.join(root, file))
    return matching_files


fileList = find_files_by_extension((mainfolder + "Data\\eeg"),'closed_eeg.vhdr')
for i in range(len(fileList)):
    print(i, fileList[i])

def find_dirs_by_extension(root_folder, extension):
    matching_dirs = []
    for root, subdirs, dirs in os.walk(root_folder):
        print(root)
        print(subdirs)
        print(dirs)
        for name in subdirs:
            if name.endswith(extension):
                matching_dirs.append(os.path.join(root, name))
                # print(matching_dirs)
    return matching_dirs



# # Preprocessing


keep_channels = pd.read_table(mainfolder + "\\Data\\eeg\\sub-CBPpa02\\eeg\\sub-CBPpa02_task-closed_channels.tsv")
keep_channels = keep_channels.to_numpy()

# print(keep_channels.shape)
# print(keep_channels[0,0])

keep_channels = keep_channels[:,0]

keep_channels = np.delete(keep_channels, (30,31,64,65))
# keep_channels = np.delete(keep_channels, (16,17,20,21,30,31,64,65))

# keep_channels = np.delete(keep_channels, (20,21,30,31,64,65))
# keep_channels = np.delete(keep_channels, (20,21,30,31,40,41,64,65))



keep_channels = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "P3",
    "P4",
    "O1",
    "O2",
    "Fz",
    "Cz",
    "Pz",
    "C3",
    "C4"
]



keep_channels = list(keep_channels)

chl = len(keep_channels)

print(chl)

# cleanornoisy = "noisy"
# cleanornoisy = "clean"

def preprocess_eeg(raweeg):
    # Select 10-20 Channels and set average reference
    #eeg.rename_channels(rename_channels)  #- comment out if the EEG has channels T7, T8, P7, or P8
    ##mine has those channels

    raweeg.pick(keep_channels)
    raweeg.load_data()
    # raweeg.resample(650)
    # raweeg.filter(1, h_freq=65)
    # raweeg.filter(1, h_freq=40)
    # raweeg.filter(1, h_freq=99)
    raweeg.filter(l_freq=1, h_freq=30)
    # raweeg.filter(1, None)

    # raweeg.filter(0, None)
    # raweeg.l_freq=1

    # raweeg.filter(1, h_freq=20)

    ##making the h freq 40 worked really well for some reason

    # raweeg.set_eeg_reference('average')    #- comment out if your EEG doesn't have reference channels

    # ##not sure if I do
    # ##I found it-I do have a reference
    # ##says all are referenced to electrode FCz
    # ##however, FCz is the only one that shows on the electrodes tsv file but not the channels file
    # ##i guess maybe it's already referenced to it and I'm not supposed to replace average

    raweeg.set_montage('standard_1020')

    # Apply ICA and reject non brain components
    # Do ICA
    ica = ICA(
        n_components=len(keep_channels),
        max_iter=1000,
        method="infomax",
        random_state=29,
        fit_params=dict(extended=True),
    )
    ica.fit(raweeg)

    print(ica.n_iter_)

    # Auto label ICA components and reject
    ic_labels = label_components(raweeg, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    ica.apply(raweeg, exclude=exclude_idx)

    # raweeg.filter(1, 40)
    end = np.floor(raweeg.times[-1]) - 0.005
    raweeg.crop(0, end)

    # raweeg.info['bads'].extend(['C3', 'C4', 'Cz', 'Pz', 'Fz'])
    # raweeg = raweeg.drop_channels(eeg.info['bads'])

    return raweeg

# Load EEG and preprocess
#original was vhdr, code below is for edf/set
##mine is vhdr

# def load(i):
#     eeg_filename = i
#     print(eeg_filename)
#     #might need to re run 20 and 325
#     eeg = mne.io.read_raw_brainvision(eeg_filename)
#     eeg_preprocessed = preprocess_eeg(eeg)
#     return


#eeg_preprocessed = eeg



# # Feature Extraction
# Extract A matrices, source sink, and powers for each window


# A matrix parameters in seconds
# window_length = 0.019
# window_length = 0.125
# window_length = 0.075
# window_length = 0.135
# window_length = 0.45
# window_length = 0.300
window_length = 0.045
# window_length = 0.25

# window_length = 0.25
# window_length = 0.5

window_advance = window_length
# alpha = 1e-10
alpha = 1e-12
# alpha = 1.5e-18
# alpha = 1e-9

# alpha = 1e-20
# alpha = 0

# Initialize feature lists
ahats = []
sink_indices = []
source_indices = []
source_influences = []
sink_connectivities = []
ssis = []
powers = []
evals = []
ahatswdiag = []

# Loop and get all features
def features():
    eeg_preprocessed.load_data()
    fs = eeg.info['sfreq']
    window_length_samples = int(fs * window_length)
    window_advance_samples = int(fs * window_advance)

    # Loop over the data
    window_start = 0
    data = eeg.get_data()
    nsamples = data.shape[1]
    while window_start <= (nsamples - window_length_samples):

        # Advance window
        window_start += window_advance_samples

        # Standardize the data
        X = data[:, window_start:window_start + window_length_samples]
        mean_vec = np.mean(X, axis=1)
        X_mean_subtract = X - mean_vec[:, np.newaxis]
        std = np.std(X_mean_subtract, axis=1)
        X_standardized = X / std[:, np.newaxis]

        # Store A hats
        ##there is something wrong with these

        ahat = dsp.computeA(X, alpha)
        ahats.append(ahat)

        # Calculate source sink features and store
        sink_idx, source_idx, source_influence, sink_connectivity, ssi = dsp.compute_source_sink_index(ahat)
        sink_indices.append(sink_idx)
        source_indices.append(source_idx)
        source_influences.append(source_influence)
        sink_connectivities.append(sink_connectivity)
        ssis.append(ssi)

        # Calculate power features and store
        powers.append(np.sum(X ** 2, axis=1))

        # Calculate evals
        vals, _ = np.linalg.eig(ahat)
        evals.append(vals)

    return

# Initialize feature lists
def featurelists():
    ahats = np.asarray(ahats)
    sink_indices = np.asarray(sink_indices)
    source_indices = np.asarray(source_indices)
    source_influences = np.asarray(source_influences)
    sink_connectivities = np.asarray(sink_connectivities)
    ssis = np.asarray(ssis)
    powers = np.asarray(powers)
    evals = np.asarray(evals)

    return

def computeA(xdata):
    """Compute the A transition matrix from a vector timeseries
    """
    ## this was broken because the linalg.inv function does not work

    nchns, T = xdata.shape

    Z = xdata[:, 0:T-1]
    #add row of 1s to Z, (n+1, t-1)
    Y = xdata[:, 1:T]

    Z2 = Z @ Z.transpose() + alpha*np.eye(chl)
    #add collumn of 1s to

    # Z2 = Z[0:10,0:10]
    #ok so if i use 10 channels it should work
    ## this is not working

    Z2 = np.asarray(Z2)

    D = np.linalg.pinv(Z2)

    # D = np.linalg.inv(Z2)

    D2 = Z.transpose() @ D

    Ahat = Y @ D2
    #first n collumns is Ahat last collumn is bhat

    # Z2 = np.linalg.pinv(Z)

    # Ahat = Y @ Z2

    ## this is definitely broken
    ##IDK why this doesnt work anymore, my only guess is that there's some overflow or floating point imprecision stuff
    ##YEP, if i decrease the size of the array the inverse of it works
    ##still causing overflows
    ##i fixed it by using only some of the channels

    return Ahat

# print(computeA(X))

def stabilize_matrix(A, eigval):
    """
    Stabilize the matrix, A based on its eigenvalues.
    Assumes discrete-time linear system.
    Parameters
    ----------
    A : np.ndarray
         CxC matrix
    eigval : float
        the maximum eigenvalue to shift all large evals to
    Returns
        -------
        A : np.ndarray
            the stabilized matrix w/ eigenvalues <= eigval
        """
    # check if there are unstable eigenvalues for this matrix
    if np.sum(np.abs(np.linalg.eigvals(A)) > eigval) > 0:
        # perform eigenvalue decomposition
        eigA, V = np.linalg.eig(A)

        # get magnitude of these eigenvalues
        abs_eigA = np.abs(eigA)

        # get the indcies of the unstable eigenvalues
        unstable_inds = np.where(abs_eigA > eigval)[0]

        # move all unstable evals to magnitude eigval
        for ind in unstable_inds:
            # extract the unstable eval and magnitude
            unstable_eig = eigA[ind]
            this_abs_eig = abs_eigA[ind]

            # compute scaling factor and rescale eval
            k = eigval / this_abs_eig
            eigA[ind] = k * unstable_eig

            # recompute A
        eigA_mat = np.diag(eigA)
        Aprime = np.dot(V, eigA_mat)
        A = np.linalg.lstsq(V.T, Aprime.T)[0].T
    return A
# print(computeA(X))
# print(stabilize_matrix(computeA(X),1).real)

# # Plots


def cluster(energy, clusters):
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(energy)

    changes=np.zeros((4,4), dtype=int)
    points=np.zeros((4,), dtype=int)
    for i in range(len(kmeans.labels_)-1):
        x=kmeans.labels_[i]
        y=kmeans.labels_[i+1]
        changes[x,y]+=1
        points[x]+=1
    points[y]+=1

    normalvarchange = np.var(changes.flatten())/energy.shape[1]
    normalvarpoint = np.var(points)/energy.shape[1]

    return normalvarchange, normalvarpoint

def naming():
    #Naming Figures
    titleString = eeg_filename.split('\\')[10]
    titleString1 = titleString.split('_')[0]

    channelHeader = ",".join(keep_channels)

    #Placing Data into Directory
    directory_path = (mainfolder + "directory\\") + titleString1 + "\\" + titleString
    os.makedirs(directory_path, exist_ok=True)
    return

def boxplot():
    # Box plot of all powers for each window
    fig, ax = plt.subplots(1, 1, figsize=(chl, 20), squeeze=False)
    ax[0,0].boxplot(powers, showfliers=False)

    # plt.boxplot(powers)
    ax[0,0].set_title("Power by Channel")
    ax[0,0].set_xticks(np.arange(1, chl+1), keep_channels)
    ##had to change number of channels
    plt.savefig(os.path.join(directory_path, (titleString + "_power_by_channel.png")))
    np.savetxt(os.path.join(directory_path, (titleString + "_power_by_channel.csv")), powers, delimiter=',', header = channelHeader)

    plt.cla()
    plt.clf()
    return

##boxplot effects the histogram
def histogram():
    # Histogram of A matrix values
    plt.hist(ahats.flatten(), bins=chl)
    plt.title("A Matrix")
    plt.xlabel("A Matrix Values")
    plt.savefig(os.path.join(directory_path, (titleString + "_A_matrix_values.png")))
    np.save(os.path.join(directory_path, (titleString + "_A_matrix_values.npy")), ahats)
    np.save(os.path.join(directory_path, (titleString + "diag_matrix_values.npy")), ahatswdiag)


    plt.cla()
    plt.clf()
    return

def eigen():
    # Plot eigenvalues on the complex plane
    plt.figure(figsize=(6, 6))

    all_evals = evals.flatten()
    # extract real part
    x = [ele.real for ele in all_evals]
    # extract imaginary part
    y = [ele.imag for ele in all_evals]
    plt.scatter(x, y)




    # Unit circle
    x = np.sin(np.arange(0, 2*np.pi, 0.01))
    y = np.cos(np.arange(0, 2*np.pi, 0.01))
    plt.plot(x, y, color='k', linestyle='dashed')

    plt.title("Eigenvalues")
    plt.savefig(os.path.join(directory_path, (titleString + "_eigenvalues.png")))
    np.savetxt(os.path.join(directory_path, (titleString + "_eigenvalues.csv")), evals, delimiter=',', header = channelHeader)

    plt.cla()
    plt.clf()

    reals = []
    imags = []

    for idx in evals:
        real = [ele.real for ele in idx]
        imag = [ele.imag for ele in idx]
        reals.append(real)
        imags.append(imag)

    reals = np.asarray(reals)
    imags = np.asarray(imags)


    np.savetxt(os.path.join(directory_path, (titleString + "_eigen_real.csv")), reals, delimiter=',', header = channelHeader)
    np.savetxt(os.path.join(directory_path, (titleString + "_eigen_imag.csv")), imags, delimiter=',', header = channelHeader)


    return

def sink():
    # Histogram of sink values
    plt.hist(sink_indices.flatten(), bins=chl)
    plt.xlabel("Sink Indices")
    plt.savefig(os.path.join(directory_path, (titleString + "_sink_indices.png")))

    plt.cla()
    plt.clf()
    return

def scatter():
    # Scatterplot of source sink values
    plt.figure(figsize=(6, 6))
    plt.scatter(sink_indices.flatten(), source_indices.flatten())
    plt.xlabel("Sink Index")
    plt.ylabel("Source Index")
    plt.savefig(os.path.join(directory_path,(titleString + "_source_sink_index.png")))

    #save source and sink indicies
    np.savetxt(os.path.join(directory_path, (titleString + "_source_indicies.csv")), source_indices, delimiter=',', header = channelHeader)
    np.savetxt(os.path.join(directory_path, (titleString + "_sink_indices.csv")), sink_indices, delimiter=',', header = channelHeader)

    plt.cla()
    plt.clf()
    return


def reconstruct():
    # adjustedmatrix = ahatswdiag*(1.233213508)
    aproxeeg = []
    aproxdata = data[:,0]


    ##this is horrible and slow but just testing

    aproxeeg.append(aproxdata)

    i = 0
    while i < ahats.shape[0]:
        # while i < 200:
        # aproxpower = np.matmul(adjustedmatrix[i], aproxpower)
        l = 1
        while l <= window_length_samples:
            aproxdata = ahatswdiag[i] @ aproxdata
            aproxeeg.append(aproxdata)
            l += 1
        i += 1
        aproxdata = data[:,(i*window_length_samples)]

    aproxeeg = np.asarray(aproxeeg)


    # print(aproxeeg[1,0])
    # print(powers[1,0])


    start = 0
    end = aproxeeg.shape[0]-1

    # start = 90000
    # end = 95000

    # start = 15000
    # end = 250
    # end = 16000
    # end = 20000

    view = 0
    # plt.ylim(-4e-5, 4e-5)


    # plt.plot(range(end+1)[start:end], data.transpose()[start:end,view])
    plt.plot(range(end+1)[start:end], aproxeeg[start:end,view])
    plt.plot(range(end+1)[start:end], data.transpose()[start:end,view])
    plt.savefig(os.path.join(directory_path, (titleString + "Data_vs_Reconstruction.png")))


    plt.cla()
    plt.clf()

    i = window_length_samples
    pcor = []
    while i < aproxeeg.shape[0]:
        s = stats.linregress(aproxeeg[i-window_length_samples:i,view], data.transpose()[i-window_length_samples:i,view])[2]
        pcor.append(s**2)
        i+=window_length_samples

    plt.plot(range(len(pcor)), pcor)
    plt.savefig(os.path.join(directory_path, (titleString + "r_over_time.png")))

    print(np.mean(pcor))

    plt.cla()
    plt.clf()

    return

# reconstruct()

norm = colors.LogNorm()
viridis = colormaps['viridis']
cmap = colors.LinearSegmentedColormap.from_list("normsvd", viridis.colors)

def plot_examples(colormaps, data):
    """
    Helper function to plot data with associated colormap.
    """
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 * 4 + 2, 3 * 2),
                            layout='constrained', squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=0, vmax=1)
        fig.colorbar(psm, ax=ax)

    # plt.title("correlation between channels")
    # plt.show()
    fig.savefig(os.path.join(directory_path, (titleString + "colormap_svds.svg")))
    np.savetxt(os.path.join(directory_path, (titleString + "SVD.csv")), SVDS, delimiter=',', header = channelHeader)
    # plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs","Pearsonbetweenchannels" + ".png"))
    plt.cla()
    plt.clf()
    return

def correlation_matrix(colormaps):
    cormat=np.zeros((chl,chl),dtype=float)
    for i in range(chl):
        for j in range(chl):
            s = stats.linregress(data[i], data[j])[2]
            cormat[i,j] = s
    # oldcormat = np.load(os.path.join(mainfolder + "Archive\\Old_Graphs\\PresentationDrawings","cormat" + ".npy"))
    # cormat=oldcormat-cormat

    """
    Helper function to plot data with associated colormap.
    """
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 * 4 + 2, 3 * 2),
                            layout='constrained', squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(cormat, cmap=cmap, rasterized=True, vmin=0, vmax=1)
        fig.colorbar(psm, ax=ax)

    plt.title("Correlation Between Channels (No BPF)")
    plt.yticks(range(chl), keep_channels)

    plt.xticks(range(chl), keep_channels)

    plt.show()
    # plt.savefig(os.path.join(directory_path, (titleString + "colormap_svds.svd")))
    # np.savetxt(os.path.join(directory_path, (titleString + "SVD.csv")), SVDS, delimiter=',', header = channelHeader)
    fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\PresentationDrawings","Pearsonbetweenchannelsnobpf" + ".svg"))
    # np.save(os.path.join(mainfolder + "Archive\\Old_Graphs\\PresentationDrawings","cormat" + ".npy"), cormat)

    # print(cormat)
    # plt.cla()
    # plt.clf()
    return

def r2andp():
    # adjustedmatrix = ahatswdiag*(1.233213508)
    aproxeeg = []
    aproxdata = data[:,0]


    ##this is horrible and slow but just testing

    aproxeeg.append(aproxdata)

    i = 0
    while i < ahatswdiag.shape[0]:
        # while i < 200:
        # aproxpower = np.matmul(adjustedmatrix[i], aproxpower)
        l = 1
        while l <= window_length_samples:
            aproxdata = ahatswdiag[i] @ aproxdata
            aproxeeg.append(aproxdata)
            l += 1
        i += 1
        aproxdata = data[:,(i*window_length_samples)]

    aproxeeg = np.asarray(aproxeeg)


    # print(aproxeeg[1,0])
    # print(powers[1,0])


    start = 0
    end = aproxeeg.shape[0]-1
    view = 0

    # start = 90000
    # end = 95000

    # start = 15000
    # end = 250
    # end = 16000
    # end = 20000

    rlist = []
    plist = []

    while view < chl:
        r,p = stats.linregress(aproxeeg[start:end,view], data.transpose()[start:end,view])[2:4]
        rlist.append(r**2)
        plist.append(p)
        view+=1

    np.savetxt(os.path.join(directory_path, (titleString + "_r^2.csv")), rlist, delimiter=',', header = channelHeader)
    np.savetxt(os.path.join(directory_path, (titleString + "_p.csv")), plist, delimiter=',', header = channelHeader)
    return

def gfpmax():

    magnitude = []

    for i in data.transpose():
        magnitude.append(np.std(i))

    maximums = []



    for i in range(1, len(magnitude)-1):
        if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1] and int(np.trunc(i/125)) < sink_indices.shape[0]:
            maximums.append(i)

    maxdata = []
    maxsink = []

    for i in maximums:
        maxdata.append(data.transpose()[i])
        maxsink.append(sink_indices[int(np.trunc(i/125))])


    maxdata = np.asarray(maxdata)
    maxsink = np.asarray(maxsink)
    maximums = np.asarray(maximums)
    np.save(os.path.join(directory_path, (titleString + "_gfpmaximums.csv")), maximums) #shouldn't be .csv
    return maxdata, maxsink

# fake = mne.create_info(keep_channels, 1000, 'eeg')
# datar = mne.io.RawArray(data, fake)

# datar.set_montage('standard_1020')

# datar.info
# mne.channels.get_builtin_montages()

# fake = mne.channels.make_standard_montage('standard_1020', head_size='auto')
# fake.pick(keep_channels)
# # fake.Info[dig]



# montadata = mne.channels.DigMontage(


# mne.viz.plot_montage(

# sinktopomap()

def sinktopomap():
    # xypos = np.asarray(pd.read_table(mainfolder + 'Data/eeg/' + titleString1 + '/eeg/' + titleString1 + '_electrodes.tsv'))
    # dels = []
    # for i in range(xypos.shape[0]):
    #     if xypos[i,0] not in keep_channels:
    #         dels.append(i)
    # xypos = np.delete(xypos, np.asarray(dels), axis = 0)
    # xypos = xypos[:,1:3]
    # xypos = xypos.astype('float64')

    # mne.set_montage('standard_1020')

    # mean_values = np.mean(sink_indices, axis=0)  # Mean over time for single recording
    # maskkeep = np.zeros(2*mask)
    # maskkeep[0:mask]=(np.argpartition(mean_values, -mask)[-mask:])
    # maskkeep[mask:2*mask]=(np.argpartition(mean_values, -mask)[:mask])
    # maskkeep=maskkeep.astype(int)
    # print(maskkeep)
    # channel_names=[]
    # for n in maskkeep:
    #     channel_names.append(keep_channels[int(n)])
    # mean_values=mean_values[maskkeep]

    channel_names = keep_channels

    pos = np.array([montage.get_positions()["ch_pos"][ch][:2] for ch in channel_names])

    fig, ax = plt.subplots()
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(sink_indices[1726], xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(np.mean(maxsink, axis = 0), pos = xypos,  size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
    mne.viz.plot_topomap(mean_values, pos,  size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = channel_names)



    # head outline not working
    fig.savefig(os.path.join(directory_path, (titleString + "gfpsinktopomap.svg")))
    # fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "gfpmax_energytopomapunhealthyexample" + ".svg"))
    return

def datatopomap():
    # xypos = np.asarray(pd.read_table(mainfolder + 'Data/eeg/' + titleString1 + '/eeg/' + titleString1 + '_electrodes.tsv'))
    # dels = []
    # for i in range(xypos.shape[0]):
    #     if xypos[i,0] not in keep_channels:
    #         dels.append(i)
    # xypos = np.delete(xypos, np.asarray(dels), axis = 0)
    # xypos = xypos[:,1:3]
    # xypos = xypos.astype('float64')
    # mean_values = np.mean(data, axis=1)  # Mean over time for single recording
    # maskkeep = np.zeros(2*mask)
    # maskkeep[0:mask]=(np.argpartition(mean_values, -mask)[-mask:])
    # maskkeep[mask:2*mask]=(np.argpartition(mean_values, -mask)[:mask])
    # maskkeep=maskkeep.astype(int)
    # print(maskkeep)
    # channel_names=[]
    # for n in maskkeep:
    #     channel_names.append(keep_channels[int(n)])
    # mean_values=mean_values[maskkeep]

    channel_names = keep_channels

    pos = np.array([montage.get_positions()["ch_pos"][ch][:2] for ch in channel_names])

    fig, ax = plt.subplots()
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(sink_indices[1726], xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(np.mean(maxdata, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)

    mne.viz.plot_topomap(mean_values, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)



    # head outline not working
    fig.savefig(os.path.join(directory_path, (titleString + "gfpdatatopomap.svg")))
    # fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "gfpmax_energytopomapunhealthyexample" + ".svg"))
    return

def energytopomap():
    # xypos = np.asarray(pd.read_table(mainfolder + 'Data/eeg/' + titleString1 + '/eeg/' + titleString1 + '_electrodes.tsv'))
    # dels = []
    # for i in range(xypos.shape[0]):
    #     if xypos[i,0] not in keep_channels:
    #         dels.append(i)
    # xypos = np.delete(xypos, np.asarray(dels), axis = 0)
    # xypos = xypos[:,1:3]
    # xypos = xypos.astype('float64')
    mean_values = np.mean(powers, axis=0)  # Mean over time for single recording

    # maskkeep = np.zeros(2*mask)
    # maskkeep[0:mask]=(np.argpartition(mean_values, -mask)[-mask:])
    # maskkeep[mask:2*mask]=(np.argpartition(mean_values, -mask)[:mask])
    # maskkeep=maskkeep.astype(int)
    # print(maskkeep)
    # channel_names=[]
    # for n in maskkeep:
    #     channel_names.append(keep_channels[int(n)])
    # mean_values=mean_values[maskkeep]

    channel_names = keep_channels

    pos = np.array([montage.get_positions()["ch_pos"][ch][:2] for ch in channel_names])

    fig, ax = plt.subplots()
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(sink_indices[1726], xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(np.mean(maxdata, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)

    mne.viz.plot_topomap(mean_values, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)



    # head outline not working
    fig.savefig(os.path.join(directory_path, (titleString + "energytopomap.svg")))
    # fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\topomapvideo", "topomap" + str(window_start) + ".png"))

    plt.cla()
    plt.clf()
    return

def savedata():
    np.save(os.path.join(directory_path, (titleString + "data.npy")), data)
    return

# # Directory


print(chl)

print(xypos)

##locating information

if chl == 62:
    rootdir = (mainfolder + "directory")
if chl == 19:
    rootdir = (mainfolder + "10-20_directory")
#eyestate = '_task-open_eeg.vhdr'
fake = mne.create_info(keep_channels, 1000, 'eeg')
datar = mne.io.RawArray(np.zeros((chl,2)), fake)
datar.set_montage('standard_1020')
xypos = datar.info
montage = mne.channels.make_standard_montage("standard_1020")


# window_length = 0.15
window_length = 0.015


# window_length = 0.25
# window_length = 0.5

# window_advance = window_length
# preprocess then process

# for file in fileList:
file = fileList[60]
if file == file:
    window_length = 0.015
    while window_length <= 0.13:
        window_length+=0.01
        window_advance = window_length
        # for file in (fileList[0], fileList[30]):
        # , fileList[60], fileList[90], fileList[120], fileList[150], fileList[180]):

        # for file in (fileList[1], fileList[2]):
        ahats = []
        sink_indices = []
        source_indices = []
        source_influences = []
        sink_connectivities = []
        ssis = []
        powers = []
        evals = []
        ahatswdiag = []
        SVDS = []

        ## I made a change here because I was confused about the raw eeg being called eeg and the eeg object from eeg_preprocessed.load_data()
        ##it doesnt do anything different so I wont change it back
        #load(file)
        eeg_filename = file
        print(eeg_filename)
        #might need to re run 20 and 325
        eeg = mne.io.read_raw_brainvision(eeg_filename)
        eeg = preprocess_eeg(eeg)
        ## I think my data may already be preprocessed

        print(eeg.info['chs'])
        #features()
        eeg.load_data()
        fs = eeg.info['sfreq']

        window_length_samples = int(fs * window_length)
        window_advance_samples = int(fs * window_advance)

        # Loop over the data
        # window_start = 100*window_advance_samples
        window_start = 0
        data = eeg.get_data()
        nsamples = data.shape[1]


        # this used to say <= which broke if it was =
        while window_start < (nsamples - window_length_samples):

            # Standardize the data
            X = data[:, window_start:window_start + window_length_samples]
            mean_vec = np.mean(X, axis=1)
            X_mean_subtract = X - mean_vec[:, np.newaxis]
            std = np.std(X_mean_subtract, axis=1)
            X_standardized = X / std[:, np.newaxis]

            ahat = computeA(X)

            # ahat = stabilize_matrix(ahat, 1.05).real


            blankArray = np.zeros((chl, chl - 1))

            #remove diagonal from ahat
            for idx, ele in enumerate(ahat):
                newRow = np.delete(ele, idx)
                blankArray[idx] = newRow

            ahats.append(blankArray)
            ahatswdiag.append(ahat)

            #calculate SVD
            # S = np.linalg.svd(X, full_matrices=True, compute_uv=False, hermitian=False)
            # SVDS.append(S)


            # Calculate source sink features and store
            # sink_idx, source_idx, source_influence, sink_connectivity, ssi = dsp.compute_source_sink_index(ahat)
            # sink_indices.append(sink_idx)
            # source_indices.append(source_idx)
            # source_influences.append(source_influence)
            # sink_connectivities.append(sink_connectivity)
            # ssis.append(ssi)

            # Calculate power features and store
            # powers.append(np.sum(X ** 2, axis=1))
            ## doesnt this just increase for a bigger x?

            # Calculate evals
            # vals, _ = np.linalg.eig(ahat)
            # evals.append(vals)

            #increment window
            window_start += window_advance_samples



        ahats = np.asarray(ahats)
        # sink_indices = np.asarray(sink_indices)
        # source_indices = np.asarray(source_indices)
        # source_influences = np.asarray(source_influences)
        # sink_connectivities = np.asarray(sink_connectivities)
        # ssis = np.asarray(ssis)
        # powers = np.asarray(powers)
        # evals = np.asarray(evals)
        ahatswdiag = np.asarray(ahatswdiag)
        # SVDS = np.asarray(SVDS)

        #naming()
        #Naming Figures
        titleString = eeg_filename.split('\\')[10]
        titleString1 = titleString.split('_')[0]

        channelHeader = ",".join(keep_channels)

        #Placing Data into Directory
        # directory_path = rootdir + "\\" + titleString1 + "\\" + titleString
        directory_path = rootdir + "\\" + titleString + "\\" + cleanornoisy + "\\" + str(window_length_samples) + "_samples"
        os.makedirs(directory_path, exist_ok=True)

        ######################
        # a_matrices = np.load(directory_path + "\\" + titleString + "diag_matrix_values.npy")
        #####################

        # SVDentropy = stats.entropy(SVDS, axis = 1)
        # np.savetxt(os.path.join(directory_path, (titleString + "SVDentropy.csv")), SVDentropy, delimiter=',')

        histogram()

        # boxplot()

        # eigen()

        # sink()

        # scatter()

        # reconstruct()

        # normsvd = norm(SVDS.transpose())
        # plot_examples([cmap], normsvd)
        #################################
        # r2andp()
        ####################################

        # maxdata, maxsink = gfpmax()

        # sinktopomap()

        # datatopomap()

        savedata()

print("Complete!")

# preprocess then process

# for file in fileList:
file = fileList[0]
if file == fileList[0]:
    # for file in (fileList[0], fileList[30], fileList[60], fileList[90], fileList[120], fileList[150], fileList[180]):

    ahats = []
    sink_indices = []
    source_indices = []
    source_influences = []
    sink_connectivities = []
    ssis = []
    powers = []
    evals = []
    ahatswdiag = []
    SVDS = []

    ## I made a change here because I was confused about the raw eeg being called eeg and the eeg object from eeg_preprocessed.load_data()
    ##it doesnt do anything different so I wont change it back
    #load(file)
    eeg_filename = file
    print(eeg_filename)
    #might need to re run 20 and 325
    eeg = mne.io.read_raw_brainvision(eeg_filename)
    eeg = preprocess_eeg(eeg)
    ## I think my data may already be preprocessed

    print(eeg.info['chs'])
    #features()
    eeg.load_data()
    fs = eeg.info['sfreq']

    window_length_samples = int(fs * window_length)
    window_advance_samples = int(fs * window_advance)

    # Loop over the data
    # window_start = 100*window_advance_samples
    window_start = 0
    data = eeg.get_data()
    nsamples = data.shape[1]


    # this used to say <= which broke if it was =
    while window_start < (nsamples - window_length_samples):

        # Standardize the data
        X = data[:, window_start:window_start + window_length_samples]
        mean_vec = np.mean(X, axis=1)
        X_mean_subtract = X - mean_vec[:, np.newaxis]
        std = np.std(X_mean_subtract, axis=1)
        X_standardized = X / std[:, np.newaxis]

        ahat = computeA(X)
        # energytopomapX()

        # ahat = stabilize_matrix(ahat, 1.05).real


        blankArray = np.zeros((chl, chl - 1))

        #remove diagonal from ahat
        for idx, ele in enumerate(ahat):
            newRow = np.delete(ele, idx)
            blankArray[idx] = newRow

        ahats.append(blankArray)
        ahatswdiag.append(ahat)

        #calculate SVD
        S = np.linalg.svd(X, full_matrices=True, compute_uv=False, hermitian=False)
        SVDS.append(S)


        # Calculate source sink features and store
        sink_idx, source_idx, source_influence, sink_connectivity, ssi = dsp.compute_source_sink_index(ahat)
        sink_indices.append(sink_idx)
        source_indices.append(source_idx)
        source_influences.append(source_influence)
        sink_connectivities.append(sink_connectivity)
        ssis.append(ssi)

        # Calculate power features and store
        powers.append(np.sum(X ** 2, axis=1))
        ## doesnt this just increase for a bigger x?

        # Calculate evals
        vals, _ = np.linalg.eig(ahat)
        evals.append(vals)

        #increment window
        window_start += window_advance_samples

    # window_start = 0

    # while window_start < nsamples:
    #     powers.append(window_start ** 2)
    #     window_start+=1

    ahats = np.asarray(ahats)
    sink_indices = np.asarray(sink_indices)
    source_indices = np.asarray(source_indices)
    source_influences = np.asarray(source_influences)
    sink_connectivities = np.asarray(sink_connectivities)
    ssis = np.asarray(ssis)
    powers = np.asarray(powers)
    evals = np.asarray(evals)
    ahatswdiag = np.asarray(ahatswdiag)
    SVDS = np.asarray(SVDS)

    #naming()
    #Naming Figures
    titleString = eeg_filename.split('\\')[10]
    titleString1 = titleString.split('_')[0]

    channelHeader = ",".join(keep_channels)

    #Placing Data into Directory
    directory_path = rootdir + "\\" + titleString1 + "\\" + titleString
    # directory_path = rootdir + "\\" + titleString + "\\" + cleanornoisy + "\\" + str(window_length_samples) + "_samples"
    os.makedirs(directory_path, exist_ok=True)

    ######################
    # a_matrices = np.load(directory_path + "\\" + titleString + "diag_matrix_values.npy")
    #####################

    SVDentropy = stats.entropy(SVDS, axis = 1)
    np.savetxt(os.path.join(directory_path, (titleString + "SVDentropy.csv")), SVDentropy, delimiter=',')

    histogram()

    boxplot()

    eigen()

    sink()

    scatter()

    reconstruct()

    normsvd = norm(SVDS.transpose())
    plot_examples([cmap], normsvd)
    #################################
    r2andp()
    ####################################

    # maxdata, maxsink = gfpmax()

    # sinktopomap()

    # datatopomap()

    # energytopomap()

    savedata()

    # correlation_matrix([cmap])

print("Complete!")

# skip preprocessing

# for file in fileList:
file = fileList[0]
if file == fileList[0]:
    # file = fileList[0]
    # if file == fileList[0]:
    # for file in fileList[0:2]:

    ahats = []
    sink_indices = []
    source_indices = []
    source_influences = []
    sink_connectivities = []
    ssis = []
    powers = []
    evals = []
    ahatswdiag = []
    SVDS = []

    ## I made a change here because I was confused about the raw eeg being called eeg and the eeg object from eeg_preprocessed.load_data()
    ##it doesnt do anything different so I wont change it back
    #load(file)
    eeg_filename = file
    print(eeg_filename)
    #might need to re run 20 and 325
    # eeg = mne.io.read_raw_brainvision(eeg_filename)
    # eeg = preprocess_eeg(eeg)
    ## I think my data may already be preprocessed

    titleString = eeg_filename.split('\\')[10]
    titleString1 = titleString.split('_')[0]

    channelHeader = ",".join(keep_channels)

    #Placing Data into Directory
    directory_path = rootdir + "\\" + titleString1 + "\\" + titleString
    os.makedirs(directory_path, exist_ok=True)

    data = np.load(directory_path + "\\" + titleString + "data.npy")

    # data = noisydata

    fs = 1000

    #features()
    # eeg.load_data()
    # fs = eeg.info['sfreq']

    window_length_samples = int(fs * window_length)
    window_advance_samples = int(fs * window_advance)

    # Loop over the data
    # window_start = 100*window_advance_samples
    window_start = 0
    # data = eeg.get_data()
    nsamples = data.shape[1]


    # this used to say <= which broke if it was =
    while window_start < (nsamples - window_length_samples):

        # Standardize the data
        X = data[:, window_start:window_start + window_length_samples]
        mean_vec = np.mean(X, axis=1)
        X_mean_subtract = X - mean_vec[:, np.newaxis]
        std = np.std(X_mean_subtract, axis=1)
        X_standardized = X / std[:, np.newaxis]

        ahat = computeA(X)

        # ahat = stabilize_matrix(ahat, 1.05).real


        blankArray = np.zeros((chl, chl - 1))

        #remove diagonal from ahat
        for idx, ele in enumerate(ahat):
            newRow = np.delete(ele, idx)
            blankArray[idx] = newRow

        ahats.append(blankArray)
        ahatswdiag.append(ahat)

        #calculate SVD
        S = np.linalg.svd(X, full_matrices=True, compute_uv=False, hermitian=False)
        SVDS.append(S)


        # Calculate source sink features and store
        sink_idx, source_idx, source_influence, sink_connectivity, ssi = dsp.compute_source_sink_index(ahat)
        sink_indices.append(sink_idx)
        source_indices.append(source_idx)
        source_influences.append(source_influence)
        sink_connectivities.append(sink_connectivity)
        ssis.append(ssi)

        # Calculate power features and store
        powers.append(np.sum(X ** 2, axis=1))
        ## doesnt this just increase for a bigger x?

        # Calculate evals
        vals, _ = np.linalg.eig(ahat)
        evals.append(vals)

        #increment window
        window_start += window_advance_samples



    ahats = np.asarray(ahats)
    sink_indices = np.asarray(sink_indices)
    source_indices = np.asarray(source_indices)
    source_influences = np.asarray(source_influences)
    sink_connectivities = np.asarray(sink_connectivities)
    ssis = np.asarray(ssis)
    powers = np.asarray(powers)
    evals = np.asarray(evals)
    ahatswdiag = np.asarray(ahatswdiag)
    SVDS = np.asarray(SVDS)

    #naming()
    #Naming Figures

    ######################
    # a_matrices = np.load(directory_path + "\\" + titleString + "diag_matrix_values.npy")
    #####################

    SVDentropy = stats.entropy(SVDS, axis = 1)
    np.savetxt(os.path.join(directory_path, (titleString + "SVDentropy.csv")), SVDentropy, delimiter=',')

    histogram()

    boxplot()

    eigen()

    sink()

    scatter()

    reconstruct()

    normsvd = norm(SVDS.transpose())
    plot_examples([cmap], normsvd)
    #################################
    r2andp()
    ####################################

    maxdata, maxsink = gfpmax()

    # sinktopomap()

    # datatopomap()

    # savedata()

print("Complete!")

#redo 4-5

# # noise


# cleanlist = find_dirs_by_extension(rootdir, "clean")

# # for i in range(len(noisylist):
# for i in range(len(fileList)):
#     noisyAmats = find_files_by_extension(cleanlist[i], 'diag_matrix_values.npy')
#     print(len(noisyAmats))

cleanlist = find_dirs_by_extension(rootdir, "clean")
noisylist = find_dirs_by_extension(rootdir, "noisy")

noisearray = np.zeros((189, 12))
cleanarray = np.zeros((189, 12))
difarray = np.zeros((189, 12))

# xticks = (70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150)
xticks = []
for j in range(2, 14):
    xticks.append(j*10+5)
# xticks = []
# for j in range(14, 76):
#     xticks.append(j*5)

# for i in range(len(noisylist):
for i in range(len(fileList)):

    # i = 0
    # if i == i:
    noisyAmats = find_files_by_extension(noisylist[i], 'diag_matrix_values.npy')
    noisydatalist = find_files_by_extension(noisylist[i], 'data.npy')
    cleandatalist = find_files_by_extension(cleanlist[i], 'data.npy')

    noisymeanr = []
    cleanmeanr = []
    difmeanr = []

    j = 0
    window_length_samples = 0


    fileorder = []
    for m in range(len(noisyAmats)):
        window_length_samples = int((noisyAmats[m].split('\\')[9]).split('_')[0])
        fileorder.append(window_length_samples)

    filereorder = [0] * len(fileorder)

    k3 = 0
    for k in fileorder:
        position = 0
        for k2 in fileorder:
            if k > k2:
                position += 1
        filereorder[position] = k3
        k3+=1


    for j in filereorder:
        window_length_samples = int((noisyAmats[j].split('\\')[9]).split('_')[0])
        print(window_length_samples)
        reconstruction = []
        Amats = np.load(noisyAmats[j])
        data = np.load(noisydatalist[j])
        cleandata = np.load(cleandatalist[j])
        aproxdata = data[:,0]
        reconstruction.append(aproxdata)

        n = 0
        while n < Amats.shape[0] and (n+1)*window_length_samples < data.shape[1]:
            # aproxpower = np.matmul(adjustedmatrix[i], aproxpower)
            l = 1
            while l <= window_length_samples:
                aproxdata = Amats[n] @ aproxdata
                reconstruction.append(aproxdata)
                l += 1
            n += 1
            aproxdata = data[:,(n*window_length_samples)]

        reconstruction = np.asarray(reconstruction)


        noisyrlist = []
        cleanrlist = []
        difrlist = []

        for view in range(chl):
            f = 0
            s1cor = []
            s2cor = []
            s3cor = []
            while f < reconstruction.shape[0] - f:
                f += window_length_samples
                s1 = stats.linregress(reconstruction[f-window_length_samples:f,view], data.transpose()[f-window_length_samples:f,view])[2]
                s2 = stats.linregress(reconstruction[f-window_length_samples:f,view], cleandata.transpose()[f-window_length_samples:f,view])[2]
                s1cor.append(s1**2)
                s2cor.append(s2**2)
                s3cor.append((s2**2)-(s1**2))

            noisyrlist.append(np.mean(s1cor))
            cleanrlist.append(np.mean(s2cor))
            difrlist.append(np.mean(s3cor))

        noisymeanr.append(np.mean(noisyrlist))
        cleanmeanr.append(np.mean(cleanrlist))
        difmeanr.append(np.mean(difrlist))


    noisearray[i] = noisymeanr
    cleanarray[i] = cleanmeanr
    difarray[i] = difmeanr

    plt.plot(xticks, noisymeanr)
    plt.ylabel("r^2 between x^t and xt")
    plt.xlabel("samples in window")
    plt.xticks(xticks)
    plt.title("patient " + str(i))
    plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\noise_experiment", "noisy" + str(i) + ".svg"))
    plt.cla()
    plt.clf()
    plt.plot(xticks, cleanmeanr)
    plt.ylabel("r^2 between x^t and xft")
    plt.xlabel("samples in window")
    plt.xticks(xticks)
    plt.title("patient " + str(i))
    plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\noise_experiment", "clean" + str(i) + ".svg"))
    plt.cla()
    plt.clf()
    plt.plot(xticks, difmeanr)
    plt.ylabel("r^2 between x^t and xft - r^2 between x^t and xt")
    plt.xlabel("samples in window")
    plt.xticks(xticks)
    plt.title("patient " + str(i))
    plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\noise_experiment", "dif" + str(i) + ".svg"))
    plt.cla()
    plt.clf()



# np.save(os.path.join(mainfolder + "Archive\\NoiseR^2", "noisearray.npy"), noisearray)
# np.save(os.path.join(mainfolder + "Archive\\NoiseR^2", "cleanarray.npy"), cleanarray)
# np.save(os.path.join(mainfolder + "Archive\\NoiseR^2", "difarray.npy"), difarray)



print(np.load(os.path.join(mainfolder + "Archive\\NoiseR^2", "difarray.npy")))

noisearray = np.load(os.path.join(mainfolder + "Archive\\NoiseR^2", "noisearray.npy"))
cleanarray = np.load(os.path.join(mainfolder + "Archive\\NoiseR^2", "cleanarray.npy"))
difarray = np.load(os.path.join(mainfolder + "Archive\\NoiseR^2", "difarray.npy"))

xticks = []
for j in range(2, 14):
    xticks.append(j*10+5)

for i in noisearray:
    plt.plot(xticks, i)
    plt.xticks(xticks)
plt.ylabel("r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("xt r^2 by window size across population")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "allnoiseplots.svg"))

for i in cleanarray:
    plt.plot(xticks, i)
    plt.xticks(xticks)
plt.ylabel("r^2 between x^t and xft")
plt.xlabel("samples in window")
plt.title("xft r^2 by window size across population")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "allcleanplots.svg"))

for i in difarray:
    plt.plot(xticks, i)
    plt.xticks(xticks)
plt.ylabel("r^2 between x^t and xft - r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("r^2(xft) - r^2(xt) by window size across population")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "alldifplots.svg"))

meandif = []
difup = []
difdown = []
for i in difarray.transpose():
    meandif.append(np.median(i))
    difup.append(np.quantile(i, 0.75))
    difdown.append(np.quantile(i, 0.25))

plt.plot(xticks, difup, color="slateblue")
plt.plot(xticks, meandif, color="black")
plt.plot(xticks, difdown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xft - r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("r^2(xft) - r^2(xt) by window size across population")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "difsummary.svg"))

meannoise = []
noiseup = []
noisedown = []
for i in noisearray.transpose():
    meannoise.append(np.mean(i))
    noiseup.append(np.quantile(i, 0.75))
    noisedown.append(np.quantile(i, 0.25))

plt.plot(xticks, noiseup, color="slateblue")
plt.plot(xticks, meannoise, color="black")
plt.plot(xticks, noisedown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("xt r^2 by window size across population")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "noisesummary.svg"))


meanclean = []
cleanup = []
cleandown = []
for i in cleanarray.transpose():
    meanclean.append(np.mean(i))
    cleanup.append(np.quantile(i, 0.75))
    cleandown.append(np.quantile(i, 0.25))

plt.plot(xticks, cleanup, color="slateblue")
plt.plot(xticks, meanclean, color="black")
plt.plot(xticks, cleandown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xft")
plt.xlabel("samples in window")
plt.title("xft r^2 by window size across population")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "cleansummary.svg"))

cleanlist = find_dirs_by_extension(rootdir, "clean")
noisylist = find_dirs_by_extension(rootdir, "noisy")
print(cleanlist[0])
print(rootdir)

print(xlarray[0])

healthy = []
unhealthy = []
CWP = []
CBP = []
JP = []
NP = []

for i in range(numPatients):
    patientid = xlarray[0, i]
    try:
        np.isnan(xlarray[1, i])
        healthy.append(rootdir + '\\' + patientid + eyestate + '\\' + noisy)
    except TypeError:
        unhealthy.append(rootdir + '\\' + patientid + eyestate + '\\' + noisy)
        match patientid:
            case "CWP":
                CWP.append(rootdir + '\\' + patientid + eyestate + '\\' + noisy)
            case "CBP":
                CBP.append(rootdir + '\\' + patientid + eyestate + '\\' + noisy)
            case "PNP":
                NP.append(rootdir + '\\' + patientid + eyestate + '\\' + noisy)
            case "NP":
                NP.append(rootdir + '\\' + patientid + eyestate + '\\' + noisy)
            case "JP":
                JP.append(rootdir + '\\' + patientid + eyestate + '\\' + noisy)
            case "PHN":
                NP.append(rootdir + '\\' + patientid + eyestate + '\\' + noisy)


for i in range(numPatients):
    if noisylist[i] in unhealthy:
        plt.plot(xticks, cleanarray[i])
        plt.xticks(xticks)
plt.ylabel("r^2 between x^t and xft")
plt.xlabel("samples in window")
plt.title("xft r^2 by window size across unhealthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "unhealtycleanplots.svg"))



for i in range(numPatients):
    if noisylist[i] in healthy:
        plt.plot(xticks, noisearray[i])
        plt.xticks(xticks)
plt.ylabel("r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("xt r^2 by window size across healthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "healthynoiseplots.svg"))



for i in range(numPatients):
    if noisylist[i] in unhealthy:
        plt.plot(xticks, difarray[i])
        plt.xticks(xticks)
plt.ylabel("r^2 between x^t and xft - r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("r^2(xft) - r^2(xt) by window size across unhealthy")

plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "unhealthydifplots.svg"))

difhealthy = []
difunhealthy = []
noisehealthy = []
noiseunhealthy = []
cleanhealthy = []
cleanunhealthy = []

for i in range(numPatients):
    if noisylist[i] in healthy:
        difhealthy.append(difarray[i])
difhealthy = np.asarray(difhealthy)
for i in range(numPatients):
    if noisylist[i] in unhealthy:
        difunhealthy.append(difarray[i])
difunhealthy = np.asarray(difunhealthy)
for i in range(numPatients):
    if noisylist[i] in unhealthy:
        noiseunhealthy.append(noisearray[i])
noiseunhealthy = np.asarray(noiseunhealthy)
for i in range(numPatients):
    if noisylist[i] in healthy:
        noisehealthy.append(noisearray[i])
noisehealthy = np.asarray(noisehealthy)
for i in range(numPatients):
    if noisylist[i] in unhealthy:
        cleanunhealthy.append(cleanarray[i])
cleanunhealthy = np.asarray(cleanunhealthy)
for i in range(numPatients):
    if noisylist[i] in healthy:
        cleanhealthy.append(cleanarray[i])
cleanhealthy = np.asarray(cleanhealthy)

meandif = []
difup = []
difdown = []
for i in difhealthy.transpose():
    meandif.append(np.median(i))
    difup.append(np.quantile(i, 0.75))
    difdown.append(np.quantile(i, 0.25))

plt.plot(xticks, difup, color="slateblue")
plt.plot(xticks, meandif, color="black")
plt.plot(xticks, difdown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xft - r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("r^2(xft) - r^2(xt) by window size across healthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "healthydifsummary.svg"))

meandif = []
difup = []
difdown = []
for i in difunhealthy.transpose():
    meandif.append(np.median(i))
    difup.append(np.quantile(i, 0.75))
    difdown.append(np.quantile(i, 0.25))

plt.plot(xticks, difup, color="slateblue")
plt.plot(xticks, meandif, color="black")
plt.plot(xticks, difdown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xft - r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("r^2(xft) - r^2(xt) by window size across unhealthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "unhealthydifsummary.svg"))

meannoise = []
noiseup = []
noisedown = []
for i in noiseunhealthy.transpose():
    meannoise.append(np.mean(i))
    noiseup.append(np.quantile(i, 0.75))
    noisedown.append(np.quantile(i, 0.25))

plt.plot(xticks, noiseup, color="slateblue")
plt.plot(xticks, meannoise, color="black")
plt.plot(xticks, noisedown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("xt r^2 by window size across unhealthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "unhealthynoisesummary.svg"))


meannoise = []
noiseup = []
noisedown = []
for i in noisehealthy.transpose():
    meannoise.append(np.mean(i))
    noiseup.append(np.quantile(i, 0.75))
    noisedown.append(np.quantile(i, 0.25))

plt.plot(xticks, noiseup, color="slateblue")
plt.plot(xticks, meannoise, color="black")
plt.plot(xticks, noisedown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.title("xt r^2 by window size across healthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "healthynoisesummary.svg"))


meanclean = []
cleanup = []
cleandown = []
for i in cleanunhealthy.transpose():
    meanclean.append(np.mean(i))
    cleanup.append(np.quantile(i, 0.75))
    cleandown.append(np.quantile(i, 0.25))

plt.plot(xticks, cleanup, color="slateblue")
plt.plot(xticks, meanclean, color="black")
plt.plot(xticks, cleandown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xft")
plt.xlabel("samples in window")
plt.title("xft r^2 by window size across unhealthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "unhealthycleansummary.svg"))

meanclean = []
cleanup = []
cleandown = []
for i in cleanhealthy.transpose():
    meanclean.append(np.mean(i))
    cleanup.append(np.quantile(i, 0.75))
    cleandown.append(np.quantile(i, 0.25))

plt.plot(xticks, cleanup, color="slateblue")
plt.plot(xticks, meanclean, color="black")
plt.plot(xticks, cleandown, color="red")
plt.xticks(xticks)
plt.legend(("Q3", "Median", "Q1"))
plt.ylabel("r^2 between x^t and xft")
plt.xlabel("samples in window")
plt.title("xft r^2 by window size across healthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "healthycleansummary.svg"))



noiseptest = []
for n in range(12):
    t, p = stats.ttest_ind_from_stats(np.mean(noiseunhealthy[:,n]), np.std(noiseunhealthy[:,n]), 101, np.mean(noisehealthy[:,n]), np.std(noisehealthy[:,n]), 88)
    noiseptest.append(p)
    print(p)

plt.plot(xticks, noiseptest)
plt.xticks(xticks)
plt.ylabel("p value")
plt.xlabel("window length")
plt.title("p between noisy for healthy and unhealthy")
plt.savefig(os.path.join(mainfolder + "Archive\\NoiseR^2", "pxt.svg"))

print(rootdir + '\\' + patientids[0] + eyestate)

# # A matrix tests


data = np.load(datalist[0])  # Expected shape: (n_channels, n_samples)

# If data is 1D, convert to 2D (1 channel)
if data.ndim == 1:
    data = data[np.newaxis, :]

n_channels, n_samples = data.shape
sfreq = 100  # Sampling frequency in Hz — update as needed

# Create MNE info structure
ch_names = [f"Ch{i+1}" for i in range(n_channels)]
ch_types = ['eeg'] * n_channels  # or use 'ecg', 'emg', etc.
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create RawArray
raw = mne.io.RawArray(data, info)

# Save to EDF
raw.export(mainfolder + "output.edf", fmt='edf')  # fmt='edf' or 'edf+'

print("EDF file saved as output.edf")

print(powers[0])

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(powers)
    inertias.append(kmeans.inertia_)
fig, ax = plt.subplots()

plt.plot(range(1,11), inertias, marker='o')
# plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "elbow.svg"))


kmeans = KMeans(n_clusters=4)
kmeans.fit(powers)

plt.scatter(powers[:,3], powers[:,2], c=kmeans.labels_)
plt.show()

print(len(kmeans.labels_))
plt.plot(kmeans.labels_[0:30])
plt.yticks((0,1,2,3))
plt.ylabel("cluster")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "clusterchange.svg"))

changes=np.zeros((4,4), dtype=int)
points=np.zeros((4,), dtype=int)
for i in range(len(kmeans.labels_)-1):
    x=kmeans.labels_[i]
    y=kmeans.labels_[i+1]
    changes[x,y]+=1
    points[x]+=1
points[y]+=1

print(changes)
print(points)

plt.bar((0,1,2,3),points)
plt.xlabel("Clusters")
plt.xticks((0,1,2,3))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "clusterbar.svg"))

plt.scatter(kmeans.cluster_centers_[:,5],kmeans.cluster_centers_[:,4])

plt.topomap()

fig, ax = plt.subplots()

n=2

mne.viz.plot_topomap(kmeans.cluster_centers_[n], xypos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "cluster" + str(n) + ".svg"))


image_folder = mainfolder + "Archive\\Old_Graphs\\topomapvideo"
video_name = mainfolder + "Archive\\Old_Graphs\\video.mp4"

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))



cv2.destroyAllWindows()
video.release()

video.release()

data = np.zeros((chl,300000), dtype=float)
for i in range(chl):
    a = np.random.default_rng().random()
    b = np.random.default_rng().random()
    for x in range(data.shape[1]):
        data[i,x] = a * np.sin((x+b))

print(j)
print(filereorder)

print(Amats.shape[0]*window_length_samples)
print(data.shape)
print(reconstruction.shape)

xticks=[]

for j in range(5, 21):
    xticks.append(j*5)

plt.plot(xticks, noisymeanr)
plt.ylabel("r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.xticks(xticks)
plt.title("patient " + str(i))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\noise_experiment", "noisy" + str(i) + ".svg"))
plt.cla()
plt.clf()
plt.plot(xticks, cleanmeanr)
plt.ylabel("r^2 between x^t and xft")
plt.xlabel("samples in window")
plt.xticks(xticks)
plt.title("patient " + str(i))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\noise_experiment", "clean" + str(i) + ".svg"))
plt.cla()
plt.clf()
plt.plot(xticks, difmeanr)
plt.ylabel("r^2 between x^t and xft - r^2 between x^t and xt")
plt.xlabel("samples in window")
plt.xticks(xticks)
plt.title("patient " + str(i))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\noise_experiment", "dif" + str(i) + ".svg"))
plt.cla()
plt.clf()



for j in range(len(noisyAmats)):
    print(noisyAmats[j])

noiseless = data

# noisydata = np.zeros(noiseless.shape)
# for i in range(noiseless.shape[0]):
# noisydata[i] = noiseless[i] + np.random.normal(0, np.std(noiseless)/10, noiseless.shape[1])

noisydata = data

print(data.shape)

m=0
options = {
    'node_color': 'blue',
    'node_size': 200,
    'font_size': 6,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 12,
}

while m < 1.01:
    plt.clf()
    plt.cla()
    A_Connections = nx.Graph()
    Connected_Nodes = []

    for i in range(chl):
        for j in range(chl):
            # if ahatswdiag[m, i, j] > 0.3:
            # if np.mean(ahatswdiag[1:, i, j], axis = 0) > 0.1:
            if np.max(ahatswdiag[1:, i, j], axis = 0) > m:

                A_Connections.add_edges_from([(keep_channels[i], keep_channels[j])])
                if i != j:
                    if keep_channels[i] not in Connected_Nodes:
                        Connected_Nodes.append(keep_channels[i])
                    if keep_channels[j] not in Connected_Nodes:
                        Connected_Nodes.append(keep_channels[j])

    nx.shell_layout(A_Connections, scale = 1)
    plt.figure(1,figsize=(10, 10))
    nx.draw_networkx(A_Connections, arrows=True, **options)

    plt.title(str(np.round(m, 2)) + ' ' + str(len(Connected_Nodes)) + " " + str(len(A_Connections.edges)))
    plt.draw()
    plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "A-Mat_Connect_Means", str(np.round(m, 2)) + '_' + str(len(Connected_Nodes)) + "_" + str(len(A_Connections.edges)) + ".png"))
    m+=0.01




Connectivity = []

while m <= 1:
    A_Connections = nx.Graph()
    Connected_Nodes = []

    for i in range(chl):
        for j in range(chl):
            # if ahatswdiag[m, i, j] > 0.3:
            # if np.mean(ahatswdiag[1:, i, j], axis = 0) > 0.1:
            if np.max(ahatswdiag[1:, i, j], axis = 0) > m:

                A_Connections.add_edges_from([(keep_channels[i], keep_channels[j])])
                if i != j:
                    if keep_channels[i] not in Connected_Nodes:
                        Connected_Nodes.append(keep_channels[i])
                    if keep_channels[j] not in Connected_Nodes:
                        Connected_Nodes.append(keep_channels[j])
    # nx.shell_layout(A_Connections, scale = 1)
    # plt.figure(1,figsize=(10, 10))
    # nx.draw_networkx(A_Connections, arrows=True, **options)
    # plt.title(str(m) + ' ' + str(len(Connected_Nodes)))
    m+=0.01
    # plt.draw()
    Connectivity.append(len(Connected_Nodes))

plt.plot(np.arange(0, 1, 0.01), Connectivity)

print(powers.shape)

for i in powers.transpose():
    plt.hist(i,bins=100, histtype='step')

plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "histofenergy.svg"))


print(data.shape)

dependents = []
dependentskip = []
for i in range(data.shape[0]):
    if i not in dependentskip:
        for j in range(i, data.shape[0]):
            if j not in dependentskip:
                k = stats.linregress(data[i], data[j])[2]
                if 0.95 < k < 1:
                    if keep_channels[i] not in dependents:
                        dependents.append(keep_channels[i])
                    if keep_channels[j] not in dependents:
                        dependentskip.append(j)

                    print(keep_channels[i] + ' ' + keep_channels[j] + ' ' + str(k))

print(len(dependentskip))
print(len(dependents))
# print(dependents)

dependents = []
dependentskip = []
for i in range(data.shape[0]):
    if i not in dependentskip:
        for j in range(i, data.shape[0]):
            if j not in dependentskip:
                k = stats.linregress(data[i], data[j])[2]
                if 0.90 < k < 1:
                    if keep_channels[i] not in dependents:
                        dependents.append(keep_channels[i])
                    if keep_channels[j] not in dependents:
                        dependentskip.append(j)

                    print(keep_channels[i] + ' ' + keep_channels[j] + ' ' + str(k) + ' ' + str(np.mean(data[i])) + ' ' + str(np.mean(data[j])))

print(len(dependentskip))
print(len(dependents))
# print(dependents)

#dont use without messing with function

kmat = []
for i in range(data.shape[0]):
    krow = []
    for j in range(data.shape[0]):
        k = stats.linregress(data[i], data[j])[2]
        krow.append(k)
    kmat.append(krow)

kmat = np.asarray(kmat)
plot_examples([cmap], kmat)
# print(dependents)

print(data.shape)

print(sink_indices.shape)

print(215780/125)
print(215825/125)

magnitude = []

for i in data.transpose():
    magnitude.append(np.std(i))

# plt.plot(magnitude[215780:215825])
plt.plot(magnitude)
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "magnitude.png"))

print(len(magnitude))
print(magnitude[len(magnitude)-1])

maximums = []

for i in range(1, len(magnitude)):
    if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1] and int(np.trunc(i/125)) < sink_indices.shape[0]:
        maximums.append(i)

print(sink_indices.shape)

maxdata = []
maxsink = []

for i in maximums:
    maxdata.append(data.transpose()[i])
    maxsink.append(sink_indices[int(np.trunc(i/125))])


maxdata = np.asarray(maxdata)
maxsink = np.asarray(maxsink)


sinktopomap()

print(np.max(magnitude))

print(powers.shape)

n = 0
for i in sink_indices.transpose():
    print(str(n)+' '+keep_channels[n]+' '+str(np.mean(i)))
    n+=1

n = 0
for i in source_indices.transpose():
    print(str(n)+' '+keep_channels[n]+' '+str(np.mean(i))+" | "+str(np.mean(powers.transpose()[n])))
    n+=1

print(a_matrices.shape)

norm = colors.Normalize()


# normsvd = norm(SVDS.transpose()[0:50])
normsvd = norm(SVDS.transpose())
plot_examples([cmap], normsvd)
# plt.savefig(os.path.join(mainfolder + "Data\\Old_Graphs", "colors.png"))


print(viridis)



print(nsamples)
print(window_length_samples)
print(nsamples/window_length_samples)
print(evals.shape)

data = noisydata

print(noisydatalist)

print(noisydatalist[0])

print(np.sum(ahatswdiag[200].flatten())/19)


# data=np.load('C:\\Users\\Ultimateo\\OneDrive\\Desktop\\ChronicPainPractice\\noisedirectory\\sub-CBPpa38_task-closed_eeg.vhdr\\noisy\\99_samples\\sub-CBPpa38_task-closed_eeg.vhdrdata.npy')
# clean=np.load('C:\\Users\\Ultimateo\\OneDrive\\Desktop\\ChronicPainPractice\\noisedirectory\\sub-CBPpa38_task-closed_eeg.vhdr\\clean\\99_samples\\sub-CBPpa38_task-closed_eeg.vhdrdata.npy')
# window_length_samples=45
# ahatswdiag=np.load('C:\\Users\\Ultimateo\\OneDrive\\Desktop\\ChronicPainPractice\\noisedirectory\\sub-CBPpa38_task-closed_eeg.vhdr\\noisy\\99_samples\\sub-CBPpa38_task-closed_eeg.vhdrdiag_matrix_values.npy')

# adjustedmatrix = ahatswdiag*(1.233213508)
aproxeeg = []
aproxdata = data[:,0]


##this is horrible and slow but just testing

aproxeeg.append(aproxdata)

i = 0
while i < ahatswdiag.shape[0]:
    # while i < 200:
    # aproxpower = np.matmul(adjustedmatrix[i], aproxpower)
    l = 1
    while l <= window_length_samples:
        aproxdata = ahatswdiag[i] @ aproxdata
        aproxeeg.append(aproxdata)
        l += 1
    i += 1
    aproxdata = data[:,(i*window_length_samples)]

aproxeeg = np.asarray(aproxeeg)


# print(aproxeeg[1,0])
# print(powers[1,0])

start = 0
end = aproxeeg.shape[0]-1

start = window_length_samples+1
# end = 95000

# start = aproxeeg.shape[0]-3000

# end = 90
# end = 45
# end = 500
# end = 85000

view = 0
# view2 = 16
# plt.ylim(-1e-4, 1e-4)

plt.plot(range(end+1)[start:end], data.transpose()[start:end,view], color = "red")
# plt.plot(range(end+1)[start:end], clean.transpose()[start:end,view], color = "lawngreen")
plt.plot(range(end+1)[start:end], aproxeeg[start:end,view], color = "slateblue")
# plt.plot(range(end+1)[start:end], data.transpose()[start:end,view], color = "red")
# plt.plot(range(end+1)[start:end], aproxeeg[start:end,view])
# plt.plot(range(end+1)[start:end], aproxeeg[start:end,view+1])
# # plt.plot(range(end+1)[start:end], data.transpose()[start:end,view])
plt.title("Example Reconstruction Full")
# plt.legend(("No Low Pass", "Full ICA", "Reconstruction"))
plt.legend(("Data",))
plt.legend(("Data","Reconstruction"))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "reconstruct_Full.svg"))
# plt.title("250 Samples 62 Channels (FM < AM)")
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\PresentationDrawings", "smoothing25062.svg"))





print(np.mean(noisydata))

data = noisydata
gausdif=[]
view=0
for i in range(500, end+1):
    # gausdif.append(data.transpose()[i,view])
    # gausdif.append(aproxeeg[i,view]-data.transpose()[i,view])
    if np.abs(aproxeeg[i,view]-data.transpose()[i,view]) < 1e-5:
        gausdif.append(aproxeeg[i,view]-data.transpose()[i,view])

mean,std=stats.norm.fit(gausdif)
plt.hist(gausdif, bins = 200)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, len(gausdif))
y = stats.norm.pdf(x, mean, std)
# plt.plot(x, y)
plt.show()

# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "low_R^2_Gaussian.png"))
data = noiseless

means125noisy = []
means125raw = []

# means250noisy = []
# print(len(means250noisy))
# means250raw = []

# means500noisy = []
# means500raw = []

view = 0
while view < chl:

    i = window_length_samples
    s1cor = []
    s2cor = []
    while i < aproxeeg.shape[0]:
        s1 = stats.linregress(aproxeeg[i-window_length_samples:i,view], noisydata.transpose()[i-window_length_samples:i,view])[2]
        s2 = stats.linregress(aproxeeg[i-window_length_samples:i,view], noiseless.transpose()[i-window_length_samples:i,view])[2]
        # s = stats.linregress(noisydata.transpose()[i-window_length_samples:i,view], noiseless.transpose()[i-window_length_samples:i,view])[2]
        s1cor.append(s1**2)
        s2cor.append(s2**2)

        i+=window_length_samples

    means125noisy.append(np.mean(s1cor))
    means125raw.append(np.mean(s2cor))

    view+=1

# plt.plot(range(len(pcor)), pcor)
# # plt.title("r^2 between reconstruction and noisy per window")
# plt.title("r^2 between reconstruction and raw per window")
# # plt.title("r^2 between noisy and raw per window")

# plt.hlines([np.mean(pcor)], 0, aproxeeg.shape[0]/window_length_samples, color = "black")
# plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.round(np.mean(pcor), 2)])
# plt.hlines([np.median(pcor)], 0, aproxeeg.shape[0]/window_length_samples, color = "black")
# plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.round(np.median(pcor), 2)])

# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "r2reconstructionandnoisy.png"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "r2reconstructionandraw.png"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "r2noisyandraw.png"))

means125dif = []
for i, j in zip(means125raw, means125noisy):
    means125dif.append(i-j)

means250dif = []
for i, j in zip(means250raw, means250noisy):
    means250dif.append(i-j)

means500dif = []
for i, j in zip(means500raw, means500noisy):
    means500dif.append(i-j)

print(len(papos))

fig, ax = plt.subplots(1, 1, figsize=(chl, 15), squeeze=False)
# ax[0,0].boxplot(meanofallsis, showfliers=False)
# ax[0,0].set_title("si by Channel")
ax[0,0].set_xticks(np.arange(0.5, chl+0.5), keep_channels)
ax[0,0].set_xlim(-0.25, 62)

# siplot = ax[0,0].boxplot(meanofallsis, showfliers=False, patch_artist=True, boxprops=dict(facecolor="red", color="red"))
papos = []
papos2 = []
for i in range(0, chl):
    papos.append(i + 1/3)
    papos2.append(i + 2/3)


plot125 = ax[0,0].bar(range(0,chl), means125raw, width = .25)
# , patch_artist=True, boxprops=dict(facecolor="red", color="red"), widths=0.25)
plot250 = ax[0,0].bar(papos, means250raw, width = .25)
# , patch_artist=True, boxprops=dict(facecolor="slateblue", color="slateblue"), widths=0.25)
plot500 = ax[0,0].bar(papos2, means500raw, width = .25)


plt.legend(("125 milliseconds", "250 milliseconds", "500 milliseconds"))
# plt.title("Difference between mean r^2 values of reconstruction with data filtered between 1-30hz and reconstruction with non-low-passed data")
# plt.title("Mean r^2 values of reconstruction and data filtered between 1-30hz")
plt.title("Mean r^2 values of reconstruction and data with no low-pass filter")
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "errormeansnoisy.svg"))


i = window_length_samples
pcor = []
while i < aproxeeg.shape[0]:
    s = stats.linregress(aproxeeg[i-window_length_samples:i,view], noisydata.transpose()[i-window_length_samples:i,view])[2]
    # s = stats.linregress(aproxeeg[i-window_length_samples:i,view], noiseless.transpose()[i-window_length_samples:i,view])[2]
    # s = stats.linregress(noisydata.transpose()[i-window_length_samples:i,view], noiseless.transpose()[i-window_length_samples:i,view])[2]
    pcor.append(s**2)
    i+=window_length_samples

plt.plot(range(len(pcor)), pcor)
plt.title("r^2 between reconstruction and noisy per window")
# plt.title("r^2 between reconstruction and filtered per window")
# plt.title("r^2 between noisy and raw per window")

plt.hlines([np.mean(pcor)], 0, aproxeeg.shape[0]/window_length_samples, color = "black")
plt.yticks(ticks=[0.0, 0.4, 0.6, 0.8, 1.0, np.round(np.mean(pcor), 2)])
# plt.hlines([np.median(pcor)], 0, aproxeeg.shape[0]/window_length_samples, color = "black")
# plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.round(np.median(pcor), 2)])

plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "r2reconstructionandnoisy.png"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "r2reconstructionandraw.png"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "r2noisyandraw.png"))

print(np.mean(pcor))

print(np.median(pcor))

print(np.median(pcor))

print(xmin)

print(stats.jarque_bera(gausdif))

print(stats.norm.fit(gausdif))

# print(stats.linregress(aproxeeg[start:end,view], data.transpose()[start:end,view]))
print(np.mean(stats.pearsonr(aproxeeg[start:end], data.transpose()[start:end])[0]))


i = 10
pcor = []
while i < aproxeeg.shape[0]:
    s = stats.linregress(aproxeeg[start:i,view], data.transpose()[start:i,view])[2]
    pcor.append(s)
    i+=10

plt.plot(range(len(pcor)), pcor)
# plt.savefig(os.path.join(mainfolder + "Data\\Old_Graphs", "r_over_time.png"))


end2=evals.shape[0]-1


# end2=53

start2=0

reals = []
imags = []

for idx in evals:
    real = [ele.real for ele in idx]
    imag = [ele.imag for ele in idx]
    reals.append(real)
    imags.append(imag)

reals = np.asarray(reals)
imags = np.asarray(imags)

# plt.ylim(.8, .9)


for channel in reals.transpose():
    plt.plot(range(len(channel)+1)[start2:end2], channel[start2:end2])

# i=0
# channel = reals[:,i]
# plt.plot(range(len(channel)+1)[0:end2-1], channel[0:end2-1])
# print(i)
# i+=1

# plt.ylim(.45,.55)

for channel in imags.transpose():
    plt.plot(range(len(channel)+1)[start2:end2], channel[start2:end2])

# i=14
#squaring an imaginary number and i get a positive? weird, maybe these are only the negative ones? no i guess it just didnt extract the imaginary part
# plt.plot(range(evals.shape[0]), (reals[:,i]**2+imags[:,i]**2))
# print(i)
# print(imags[0:20,i]**2)
# print(imags[0:20,i])
maxes = []
for i in range(evals.shape[0]):
    maxes.append(max(reals[i]**2+imags[i]**2))
plt.plot(range(evals.shape[0]), maxes)


view = 0
# start = 55000
# end = data.shape[1]-1
start = 0
end = 5000
plt.plot(range(end+1)[start:end], data[view, start:end])
# plt.savefig(os.path.join(mainfolder + "Data\\Old_Graphs", "preprocessed_data_1A.png"))

start = 0
end = data.shape[1]-1


plt.plot(range(end+1)[start:end], data[view, start:end])
# plt.savefig(os.path.join(mainfolder + "Data\\Old_Graphs", "preprocessed_data_1.png"))

unprocessed = mne.io.read_raw_brainvision(eeg_filename)
unprocessed.pick(keep_channels)
unprocessed.load_data()
rawdata = unprocessed.get_data()

plt.plot(range(rawdata.shape[1]), rawdata[view])
# plt.savefig(os.path.join(mainfolder + "Data\\Old_Graphs", "rawdataexample.png"))

# # Results


# ## Preprocessing


##the dataset I'm using has eyes closed eegs for all patients but some didn't do an eyes open eeg
##since I'm only doing the eyes closed data for now I could just add all the data from participants.tsv to the same row in PopulationStats.xlsx
##But I want to make this work for if I use the open eyes eegs later

print(rootdir)

##locations of all the data i'm using
eyestate = '_task-closed_eeg.vhdr'

subdirlist = find_dirs_by_extension(rootdir, eyestate)
# print(subdirlist)

##patient id
patientids = []

for name in subdirlist:
    pid = name.split('\\')[7]
    pid = pid.split('_')[0]
    patientids.append(pid)

numPatients = len(patientids)

print(patientids)

##extract participants.tsv as a DataFrame then remove the entries that dont have closed eye eegs

##defining participants

participants = pd.read_table(mainfolder + "Data\\participants.tsv")

dtypes = participants.dtypes
pcolumns = participants.columns

participants = participants.to_numpy()

removerows = []

##remove unwanted data from participants

x = 0

while x < (len(participants)):
    if participants[x][0] not in patientids:
        removerows.append(x)
    x += 1

participants = np.delete(participants, (removerows), axis = 0)



##define and order variables from participants
patientids = participants[:, 0]
diseasestates = participants[:, 5]
sex = participants[:, 3]
age = participants[:, 4]

currentpain = participants[:, 12]
avgpain = participants[:, 13]
paindur = participants[:, 15]

medquants = participants[:, 19]
pdisq = participants[:, 14]
pdisi = participants[:, 18]
mcgill = participants[:,11]


# print(subdirlist)
#put list of subdirectories in order
x = 0
while x < numPatients:
    subdirlist[x] = rootdir + '\\' + patientids[x] + '\\' + patientids[x] + eyestate
    x += 1

print(subdirlist)

##sort the a matrix, sink indices, entropy files
amatlist = []
silist = []
entrolist = []
diaglist = []
powerslist = []
r2list = []
plist = []
powerlist = []
SVDlist = []
eigenlist = []
datalist = []
reallist = []
imaglist = []


for file in subdirlist:
    r2list.append(find_files_by_extension(file, 'r^2.csv')[0])
    plist.append(find_files_by_extension(file, '_p.csv')[0])
    powerlist.append(find_files_by_extension(file, 'power_by_channel.csv')[0])
    amatlist.append(find_files_by_extension(file, 'A_matrix_values.npy')[0])
    silist.append(find_files_by_extension(file, 'sink_indices.csv')[0])
    entrolist.append(find_files_by_extension(file, 'SVDentropy.csv')[0])
    diaglist.append(find_files_by_extension(file, 'diag_matrix_values.npy')[0])
    powerslist.append(find_files_by_extension(file, 'power_by_channel.csv')[0])
    SVDlist.append(find_files_by_extension(file, 'SVD.csv')[0])
    eigenlist.append(find_files_by_extension(file, '_eigenvalues.csv')[0])
    datalist.append(find_files_by_extension(file, 'data.npy')[0])
    reallist.append(find_files_by_extension(file, 'eigen_real.csv')[0])
    imaglist.append(find_files_by_extension(file, 'eigen_imag.csv')[0])

# ## Kmeans


print(xlarray[27])

nvclist = []
nvplist = []

for file in powerslist:
    energy = pd.read_csv(file)
    energy = energy.to_numpy().transpose()
    # print(energy.shape)
    nvc, nvp = cluster(energy, 4)
    nvclist.append(nvc)
    nvplist.append(nvp)

nvclist = np.asarray(nvclist)
nvplist = np.asarray(nvplist)

print(energy.shape)


def normalize_matrix(matrix):
    # Convert to a numpy array if it's not already
    matrix = np.array(matrix)

    # Find the minimum and maximum values of the matrix
    min_val = matrix.min()
    max_val = matrix.max()

    # Normalize the matrix between 0 and 1
    normalized_matrix = (matrix - min_val) / (max_val - min_val)

    return normalized_matrix

# Example usage:
matrix = [[3, 7, 9], [1, 4, 6], [2, 8, 5]]
normalized_matrix = normalize_matrix(matrix)
print(normalized_matrix)

print(np.mean(energy))

si = pd.read_csv(silist[0])
si = si.to_numpy().transpose()

print(si.shape)

amat = np.load(diaglist[0])
amat = amat.reshape(amat.shape[0],-1)

print(amat.shape)

# 88 hc 47 cbp 30 np
# 62 33 21
hc = 0
pa = 0


allenergy = np.zeros((19,1))
usedlen = []
n=0
# for file in silist:
# for file in datalist:
# for file in diaglist:
# for file in powerslist:
for file in silist:
    energy = pd.read_csv(file)
    energy = energy.to_numpy().transpose()
    # print(energy.shape)
    # energy = np.load(file)
    # energy = energy.reshape(energy.shape[0],-1)
    energy = normalize_matrix(energy)
    usedlen.append(energy.shape[0])
    try:
        np.isnan(diseasestates[n])
        n+=0
        if hc < 62:
            allenergy = np.concatenate((allenergy, energy), axis = 1)
            hc += 1
    except TypeError:
        n+=0
        # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
        if diseasestates[n] == 'CBP':
            if pa < 33:
                allenergy = np.concatenate((allenergy, energy), axis = 1)
                pa += 1
    n+=1

allenergy = allenergy

# 88 hc 47 cbp 30 np
# 62 33 21
hc = 0
pa = 0


allenergy = np.zeros((19,1))
usedlen = []
n=0
# for file in silist:
# for file in datalist:
# for file in diaglist:
# for file in powerslist:
for file in silist:
    energy = pd.read_csv(file)
    energy = energy.to_numpy().transpose()
    # print(energy.shape)
    # energy = np.load(file)
    # energy = energy.reshape(energy.shape[0],-1)
    energy = normalize_matrix(energy)
    usedlen.append(energy.shape[0])
    try:
        np.isnan(diseasestates[n])
        n+=0
        # if hc < 62:
        #     allenergy = np.concatenate((allenergy, energy), axis = 1)
        #     hc += 1
    except TypeError:
        n+=0
        # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
        if diseasestates[n] == 'CBP':
            if pa < 33:
                allenergy = np.concatenate((allenergy, energy), axis = 1)
                pa += 1
    n+=1

allenergy = allenergy

# 88 hc 47 cbp 30 np
# 62 33 21
hc = 0
pa = 0


allenergy1 = np.zeros((19,1))
usedlen1 = []
n=0
# for file in silist:
# for file in datalist:
# for file in diaglist:
# for file in powerslist:
for file in silist:
    energy = pd.read_csv(file)
    energy = energy.to_numpy().transpose()
    # print(energy.shape)
    # energy = np.load(file)
    # energy = energy.reshape(energy.shape[0],-1)
    energy = normalize_matrix(energy)
    usedlen1.append(energy.shape[0])
    try:
        np.isnan(diseasestates[n])
        n+=0
        if hc < 62:
            allenergy1 = np.concatenate((allenergy1, energy), axis = 1)
            hc += 1
    except TypeError:
        n+=0
        # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
        # if diseasestates[n] == 'CBP':
        #     if pa < 33:
        #         allenergy = np.concatenate((allenergy, energy), axis = 1)
        #         pa += 1
    n+=1

allenergy = allenergy

print(allenergy.shape)

print(powers.shape)

inertias = []

for i in range(1,20):
    kmeans = KMeans(n_clusters=i, max_iter=10000)
    kmeans.fit(allenergy.transpose())
    inertias.append(kmeans.inertia_)
fig, ax = plt.subplots()

plt.plot(range(1,20), inertias, marker='o')
# plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.xticks(range(20))
plt.ylabel('Inertia')
plt.show()

fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "superelbow.svg"))

inertias1 = []

for i in range(1,20):
    kmeans = KMeans(n_clusters=i, max_iter=10000)
    kmeans.fit(allenergy1.transpose())
    inertias1.append(kmeans.inertia_)
fig, ax = plt.subplots()

plt.plot(range(1,20), inertias1, marker='o')
# plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.xticks(range(20))
plt.ylabel('Inertia')
plt.show()

fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "superelbow1.svg"))

inertias = []

nclusters=4
kmeans = KMeans(n_clusters=nclusters, max_iter=1000)
kmeans.fit(allenergy.transpose())
inertias.append(kmeans.inertia_)

inertias1 = []

nclusters=4
kmeans1 = KMeans(n_clusters=nclusters, max_iter=1000)
kmeans1.fit(allenergy1.transpose())
inertias1.append(kmeans.inertia_)



changes=np.zeros((nclusters,nclusters), dtype=int)
points=np.zeros((nclusters,), dtype=int)
for i in range(len(kmeans.labels_)-1):
    x=kmeans.labels_[i]
    y=kmeans.labels_[i+1]
    changes[x,y]+=1
    points[x]+=1
points[y]+=1

print(changes)
print(points)



changes1=np.zeros((nclusters,nclusters), dtype=int)
points1=np.zeros((nclusters,), dtype=int)
for i in range(len(kmeans1.labels_)-1):
    x=kmeans1.labels_[i]
    y=kmeans1.labels_[i+1]
    changes1[x,y]+=1
    points1[x]+=1
points1[y]+=1

print(changes1)
print(points1)

distmat = np.zeros((nclusters,nclusters))

closestk = 0
for i in range(nclusters):
    for j in range(nclusters):
        distmat[i,j] = np.linalg.norm(kmeans.cluster_centers_[i]-kmeans.cluster_centers_[j])**2

print(distmat)
fig, ax = plt.subplots()
im = ax.pcolormesh(distmat)
fig.colorbar(im)
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "eucdist.svg"))

plt.step(np.arange(16749), kmeans.labels_[0:16749])
plt.yticks(range(nclusters))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "hypnogram.svg"))

plt.bar(range(0,nclusters),points1)
plt.xlabel("Clusters")
plt.xticks(range(0,nclusters))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "superclusterbar1.svg"))

print(len(kmeans.cluster_centers_))

fig, ax = plt.subplots()

n=3

mne.viz.plot_topomap(kmeans1.cluster_centers_[n], xypos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "1supercluster" + str(n) + ".svg"))
# n+=1

n=0

fig, ax = plt.subplots()
im = ax.pcolormesh(kmeans.cluster_centers_[n].reshape(19,-1))
fig.colorbar(im)
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "squarecluster" + str(n) +".svg"))
n+=1

print(kmeans.cluster_centers_[0])

print(len(kmeans.labels_))

print(usedlen)

n=0
hc=0
pa=0
curlen = 0
hclabels = []
nplabels = []
# for file in powerslist:
# for file in diaglist:
for file in silist:
    if n == n:
        try:
            np.isnan(diseasestates[n])
            n+=0
            if hc < 62:
                hclabels.append(kmeans.labels_[curlen:curlen+usedlen[n]])
                curlen=curlen+usedlen[n]+1
                hc+=1
        except TypeError:
            n+=0
            # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            if diseasestates[n] == 'CBP':
                if pa < 33:
                    nplabels.append(kmeans.labels_[curlen:curlen+usedlen[n]])
                    curlen=curlen+usedlen[n]+1
                    pa+=1
        n+=1


print(hclabels)
# print(nplabels)

flat_hc = []
flat_np = []


for xs in hclabels:
    for x in xs:
        flat_hc.append(int(x))

flat_list = []

for xs in nplabels:
    for x in xs:
        flat_np.append(int(x))

dwell0 = []
dwell1 = []
dwell2 = []
dwell3 = []

i=0
n=0
lastn=1000
combo=0
while i < len(flat_np):
    n=flat_np[i]
    if n != lastn:
        if lastn == 0:
            dwell0.append(combo)
        if lastn == 1:
            dwell1.append(combo)
        if lastn == 2:
            dwell2.append(combo)
        if lastn == 3:
            dwell3.append(combo)
        combo = 0

    combo+=1
    i+=1
    lastn=n

dwain = [np.mean(dwell0), np.mean(dwell1), np.mean(dwell2),np.mean(dwell3)]


plt.bar(range(0,nclusters),dwain)
plt.title("Pain Dwell Times")
plt.xticks(range(nclusters))
plt.yticks(range(0,7))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "paindwell.svg"))

dwell0 = []
dwell1 = []
dwell2 = []
dwell3 = []

i=0
n=0
lastn=1000
combo=0
while i < len(flat_hc):
    n=flat_hc[i]
    if n != lastn:
        if lastn == 0:
            dwell0.append(combo)
        if lastn == 1:
            dwell1.append(combo)
        if lastn == 2:
            dwell2.append(combo)
        if lastn == 3:
            dwell3.append(combo)
        combo = 0

    combo+=1
    i+=1
    lastn=n


dwelthy = [np.mean(dwell0), np.mean(dwell1), np.mean(dwell2),np.mean(dwell3)]


plt.bar(range(0,nclusters),dwelthy)
plt.title("Healthy Dwell Times")
plt.xticks(range(nclusters))
plt.yticks(range(0,7))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "healthydwell.svg"))

hcchanges=np.zeros((nclusters,nclusters), dtype=int)
hcpoints=np.zeros((nclusters,), dtype=int)

for i in range(len(flat_hc)-1):
    x=flat_hc[i]
    y=flat_hc[i+1]
    hcchanges[x,y]+=1
    hcpoints[x]+=1
hcpoints[y]+=1


print(hcpoints)

hctrans = []
for i in hcchanges:
    print(np.round(i/np.sum(i),2))
    hctrans.append(i/np.sum(i))

hctrans = np.asarray(hctrans)

plt.bar(range(0,nclusters),hcpoints)
plt.xlabel("Clusters")
# plt.xlim(0,nclusters)
plt.xticks(range(nclusters))
plt.title("Healthy Cluster Bar")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "hccluster.svg"))

npchanges=np.zeros((nclusters,nclusters), dtype=int)
nppoints=np.zeros((nclusters,), dtype=int)

for i in range(len(flat_np)-1):
    x=flat_np[i]
    y=flat_np[i+1]
    npchanges[x,y]+=1
    nppoints[x]+=1
nppoints[y]+=1

print(npchanges)
# /len(flat_np))
print(nppoints)

nptrans = []
for i in npchanges:
    print(np.round(i/np.sum(i),2))
    nptrans.append(i/np.sum(i))

nptrans = np.asarray(nptrans)

plt.bar(range(0,nclusters),nppoints)
plt.xlabel("Clusters")
plt.xticks(range(nclusters))
plt.title("Back Pain Cluster Bar")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "npclusterbar.svg"))

print(energy.shape)

print(testtrans.shape)



n=0
hc=0
pa=0
curlen = 0
hctest = []
nptest = []

hchc = []
hcpa = []

# for file in datalist:
# for file in powerslist:
# for file in diaglist:
for file in silist:
    if n == n:
        flattest = []
        try:
            np.isnan(diseasestates[n])
            n+=0
            if hc > 62:
                energy = pd.read_csv(file)
                energy = energy.to_numpy()
                # energy = np.load(file).transpose()
                energy = normalize_matrix(energy)
                # energy = np.load(file)
                # energy = energy.reshape(energy.shape[0],-1)
                for i in energy:
                    closest = np.linalg.norm(i-kmeans.cluster_centers_[0])**2
                    closestk = 0
                    for k in range(nclusters):
                        if np.linalg.norm(i-kmeans.cluster_centers_[k])**2 < closest:
                            closest = np.linalg.norm(i-kmeans.cluster_centers_[k])**2
                            closestk = k
                    flattest.append(closestk)
                testchanges=np.zeros((nclusters,nclusters), dtype=int)

                for l in range(len(flattest)-1):
                    x=flattest[l]
                    y=flattest[l+1]
                    testchanges[x,y]+=1
                testtrans = []
                for j in testchanges:
                    # print(i)
                    testtrans.append(j/np.sum(i))
                    # print(np.sum(i))

                testtrans = np.asarray(testtrans)
                # hchc.append(np.linalg.norm(testtrans - hctrans)**2)
                # hcpa.append(np.linalg.norm(testtrans - nptrans)**2)
                # hchc.append(np.sum(np.matmul(testtrans, np.linalg.inv(hctrans))))
                # hcpa.append(np.sum(np.matmul(testtrans, np.linalg.inv(nptrans))))
                hchc.append(stats.pearsonr(testtrans.flatten(), hctrans.flatten())[0])
                hcpa.append(stats.pearsonr(testtrans.flatten(), nptrans.flatten())[0])





            hc+=1
        except TypeError:
            n+=0
            # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            if diseasestates[n] == 'CBP':
                if pa > 33:
                    energy = pd.read_csv(file)
                    energy = energy.to_numpy()
                    # energy = np.load(file).transpose()
                    # energy = np.load(file)
                    # energy = energy.reshape(energy.shape[0],-1)
                    energy = normalize_matrix(energy)
                    for i in energy:
                        closest = np.linalg.norm(i-kmeans.cluster_centers_[0])**2
                        closestk = 0
                        for k in range(nclusters):
                            if np.linalg.norm(i-kmeans.cluster_centers_[k])**2 < closest:
                                closest = np.linalg.norm(i-kmeans.cluster_centers_[k])**2
                                closestk = k
                        flattest.append(closestk)

                    testchanges=np.zeros((nclusters,nclusters), dtype=int)

                    for l in range(len(flattest)-1):
                        x=flattest[l]
                        y=flattest[l+1]
                        testchanges[x,y]+=1
                    testtrans = []
                    for j in testchanges:
                        testtrans.append(j/np.sum(i))
                        print(j)

                        # print(np.sum(i))
                    testtrans = np.asarray(testtrans)
                    # pahc.append(np.linalg.norm(testtrans - hctrans)**2)
                    # papa.append(np.linalg.norm(testtrans - nptrans)**2)
                    # pahc.append(np.sum(np.matmul(testtrans, np.linalg.inv(hctrans))))
                    # papa.append(np.sum(np.matmul(testtrans, np.linalg.inv(nptrans))))
                    pahc.append(stats.pearsonr(testtrans.flatten(), hctrans.flatten())[0])
                    papa.append(stats.pearsonr(testtrans.flatten(), nptrans.flatten())[0])
                pa+=1
        n+=1


n=0
hc=0
pa=0
curlen = 0
hctest = []
nptest = []

hchc = []
hcpa = []
pahc = []
papa = []
# for file in datalist:
# for file in powerslist:
# for file in diaglist:
for file in silist:
    if n == n:
        flattest = []
        try:
            np.isnan(diseasestates[n])
            n+=0
            if hc > 62:
                energy = pd.read_csv(file)
                energy = energy.to_numpy()
                # energy = np.load(file).transpose()
                energy = normalize_matrix(energy)
                # energy = np.load(file)
                # energy = energy.reshape(energy.shape[0],-1)
                for i in energy:
                    closest = np.linalg.norm(i-kmeans.cluster_centers_[0])**2
                    closestk = 0
                    for k in range(nclusters):
                        if np.linalg.norm(i-kmeans.cluster_centers_[k])**2 < closest:
                            closest = np.linalg.norm(i-kmeans.cluster_centers_[k])**2
                            closestk = k
                    flattest.append(closestk)
                testchanges=np.zeros((nclusters,nclusters), dtype=int)

                for l in range(len(flattest)-1):
                    x=flattest[l]
                    y=flattest[l+1]
                    testchanges[x,y]+=1
                testtrans = []
                for j in testchanges:
                    # print(i)
                    testtrans.append(j/np.sum(i))
                    # print(np.sum(i))

                testtrans = np.asarray(testtrans)
                # hchc.append(np.linalg.norm(testtrans - hctrans)**2)
                # hcpa.append(np.linalg.norm(testtrans - nptrans)**2)
                # hchc.append(np.sum(np.matmul(testtrans, np.linalg.inv(hctrans))))
                # hcpa.append(np.sum(np.matmul(testtrans, np.linalg.inv(nptrans))))
                hchc.append(stats.pearsonr(testtrans.flatten(), hctrans.flatten())[0])
                hcpa.append(stats.pearsonr(testtrans.flatten(), nptrans.flatten())[0])





            hc+=1
        except TypeError:
            n+=0
            # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            if diseasestates[n] == 'CBP':
                if pa > 33:
                    energy = pd.read_csv(file)
                    energy = energy.to_numpy()
                    # energy = np.load(file).transpose()
                    # energy = np.load(file)
                    # energy = energy.reshape(energy.shape[0],-1)
                    energy = normalize_matrix(energy)
                    for i in energy:
                        closest = np.linalg.norm(i-kmeans.cluster_centers_[0])**2
                        closestk = 0
                        for k in range(nclusters):
                            if np.linalg.norm(i-kmeans.cluster_centers_[k])**2 < closest:
                                closest = np.linalg.norm(i-kmeans.cluster_centers_[k])**2
                                closestk = k
                        flattest.append(closestk)

                    testchanges=np.zeros((nclusters,nclusters), dtype=int)

                    for l in range(len(flattest)-1):
                        x=flattest[l]
                        y=flattest[l+1]
                        testchanges[x,y]+=1
                    testtrans = []
                    for j in testchanges:
                        testtrans.append(j/np.sum(i))
                        print(j)

                        # print(np.sum(i))
                    testtrans = np.asarray(testtrans)
                    # pahc.append(np.linalg.norm(testtrans - hctrans)**2)
                    # papa.append(np.linalg.norm(testtrans - nptrans)**2)
                    # pahc.append(np.sum(np.matmul(testtrans, np.linalg.inv(hctrans))))
                    # papa.append(np.sum(np.matmul(testtrans, np.linalg.inv(nptrans))))
                    pahc.append(stats.pearsonr(testtrans.flatten(), hctrans.flatten())[0])
                    papa.append(stats.pearsonr(testtrans.flatten(), nptrans.flatten())[0])
                pa+=1
        n+=1


n=0
hc=0
pa=0
curlen = 0
hctest = []
nptest = []

hchc = []
hcpa = []
pahc = []
papa = []
# for file in datalist:
# for file in powerslist:
# for file in diaglist:
for file in silist:
    if n == n:
        flattest = []
        try:
            np.isnan(diseasestates[n])
            n+=0
            if hc > 62:
                energy = pd.read_csv(file)
                energy = energy.to_numpy()
                # energy = np.load(file).transpose()
                energy = normalize_matrix(energy)
                # energy = np.load(file)
                # energy = energy.reshape(energy.shape[0],-1)
                for i in energy:
                    closest = np.linalg.norm(i-kmeans.cluster_centers_[0])**2
                    closestk = 0
                    for k in range(nclusters):
                        if np.linalg.norm(i-kmeans.cluster_centers_[k])**2 < closest:
                            closest = np.linalg.norm(i-kmeans.cluster_centers_[k])**2
                            closestk = k
                    flattest.append(closestk)
                testchanges=np.zeros((nclusters,nclusters), dtype=int)

                for l in range(len(flattest)-1):
                    x=flattest[l]
                    y=flattest[l+1]
                    testchanges[x,y]+=1
                testtrans = []
                for j in testchanges:
                    # print(i)
                    testtrans.append(j/np.sum(i))
                    # print(np.sum(i))

                testtrans = np.asarray(testtrans)
                # hchc.append(np.linalg.norm(testtrans - hctrans)**2)
                # hcpa.append(np.linalg.norm(testtrans - nptrans)**2)
                # hchc.append(np.sum(np.matmul(testtrans, np.linalg.inv(hctrans))))
                # hcpa.append(np.sum(np.matmul(testtrans, np.linalg.inv(nptrans))))
                hchc.append(stats.pearsonr(testtrans.flatten(), hctrans.flatten())[0])
                hcpa.append(stats.pearsonr(testtrans.flatten(), nptrans.flatten())[0])





            hc+=1
        except TypeError:
            n+=0
            # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            if diseasestates[n] == 'CBP':
                if pa > 33:
                    energy = pd.read_csv(file)
                    energy = energy.to_numpy()
                    # energy = np.load(file).transpose()
                    # energy = np.load(file)
                    # energy = energy.reshape(energy.shape[0],-1)
                    energy = normalize_matrix(energy)
                    for i in energy:
                        closest = np.linalg.norm(i-kmeans.cluster_centers_[0])**2
                        closestk = 0
                        for k in range(nclusters):
                            if np.linalg.norm(i-kmeans.cluster_centers_[k])**2 < closest:
                                closest = np.linalg.norm(i-kmeans.cluster_centers_[k])**2
                                closestk = k
                        flattest.append(closestk)

                    testchanges=np.zeros((nclusters,nclusters), dtype=int)

                    for l in range(len(flattest)-1):
                        x=flattest[l]
                        y=flattest[l+1]
                        testchanges[x,y]+=1
                    testtrans = []
                    for j in testchanges:
                        testtrans.append(j/np.sum(i))
                        print(j)

                        # print(np.sum(i))
                    testtrans = np.asarray(testtrans)
                    # pahc.append(np.linalg.norm(testtrans - hctrans)**2)
                    # papa.append(np.linalg.norm(testtrans - nptrans)**2)
                    # pahc.append(np.sum(np.matmul(testtrans, np.linalg.inv(hctrans))))
                    # papa.append(np.sum(np.matmul(testtrans, np.linalg.inv(nptrans))))
                    pahc.append(stats.pearsonr(testtrans.flatten(), hctrans.flatten())[0])
                    papa.append(stats.pearsonr(testtrans.flatten(), nptrans.flatten())[0])
                pa+=1
        n+=1


n=0
hc=0
pa=0
curlen = 0
hctest = []
nptest = []

hchc = []
hcpa = []
pahc = []
papa = []
for file in datalist:
    # for file in powerslist:
    if n == n:
        flattest = []
        try:
            np.isnan(diseasestates[n])
            n+=0
            if hc > 62:
                # energy = pd.read_csv(file)
                # energy = energy.to_numpy()
                energy = np.load(file).transpose()
                # energy = normalize_matrix(energy)
                dwell0 = []
                dwell1 = []
                dwell2 = []
                dwell3 = []

                i=0
                n=0
                lastn=1000
                combo=0
                while i < len(flat_hc):
                    n=flat_hc[i]
                    if n != lastn:
                        if lastn == 0:
                            dwell0.append(combo)
                        if lastn == 1:
                            dwell1.append(combo)
                        if lastn == 2:
                            dwell2.append(combo)
                        if lastn == 3:
                            dwell3.append(combo)
                        combo = 0

                    combo+=1
                    i+=1
                    lastn=n
                testtrans=[np.mean(dwell0), np.mean(dwell1), np.mean(dwell2),np.mean(dwell3)]
                testtrans = np.asarray(testtrans)
                dwelthy = np.asarray(dwelthy)
                dwain = np.asarray(dwain)

                hchc.append(np.mean(np.abs(testtrans - dwelthy)))
                hcpa.append(np.mean(np.abs(testtrans - dwain)))





            hc+=1
        except TypeError:
            n+=0
            # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            if diseasestates[n] == 'CBP':
                if pa > 33:
                    # energy = pd.read_csv(file)
                    # energy = energy.to_numpy()
                    energy = np.load(file).transpose()

                    # energy = normalize_matrix(energy)
                    i=0
                    n=0
                    lastn=1000
                    combo=0
                    while i < len(flat_hc):
                        n=flat_hc[i]
                        if n != lastn:
                            if lastn == 0:
                                dwell0.append(combo)
                            if lastn == 1:
                                dwell1.append(combo)
                            if lastn == 2:
                                dwell2.append(combo)
                            if lastn == 3:
                                dwell3.append(combo)
                            combo = 0

                        combo+=1
                        i+=1
                        lastn=n
                    testtrans=[np.mean(dwell0), np.mean(dwell1), np.mean(dwell2),np.mean(dwell3)]
                    testtrans = np.asarray(testtrans)
                    dwelthy = np.asarray(dwelthy)
                    dwain = np.asarray(dwain)


                    pahc.append(np.mean(np.abs(testtrans - dwelthy)))
                    papa.append(np.mean(np.abs(testtrans - dwain)))

                pa+=1
        n+=1


indexes = [1,3,5,12,15]
n=0
hc=0
pa=0
curlen = 0
hctest = []
nptest = []

hchc = []
hcpa = []
pahc = []
papa = []
# for file in datalist:
# for file in powerslist:
for file in diaglist:
    if n == n:
        flattest = []
        try:
            np.isnan(diseasestates[n])
            n+=0
            if hc > 62:
                # energy = pd.read_csv(file)
                # energy = energy.to_numpy()
                # energy = normalize_matrix(energy)
                # energy = np.load(file).transpose()
                energy = np.load(file)
                energy = energy.reshape(energy.shape[0],-1)

                for i in energy:
                    closest = np.linalg.norm(i-kmeans.cluster_centers_[0])**2
                    closestk = 0
                    for k in range(nclusters):
                        if np.linalg.norm(i-kmeans.cluster_centers_[k])**2 < closest:
                            closest = np.linalg.norm(i-kmeans.cluster_centers_[k])**2
                            closestk = k
                    flattest.append(closestk)
                testchanges=np.zeros((nclusters,nclusters), dtype=int)

                for l in range(len(flattest)-1):
                    x=flattest[l]
                    y=flattest[l+1]
                    testchanges[x,y]+=1
                testtrans = []
                for j in testchanges:
                    # print(i)
                    testtrans.append(j/np.sum(i))
                    # print(np.sum(i))

                testtrans = np.asarray(testtrans)
                # hchc.append(np.linalg.norm(testtrans - hctrans)**2)
                # hcpa.append(np.linalg.norm(testtrans - nptrans)**2)
                # hchc.append(np.sum(np.matmul(testtrans, np.linalg.inv(hctrans))))
                # hcpa.append(np.sum(np.matmul(testtrans, np.linalg.inv(nptrans))))
                hchc.append(stats.pearsonr([testtrans.flatten()[x] for x in indexes], [hctrans.flatten()[x] for x in indexes])[0])
                hcpa.append(stats.pearsonr([testtrans.flatten()[x] for x in indexes], [nptrans.flatten()[x] for x in indexes])[0])





            hc+=1
        except TypeError:
            n+=0
            # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            if diseasestates[n] == 'CBP':
                if pa > 33:
                    # energy = pd.read_csv(file)
                    # energy = energy.to_numpy()
                    # energy = normalize_matrix(energy)
                    # energy = np.load(file).transpose()
                    energy = np.load(file)
                    energy = energy.reshape(energy.shape[0],-1)
                    for i in energy:
                        closest = np.linalg.norm(i-kmeans.cluster_centers_[0])**2
                        closestk = 0
                        for k in range(nclusters):
                            if np.linalg.norm(i-kmeans.cluster_centers_[k])**2 < closest:
                                closest = np.linalg.norm(i-kmeans.cluster_centers_[k])**2
                                closestk = k
                        flattest.append(closestk)

                    testchanges=np.zeros((nclusters,nclusters), dtype=int)

                    for l in range(len(flattest)-1):
                        x=flattest[l]
                        y=flattest[l+1]
                        testchanges[x,y]+=1
                    testtrans = []
                    for j in testchanges:
                        testtrans.append(j/np.sum(i))
                        print(j)

                        # print(np.sum(i))
                    testtrans = np.asarray(testtrans)
                    # pahc.append(np.linalg.norm(testtrans - hctrans)**2)
                    # papa.append(np.linalg.norm(testtrans - nptrans)**2)
                    # pahc.append(np.sum(np.matmul(testtrans, np.linalg.inv(hctrans))))
                    # papa.append(np.sum(np.matmul(testtrans, np.linalg.inv(nptrans))))
                    pahc.append(stats.pearsonr([testtrans.flatten()[x] for x in indexes], [hctrans.flatten()[x] for x in indexes])[0])
                    papa.append(stats.pearsonr([testtrans.flatten()[x] for x in indexes], [nptrans.flatten()[x] for x in indexes])[0])
                pa+=1
        n+=1


print(testtrans.flatten()[main_list[x] for x in indexes])


plt.boxplot((hchc, hcpa, pahc, papa), tick_labels=("healthy healthy", "healthy pain", "pain healthy", "pain pain"), showfliers=True)
plt.title("pearson correlation")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "nothingcorrelation.svg"))

G = nx.DiGraph()

# Add edges with transition probabilities
for i in range(nclusters):
    for j in range(nclusters):
        prob = nptrans[i, j]
        if prob > 0.009:  # Only add edges with non-zero probability
            G.add_edge(range(nclusters)[i], range(nclusters)[j], weight=prob)

# Define layout for visualization
pos = nx.spring_layout(G, seed=42)  # You can try other layouts too

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=150, node_color='lightblue')

# Draw edges
edges = G.edges(data=True)
edge_weights = [f"{d['weight']:.2f}" for (_, _, d) in edges]
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold')

# Draw edge labels (transition probabilities)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("Pain Markov Chain")
plt.axis('off')
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs\\kmeans", "npmarkov.svg"))
plt.show()


print(hchc)#, hcpa, pahc, papa)

# ## Processing Data




print(AMat.shape)

##Processing A Matrix

##mostly stole this whole part from Rina's code with some small changes to make it work for my data and add some other functions
##removes diagonals and runs most of the tests on the a matrix

#initialize lists
statAL_J = []
pvalAL_J = []
statAL_K = []
pvalAL_K = []
meanAL = []
varAL = []
vcondAL = []
mcondAL = []

# file = amatlist[0]
# if file == amatlist[0]:
for file in diaglist:
    #A Matrix Loading of Patient
    # print(file)
    AMat = np.load(file)[1:]

    ##ohhhhhhhhh i think the flatten may be why not normal
    #normality test on A & SI
    #1. Jarque Bera - Statstic = about 0 and pvalue for the following hypotheses (H0 - Normally Distributed, Ha - Not Normally Distributed)
    statA, pvalA = stats.jarque_bera(AMat.flatten())
    statAL_J.append(statA)
    pvalAL_J.append(pvalA)

    #2 Kolmogorov Smirnov - Statstic = about 0 and pvalue for the following hypotheses (H0 - Normally Distributed, Ha - Not Normally Distributed)

    #why do we reference number of patients here this doesnt have any effect on the other ones?
    #is is because you expect less normal results to appear in a larger sample??
    statA, pvalA = stats.kstest(AMat.flatten(), stats.norm.cdf, N = numPatients)
    statAL_K.append(statA)
    pvalAL_K.append(pvalA)

    # print(min(AMat))

    #mean and std of A & SI
    meanAL.append(np.mean(AMat.flatten()))
    varAL.append(np.var(AMat.flatten()))

    cond = np.linalg.cond(AMat)

    cond = np.delete(cond, len(cond)-1)

    outliers = []
    for i in range(len(cond)):
        if cond[i] > 2e18:
            outliers.append(i)
    cond = np.delete(cond, outliers)

    print(len(cond))
    # mean and variance of condition number of A
    vcondAL.append(np.var(cond))
    mcondAL.append(np.mean(cond))

meanALbad = []
for file in amatlist:
    #A Matrix Loading of Patient
    # print(file)
    AMat = np.load(file)[1:]
    meanALbad.append(np.mean(AMat.flatten()))

print(np.linalg.cond(AMat))

plt.plot(range(len(cond)), cond)
# plt.plot(range(AMat.shape[0]), AMat[:,0,0])
# plt.plot(range(data.shape[1]), data[0,:])

meanwdiag = []
meandiag = []

for file in diaglist:
    #A Matrix Loading of Patient
    # print(file)
    diag = np.load(file)[1:]
    meanwdiag.append(np.mean(diag.flatten()))

diaglen = diag.size - AMat.size

i = 0
while i < numPatients:
    totalA = meanALbad[i]*AMat.size
    totalD = meanwdiag[i]*diag.size
    meandiag.append((totalD - totalA)/diaglen)
    i += 1

##processing entropy
mentro = []
ventro = []


for file in entrolist:
    entropyvalues = pd.read_csv(file)
    entropyvalues = entropyvalues.to_numpy()
    mentro.append(np.mean(entropyvalues))
    ventro.append(np.var(entropyvalues))

#0.014433886778617509

##i just realized the mean and var entropy is not of the channels but over the range of the data. otherwise it would just increase

powermeans = []
powerpa = []
powerhc = []


n = 0
for file in powerlist:
    # file = powerlist[0]
    # if file == powerlist[0]:
    powerrow = []
    powervalues = pd.read_csv(file)
    powervalues = powervalues.to_numpy().transpose()
    for i in powervalues:
        powerrow.append(np.mean(i))
    powermeans.append(np.asarray(powerrow))
    # print(len(powerrow))
    try:
        np.isnan(diseasestates[n])
        powerhc.append(np.asarray(powerrow))
    except TypeError:
        powerpa.append(np.asarray(powerrow))
    n+=1


meanofallpowers = np.asarray(powermeans)
powerpa = np.asarray(powerpa)
powerhc = np.asarray(powerhc)


m=0

amatrix=np.load(diaglist[m])
print(subdirlist[m])
n = len([cmap])
fig, axs = plt.subplots(1, n, figsize=(n * 2 * 4 + 2, 3 * 2),
                        layout='constrained', squeeze=False)
for [ax, cmap] in zip(axs.flat, [cmap]):
    psm = ax.pcolormesh(np.max(amatrix[1:], axis = 0), cmap=cmap, rasterized=True)
    fig.colorbar(psm, ax=ax)

    # plt.title("correlation between channels")
# amatview += 1
# plt.show()
# m+=1
plt.title("Max of A-Matrix")
# fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "max_amatrix_second_example.png"))

n = 0
for file in diaglist:
    m=0
    amatrix = np.load(file)

    Connectivity = []

    while m <= 1:
        A_Connections = nx.Graph()
        Connected_Nodes = []

        for i in range(chl):
            for j in range(chl):
                # if ahatswdiag[m, i, j] > 0.3:
                # if np.mean(ahatswdiag[1:, i, j], axis = 0) > 0.1:
                if np.max(amatrix[1:, i, j], axis = 0) > m:

                    A_Connections.add_edges_from([(keep_channels[i], keep_channels[j])])
                    if i != j:
                        if keep_channels[i] not in Connected_Nodes:
                            Connected_Nodes.append(keep_channels[i])
                        if keep_channels[j] not in Connected_Nodes:
                            Connected_Nodes.append(keep_channels[j])
        # nx.shell_layout(A_Connections, scale = 1)
        # plt.figure(1,figsize=(10, 10))
        # nx.draw_networkx(A_Connections, arrows=True, **options)
        # plt.title(str(m) + ' ' + str(len(Connected_Nodes)))
        m+=0.01
        # plt.draw()
        Connectivity.append(len(Connected_Nodes))
    try:
        np.isnan(diseasestates[n])
        plt.plot(np.arange(0, 1, 0.01), Connectivity)
        n+=1
    except TypeError:
        # plt.plot(np.arange(0, 1, 0.01), Connectivity)
        # if diseasestates[n] == 'CBP':
        # plt.plot(np.arange(0, 1, 0.01), Connectivity)
        n+=1

plt.title("Size of A Matrix Graph (Healthy)")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "A_Mat_Size_10-20_Healthy" + ".png"))

options = {
    'node_color': 'blue',
    'node_size': 200,
    'font_size': 6,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 12,
}

amatrix = np.load(diaglist[0])
m=0
while m < 1.01:
    plt.clf()
    plt.cla()
    A_Connections = nx.Graph()
    Connected_Nodes = []

    for i in range(chl):
        for j in range(chl):
            if np.mean(np.abs(amatrix[1:, i, j])) > m:
                A_Connections.add_edges_from([(keep_channels[i], keep_channels[j])])
                if i != j:
                    if keep_channels[i] not in Connected_Nodes:
                        Connected_Nodes.append(keep_channels[i])
                    if keep_channels[j] not in Connected_Nodes:
                        Connected_Nodes.append(keep_channels[j])

    nx.shell_layout(A_Connections, scale = 1)
    plt.figure(1,figsize=(10, 10))
    nx.draw_networkx(A_Connections, arrows=True, **options)

    plt.title(str(np.round(m, 2)) + ' ' + str(len(Connected_Nodes)) + " " + str(len(A_Connections.edges)))
    plt.draw()
    plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "A-Mat_Connect_Means", str(np.round(m, 2)) + '_' + str(len(Connected_Nodes)) + "_" + str(len(A_Connections.edges)) + ".png"))
    m+=0.01

#connections

n = 0
for file in diaglist:
    m=0
    amatrix = np.load(file)

    Connectivity = []

    # while m < 1.01:
    while m < 0.2:
        A_Connections = nx.Graph()
        Connected_Nodes = []

        for i in range(chl):
            for j in range(chl):
                # if ahatswdiag[m, i, j] > 0.3:
                # if np.mean(ahatswdiag[1:, i, j], axis = 0) > 0.1:
                # if np.mean(np.abs(amatrix[1:, i, j]), axis = 0) + np.std(np.abs(amatrix[1:, i, j]), axis = 0) > m:
                if np.mean(np.abs(amatrix[1:, i, j]), axis = 0) > m:
                    A_Connections.add_edges_from([(keep_channels[i], keep_channels[j])])

        m+=0.01
        # plt.draw()
        Connectivity.append(len(A_Connections.edges))
    try:
        np.isnan(diseasestates[n])
        # plt.plot(np.arange(0, 1.01, 0.01), Connectivity, color="slateblue")
        # plt.plot(np.arange(0, 1.01, 0.01), Connectivity)
        # plt.plot(np.arange(0, 0.2, 0.01), Connectivity)
        plt.plot(np.arange(0, 0.2, 0.01), Connectivity, color="slateblue")

        n+=1
    except TypeError:
        # plt.plot(np.arange(0, 1.01, 0.01), Connectivity, color="red")
        # plt.plot(np.arange(0, 1.01, 0.01), Connectivity)
        plt.plot(np.arange(0, 0.2, 0.01), Connectivity, color="red")
        # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
        # plt.plot(np.arange(0, 1.01, 0.01), Connectivity)
        # if diseasestates[n] == 'CBP':
        #     plt.plot(np.arange(0, 1.01, 0.01), Connectivity)
        # if diseasestates[n] == 'CWP':
        #     plt.plot(np.arange(0, 1.01, 0.01), Connectivity)
        n+=1

plt.title("Edges in Mean Abs A-Matrix Graph (Blue:Healthy, Red:Unhealty)")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "A_Mat_Graph_Edges_Mean_Compare_Close" + ".png"))

#healthy
#unhealthy
#Neuropathic SMALL
#Widespread
#back COMPLETE

fig, ax = plt.subplots(1, 1, figsize=(chl, 10), squeeze=False)
# ax[0,0].boxplot(meanofallpowers, showfliers=False)
# ax[0,0].set_title("Power by Channel")
# ax[0,0].set_xticks(np.arange(1, chl+1), keep_channels)
# ax[0,0].set_ylim(bottom=0)

# powplot = ax[0,0].boxplot(meanofallpowers, showfliers=False, patch_artist=True, boxprops=dict(facecolor="red", color="red"))
papos = []

for i in range(0, chl):
    papos.append(i + 0.5)


powplot = ax[0,0].boxplot(powerpa, showfliers=True, patch_artist=True, boxprops=dict(facecolor="red", color="red"), widths=0.25)
powplot1 = ax[0,0].boxplot(powerhc, showfliers=True, patch_artist=True, positions=papos, boxprops=dict(facecolor="slateblue", color="slateblue"), widths=0.25)

ax[0,0].set_title("Mean Energy by Channel (Red=Unhealthy, Blue=Healthy)")
ax[0,0].set_xticks(np.arange(0.75, chl), keep_channels)
ax[0,0].set_ylim(bottom=0)


plt.setp(powplot['medians'], color='black', linewidth=3.0)
plt.setp(powplot1['medians'], color='black', linewidth=3.0)

# n = 0
# for box in powplot['boxes']:
#     if n % 2 != 0:
#         plt.setp(box, color='slateblue')
#     else:
#         plt.setp(box, color='red')
#     n+=1



# plt.setp(powplot['boxes'], color='slateblue')


plt.show()
##had to change number of channels
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "healthyvsunhealthypowers" + ".png"))


print(subdirlist[0])
print(titleString)

print(powers.shape)

powermeans = []
powerpa = []
powerhc = []
n=0
for file in subdirlist:
    dels = []
    powerrow = []
    # keepgfp = np.load(find_files_by_extension(file, "_gfpmaximums.csv.npy")[0])
    powervalues = pd.read_csv(find_files_by_extension(file, 'power_by_channel.csv')[0])
    powervalues = powervalues.to_numpy()
    # for i in range(powervalues.shape[0]):
    #     # if i not in keepgfp:
    #     dels.append(i)
    for i in powervalues.transpose():
        # j = np.delete(i, dels)
        powerrow.append(np.mean(i))
    # powermeans.append(np.asarray(powerrow))
    # print(len(powerrow))
    try:
        np.isnan(diseasestates[n])
        powerhc.append(np.asarray(powerrow))
    except TypeError:
        powerpa.append(np.asarray(powerrow))
    n+=1

# int(np.trunc(i/125))
#horrible name for my files... i should redo the directory

meanofallpowers = np.asarray(powermeans)
powerpa = np.asarray(powerpa)
powerhc = np.asarray(powerhc)

simeans = []
sipa = []
sihc = []
n=0
for file in subdirlist:
    dels = []
    sirow = []
    keepgfp = np.load(find_files_by_extension(file, "_gfpmaximums.csv.npy")[0])
    for i in range(len(keepgfp)):
        keepgfp[i] = int(np.trunc(keepgfp[i]/125))
    sivalues = pd.read_csv(find_files_by_extension(file, 'sink_indices.csv')[0])
    sivalues = sivalues.to_numpy()
    for i in range(sivalues.shape[0]):
        if i not in keepgfp:
            dels.append(i)
    for i in sivalues.transpose():
        j = np.delete(i, dels)
        sirow.append(np.mean(j))
    simeans.append(np.asarray(sirow))
    # print(len(sirow))
    try:
        np.isnan(diseasestates[n])
        sihc.append(np.asarray(sirow))
    except TypeError:
        sipa.append(np.asarray(sirow))
    n+=1

# int(np.trunc(i/125))
#horrible name for my files... i should redo the directory

meanofallsis = np.asarray(simeans)
sipa = np.asarray(sipa)
sihc = np.asarray(sihc)

print(sivalues.shape)
print(sipa.shape)
print(sihc.shape)

simeans = []
sipa = []
sihc = []


n = 0
for file in silist:
    # file = silist[0]
    # if file == silist[0]:
    sirow = []
    sivalues = pd.read_csv(file)
    sivalues = sivalues.to_numpy().transpose()
    for i in sivalues:
        sirow.append(np.mean(i))
    simeans.append(np.asarray(sirow))
    # print(len(sirow))
    try:
        np.isnan(diseasestates[n])
        sihc.append(np.asarray(sirow))
    except TypeError:
        sipa.append(np.asarray(sirow))
    n+=1


meanofallsis = np.asarray(simeans)
sipa = np.asarray(sipa)
sihc = np.asarray(sihc)


# xypos = np.asarray(pd.read_table(mainfolder + 'Data/eeg/' + titleString1 + '/eeg/' + titleString1 + '_electrodes.tsv'))
# dels = []
# for i in range(xypos.shape[0]):
#     if xypos[i,0] not in keep_channels:
#         dels.append(i)
# xypos = np.delete(xypos, np.asarray(dels), axis = 0)
# xypos = xypos[:,1:3]
# xypos = xypos.astype('float64')

fig, ax = plt.subplots()
mne.viz.plot_topomap(np.mean(powerhc, axis = 0), xypos, size = 2, axes = ax, cmap = 'RdBu_r')
# mne.viz.plot_topomap(np.mean(sihc, axis = 0), xypos, size = 2, axes = ax, vlim = (0, 1), cmap = 'RdBu_r')
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "topomaphealthy_power_10-20" + ".svg"))

# xypos = np.asarray(pd.read_table(mainfolder + 'Data/eeg/' + titleString1 + '/eeg/' + titleString1 + '_electrodes.tsv'))
# dels = []
# for i in range(xypos.shape[0]):
#     if xypos[i,0] not in keep_channels:
#         dels.append(i)
# xypos = np.delete(xypos, np.asarray(dels), axis = 0)
# xypos = xypos[:,1:3]
# xypos = xypos.astype('float64')

fig, ax = plt.subplots()
mne.viz.plot_topomap(np.mean(powerpa, axis = 0), xypos, size = 2, axes = ax, cmap = 'RdBu_r')
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs","topomapunhealthy_power_10-20" + ".svg"))

fig, ax = plt.subplots(1, 1, figsize=(chl, 10), squeeze=False)
# ax[0,0].boxplot(meanofallsis, showfliers=False)
# ax[0,0].set_title("si by Channel")
# ax[0,0].set_xticks(np.arange(1, chl+1), keep_channels)
# ax[0,0].set_ylim(bottom=0)

# siplot = ax[0,0].boxplot(meanofallsis, showfliers=False, patch_artist=True, boxprops=dict(facecolor="red", color="red"))
papos = []

for i in range(0, chl):
    papos.append(i + 0.5)


siplot = ax[0,0].boxplot(sipa, showfliers=True, patch_artist=True, boxprops=dict(facecolor="red", color="red"), widths=0.25)
siplot1 = ax[0,0].boxplot(sihc, showfliers=True, patch_artist=True, positions=papos, boxprops=dict(facecolor="slateblue", color="slateblue"), widths=0.25)

ax[0,0].set_title("Local SI Values (Red=Unhealthy, Blue=Healthy)")
ax[0,0].set_xticks(np.arange(0.75, chl), range(chl))
ax[0,0].set_ylim(bottom=0)


plt.setp(siplot['medians'], color='black', linewidth=3.0)
plt.setp(siplot1['medians'], color='black', linewidth=3.0)

# n = 0
# for box in siplot['boxes']:
#     if n % 2 != 0:
#         plt.setp(box, color='slateblue')
#     else:
#         plt.setp(box, color='red')
#     n+=1



# plt.setp(siplot['boxes'], color='slateblue')


plt.show()
##had to change number of channels
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "healthyvsunhealthysis_1020" + ".png"))


# ## SVD stuff


n = 0
for file in SVDlist:
    # for file in SVDlist[0:10]:
    # file = SVDlist[n]
    # if file == SVDlist[n]:
    SVDpercent = []
    SVDvalues = pd.read_csv(file)
    SVDvalues = SVDvalues.to_numpy().transpose()
    for i in SVDvalues:
        x = 0
        for j in i:
            if j < 1e-15:
                x += 1
        SVDpercent.append(np.trunc(np.round(x/len(i)*100)))
    # try:
    #     np.isnan(diseasestates[n])
    #     plt.plot(range(33, 62), SVDpercent[33:62], color = "slateblue")
    # except TypeError:
    #     n+=0
    #     plt.plot(range(33, 62), SVDpercent[33:62], color = "red")
    # n+=1
    # try:
    #     np.isnan(diseasestates[n])
    #     plt.plot(SVDpercent, color = "slateblue")
    # except TypeError:
    #     n+=0
    #     plt.plot(SVDpercent, color = "red")
    # n+=1
    # try:
    #     np.isnan(diseasestates[n])
    #     plt.plot(range(25, 50), SVDpercent[25:50], color = "slateblue")
    # except TypeError:
    #     n+=0
    #     plt.plot(range(25, 50), SVDpercent[25:50], color = "red")
    # n+=1

# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "SVD_individual_curves_5e-10" + ".png"))


print(diseasestates)

n = 0
for file in SVDlist:
    # for file in SVDlist[12:112]:
    # file = SVDlist[n]
    # if file == SVDlist[n]:
    SVDindimean = []
    SVDvalues = pd.read_csv(file)
    SVDvalues = SVDvalues.to_numpy().transpose()
    for i in SVDvalues:
        SVDindimean.append(np.mean(i))
    # try:
    #     np.isnan(diseasestates[n])
    #     plt.plot(range(12, chl), SVDindimean[12:chl], color = "slateblue")
    # except TypeError:
    #     n+=12
    #     plt.plot(range(12, chl), SVDindimean[12:chl], color = "red")
    # n+=1
    # try:
    #     np.isnan(diseasestates[n])
    #     plt.plot(SVDindimean, color = "slateblue")
    # except TypeError:
    #     n+=12
    #     plt.plot(SVDindimean, color = "red")
    # n+=1
    try:
        np.isnan(diseasestates[n])
        # plt.plot(range(12, chl), SVDindimean[12:chl], color = "slateblue")
        n+=0
    except TypeError:
        n+=0
        # plt.plot(range(12, chl), SVDindimean[12:chl], color = "red")

        if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            plt.plot(range(12, chl), SVDindimean[12:chl], color = "red")
        elif diseasestates[n] == 'CBP':
            plt.plot(range(12, chl), SVDindimean[12:chl], color = "violet")
        elif diseasestates[n] == 'CWP':
            plt.plot(range(12, chl), SVDindimean[12:chl], color = "darkgreen")
        elif diseasestates[n] == 'JP':
            plt.plot(range(12, chl), SVDindimean[12:chl], color = "lawngreen")
        # if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
        #     plt.plot(range(12, chl), SVDindimean[12:chl], color = "red", label="neuropathic")
        # elif diseasestates[n] == 'CBP':
        #     plt.plot(range(12, chl), SVDindimean[12:chl], color = "violet", label="back")
        # elif diseasestates[n] == 'CWP':
        #     plt.plot(range(12, chl), SVDindimean[12:chl], color = "darkgreen", label="widespread")
        # elif diseasestates[n] == 'JP':
        #     plt.plot(range(12, chl), SVDindimean[12:chl], color = "lawngreen", label="joint")


    n+=1
# plt.title('Red:Unhealthy, Blue:Healthy')
plt.title("red:neuro,back:violet,widespread:green,joint:lime")
# plt.

# plt.ylim(top=2e-12)
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "SVD_individual_curves_mean_unhealthytypes_10-20" + ".png"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "SVD_individual_curves_mean_compare_10-20" + ".png"))


n = 0
for file in SVDlist:
    # for file in SVDlist[0:10]:
    # file = SVDlist[n]
    # if file == SVDlist[n]:
    SVDindimean = []
    SVDvalues = pd.read_csv(file)
    SVDvalues = SVDvalues.to_numpy().transpose()
    for i in SVDvalues:
        SVDindimean.append(np.mean(i))
    # try:
    #     np.isnan(diseasestates[n])
    #     plt.plot(range(33, 62), SVDindimean[33:62], color = "slateblue")
    # except TypeError:
    #     n+=0
    #     plt.plot(range(33, 62), SVDindimean[33:62], color = "red")
    # n+=1
    # try:
    #     np.isnan(diseasestates[n])
    #     plt.plot(SVDindimean, color = "slateblue")
    # except TypeError:
    #     n+=0
    #     plt.plot(SVDindimean, color = "red")
    # n+=1
    try:
        np.isnan(diseasestates[n])
        plt.plot(range(35, 58), SVDindimean[35:58], color = "slateblue")
        n+=0
    except TypeError:
        if diseasestates[n] == 'CBP':
            plt.plot(range(35, 58), SVDindimean[35:58], color = "red")

    n+=1
plt.title('Red:Back, Blue:Healthy')


# plt.ylim(top=3e-10)
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "SVD_Curves_CBPvsHC" + ".png"))


print(meanofallSVDs.shape)

SVDmeans = []
SVDpa = []
SVDhc = []
SVDnp = []
SVDcbp = []
SVDcwp = []
SVDjp = []

n = 0
for file in SVDlist:
    # file = SVDlist[0]
    # if file == SVDlist[0]:
    SVDrow = []
    SVDvalues = pd.read_csv(file)
    SVDvalues = SVDvalues.to_numpy().transpose()
    for i in SVDvalues:
        SVDrow.append(np.mean(i))
    SVDmeans.append(np.asarray(SVDrow))
    # print(len(SVDrow))
    try:
        np.isnan(diseasestates[n])
        SVDhc.append(np.asarray(SVDrow))
    except TypeError:
        SVDpa.append(np.asarray(SVDrow))
        if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            SVDnp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'CBP':
            SVDcbp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'CWP':
            SVDcwp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'JP':
            SVDjp.append(np.asarray(SVDrow))
    n+=1
    # except TypeError:
    #     if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
    #         SVDnp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'CBP':
    #         SVDcbp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'CWP':
    #         SVDcwp.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'JP':
    #         SVDjp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    # n+=1

meanofallSVDs = np.asarray(SVDmeans)
SVDpa = np.asarray(SVDpa)
SVDhc = np.asarray(SVDhc)
SVDnp = np.asarray(SVDnp)
SVDcbp = np.asarray(SVDcbp)
SVDcwp = np.asarray(SVDcwp)
SVDjp = np.asarray(SVDjp)

print(meanofallSVDs.shape)

nonzeros = []
for i in meanofallSVDs.transpose():
    n=189
    for j in i:
        if j < 1e-15:
            n-=1
    nonzeros.append(n)

dSVDcbp = np.zeros((47,61), dtype=float)
dSVDhc = np.zeros((88, 61), dtype=float)
for i in range(61):
    dSVDcbp[:,i]=SVDcbp[:,i]-SVDcbp[:,i+1]
    dSVDhc[:,i]=SVDhc[:,i]-SVDhc[:,i+1]


dcbpmax = []
dhcmax = []
for i in range(dSVDcbp.shape[0]):
    # plt.plot(i, color="red")
    for j in range(dSVDcbp.shape[1])[39:]:
        if dSVDcbp[i, j] == max(dSVDcbp[i,39:]):
            # if max(dSVDcbp[i,39:])/np.mean(meanofallSVDs[:,j]) < 5:
            dcbpmax.append(max(dSVDcbp[i,39:])/(np.sum(meanofallSVDs[:,j])/nonzeros[j]))

for i in range(dSVDhc.shape[0]):
    # plt.plot(i, color="red")
    for j in range(dSVDhc.shape[1])[39:]:
        if dSVDhc[i, j] == max(dSVDhc[i, 39:]):
            # if max(dSVDhc[i,39:])/np.mean(meanofallSVDs[:,j]) < 5:
            dhcmax.append(max(dSVDhc[i,39:])/(np.sum(meanofallSVDs[:,j])/nonzeros[j]))

dcbpmax = np.array(dcbpmax)
dhcmax = np.array(dhcmax)

plt.hist(dcbpmax, bins = 30, color="red", alpha=0.5)
plt.hist(dhcmax, bins = 100, color="slateblue", alpha=0.5)
plt.savefig("weirdmetrichist")

print(np.median(dcbpmax))
print(np.median(dhcmax))
print(np.std(dcbpmax))
print(np.std(dhcmax))
print(np.mean(dcbpmax))
print(np.mean(dhcmax))


plt.boxplot((dcbpmax, dhcmax), showfliers=False, tick_labels = ("back", "healthy"))
plt.title("magnitude of last non-zero singular value divided by mean of nonzero singular values")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "weirdmetric" + ".png"))

print(dSVDcbp.shape)

print(dcbpmax)
print(dhcmax)

fig, ax = plt.subplots(1, 1, figsize=(7, 7), squeeze=False)
# ax[0,0].boxplot(meanofallSVDs, showfliers=False)
# ax[0,0].set_title("SVD by Channel")
# ax[0,0].set_xticks(np.arange(1, chl+1), keep_channels)
# ax[0,0].set_ylim(bottom=0)

# SVDplot = ax[0,0].boxplot(meanofallSVDs, showfliers=False, patch_artist=True, boxprops=dict(facecolor="red", color="red"))
papos = []

for i in range(chl-12):
    papos.append(i + 0.5)


SVDplot = ax[0,0].boxplot(SVDpa[:,12:], showfliers=True, patch_artist=True, boxprops=dict(facecolor="red", color="red"), widths=0.25)
SVDplot1 = ax[0,0].boxplot(SVDhc[:,12:], showfliers=True, patch_artist=True, positions=papos, boxprops=dict(facecolor="slateblue", color="slateblue"), widths=0.25)

ax[0,0].set_title("Local SVD Values (Red=Unhealthy, Blue=Healthy)")
ax[0,0].set_xticks(np.arange(0.75, chl-12), range(12,chl))
ax[0,0].set_ylim(bottom=0)


plt.setp(SVDplot['medians'], color='black', linewidth=3.0)
plt.setp(SVDplot1['medians'], color='black', linewidth=3.0)

# n = 0
# for box in SVDplot['boxes']:
#     if n % 2 != 0:
#         plt.setp(box, color='slateblue')
#     else:
#         plt.setp(box, color='red')
#     n+=1



# plt.setp(SVDplot['boxes'], color='slateblue')


plt.show()
##had to change number of channels
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "healthyvsunhealthySVDs(smaller)_10-20" + ".png"))


print(SVDpa.shape)

dSVDmeans = np.zeros((189, 61))

for i in range(dSVDmeans.shape[1]):
    dSVDmeans[:,i] = meanofallSVDs[:,i] - meanofallSVDs[:,i+1]

SVDpa = []
SVDhc = []
SVDnp = []
SVDcbp = []
SVDcwp = []
SVDjp = []

n = 0
for file in SVDlist:

    SVDrow = dSVDmeans[n]

    try:
        np.isnan(diseasestates[n])
        SVDhc.append(np.asarray(SVDrow))
    except TypeError:
        SVDpa.append(np.asarray(SVDrow))
        if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            SVDnp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'CBP':
            SVDcbp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'CWP':
            SVDcwp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'JP':
            SVDjp.append(np.asarray(SVDrow))
    n+=1
    # except TypeError:
    #     if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
    #         SVDnp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'CBP':
    #         SVDcbp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'CWP':
    #         SVDcwp.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'JP':
    #         SVDjp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    # n+=1

dSVDpa = np.asarray(SVDpa)
dSVDhc = np.asarray(SVDhc)
dSVDnp = np.asarray(SVDnp)
dSVDcbp = np.asarray(SVDcbp)
dSVDcwp = np.asarray(SVDcwp)
dSVDjp = np.asarray(SVDjp)

fig, ax = plt.subplots(1, 1, figsize=(chl-20-25, 15), squeeze=False)
# ax[0,0].boxplot(meanofalldSVDs, showfliers=False)
# ax[0,0].set_title("dSVD by Channel")
# ax[0,0].set_xticks(np.arange(1, chl+1), keep_channels)
# ax[0,0].set_ylim(bottom=0)

# dSVDplot = ax[0,0].boxplot(meanofalldSVDs, showfliers=False, patch_artist=True, boxprops=dict(facecolor="red", color="red"))
papos = []

for i in range(chl-26-1):
    papos.append(i + 0.5)


dSVDplot = ax[0,0].boxplot(dSVDcbp[:,26:], showfliers=True, patch_artist=True, boxprops=dict(facecolor="red", color="red"), widths=0.25)
dSVDplot1 = ax[0,0].boxplot(dSVDhc[:,26:], showfliers=True, patch_artist=True, positions=papos, boxprops=dict(facecolor="slateblue", color="slateblue"), widths=0.25)

ax[0,0].set_title("Derivative of SVDs (Red=Unhealthy, Blue=Healthy)")
ax[0,0].set_xticks(np.arange(0.75, chl-26-1), range(26,chl-1))
ax[0,0].set_ylim(bottom=0)


plt.setp(dSVDplot['medians'], color='black', linewidth=3.0)
plt.setp(dSVDplot1['medians'], color='black', linewidth=3.0)

# n = 0
# for box in dSVDplot['boxes']:
#     if n % 2 != 0:
#         plt.setp(box, color='slateblue')
#     else:
#         plt.setp(box, color='red')
#     n+=1



# plt.setp(dSVDplot['boxes'], color='slateblue')


plt.show()
##had to change number of channels
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "derivative_healthyvsunhealthydSVDs(smaller)" + ".png"))

ddSVDmeans = np.zeros((189, 60))

for i in range(ddSVDmeans.shape[1]):
    ddSVDmeans[:,i] = dSVDmeans[:,i] - dSVDmeans[:,i+1]

SVDpa = []
SVDhc = []
SVDnp = []
SVDcbp = []
SVDcwp = []
SVDjp = []

n = 0
for file in SVDlist:

    SVDrow = ddSVDmeans[n]

    try:
        np.isnan(diseasestates[n])
        SVDhc.append(np.asarray(SVDrow))
    except TypeError:
        SVDpa.append(np.asarray(SVDrow))
        if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
            SVDnp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'CBP':
            SVDcbp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'CWP':
            SVDcwp.append(np.asarray(SVDrow))
        elif diseasestates[n] == 'JP':
            SVDjp.append(np.asarray(SVDrow))
    n+=1
    # except TypeError:
    #     if diseasestates[n] == 'PNP' or diseasestates[n] == 'PHN' or diseasestates[n] == 'NP':
    #         SVDnp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'CBP':
    #         SVDcbp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'CWP':
    #         SVDcwp.append(np.asarray(SVDrow))
    #     elif diseasestates[n] == 'JP':
    #         SVDjp.append(np.asarray(SVDrow))
    #         SVDpa.append(np.asarray(SVDrow))
    # n+=1

ddSVDpa = np.asarray(SVDpa)
ddSVDhc = np.asarray(SVDhc)
ddSVDnp = np.asarray(SVDnp)
ddSVDcbp = np.asarray(SVDcbp)
ddSVDcwp = np.asarray(SVDcwp)
ddSVDjp = np.asarray(SVDjp)

fig, ax = plt.subplots(1, 1, figsize=(chl-20-25, 15), squeeze=False)
# ax[0,0].boxplot(meanofallddSVDs, showfliers=False)
# ax[0,0].set_title("ddSVD by Channel")
# ax[0,0].set_xticks(np.arange(1, chl+1), keep_channels)
# ax[0,0].set_ylim(bottom=0)

# ddSVDplot = ax[0,0].boxplot(meanofallddSVDs, showfliers=False, patch_artist=True, boxprops=dict(facecolor="red", color="red"))
papos = []

for i in range(chl-26-2):
    papos.append(i + 0.5)


ddSVDplot = ax[0,0].boxplot(ddSVDpa[:,26:], showfliers=True, patch_artist=True, boxprops=dict(facecolor="red", color="red"), widths=0.25)
ddSVDplot1 = ax[0,0].boxplot(ddSVDhc[:,26:], showfliers=True, patch_artist=True, positions=papos, boxprops=dict(facecolor="slateblue", color="slateblue"), widths=0.25)

ax[0,0].set_title("2nd Derivative of dSVDs (Red=Unhealthy, Blue=Healthy)")
ax[0,0].set_xticks(np.arange(0.75, chl-26-2), range(26,chl-2))
# ax[0,0].set_ylim(bottom=0)


plt.setp(ddSVDplot['medians'], color='black', linewidth=3.0)
plt.setp(ddSVDplot1['medians'], color='black', linewidth=3.0)

# n = 0
# for box in ddSVDplot['boxes']:
#     if n % 2 != 0:
#         plt.setp(box, color='slateblue')
#     else:
#         plt.setp(box, color='red')
#     n+=1



# plt.setp(ddSVDplot['boxes'], color='slateblue')


plt.show()
##had to change number of channels
# fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "2derivative_healthyvsunhealthyddSVDs(smaller)" + ".png"))

tsvd = []
psvd = []

for i in range(chl):
    t, p = stats.ttest_ind_from_stats(np.mean(SVDpa[:,i]), np.std(SVDpa[:,i]), 101, np.mean(SVDhc[:,i]), np.std(SVDhc[:,i]), 88)
    tsvd.append(t)
    psvd.append(p)

print(SVDcbp.shape)

tsvd = []
psvd = []

for i in range(chl):
    t, p = stats.ttest_ind_from_stats(np.mean(SVDcbp[:,i]), np.std(SVDcbp[:,i]), 47, np.mean(SVDhc[:,i]), np.std(SVDhc[:,i]), 88)
    tsvd.append(t)
    psvd.append(p)

# plt.plot(tsvd, color = "slateblue")
# plt.plot(psvd[40:58], color = "red")
plt.plot(psvd, color = "red")
# plt.hlines([0.05,0.1], 0, 20, color = "slateblue")
plt.hlines([0.001], 0, 18, color = "slateblue")
plt.title("p of null hypothesis across SVD (Back Pain)")
# plt.yticks(ticks=[0, 0.05,0.1, 1])
plt.yticks(ticks=[0.001, 0.04])
# plt.xticks(range(18), range(40,58))
plt.xticks(range(18), range(0,18))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "zoom_cbp_p_SVD_10-20" + ".png"))




eigenmeans = []
eigenpa = []
eigenhc = []


n = 0
for file in eigenlist:
    # file = eigenlist[0]
    # if file == eigenlist[0]:
    eigenrow = []
    eigenvalues = pd.read_csv(file)
    eigenvalues = eigenvalues.to_numpy().transpose()
    for i in eigenvalues:
        eigencol=[]
        for j in i:
            eigencol.append(complex(j).real)
        eigenrow.append(np.mean(eigencol))
    eigenmeans.append(np.asarray(np.mean(eigenrow)))
    # print(len(eigenrow))
    try:
        np.isnan(diseasestates[n])
        eigenhc.append(np.asarray(eigenrow))
    except TypeError:
        eigenpa.append(np.asarray(eigenrow))
    n+=1


meanofalleigens = np.asarray(eigenmeans)
eigenpa = np.asarray(eigenpa)
eigenhc = np.asarray(eigenhc)
# for idx in evals:
#     real = [ele.real for ele in idx]
#     imag = [ele.imag for ele in idx]
#     reals.append(real)
#     imags.append(imag)

# reals = np.asarray(reals)
# imags = np.asarray(imags)

fig, ax = plt.subplots(1, 1, figsize=(chl, 10), squeeze=False)
# ax[0,0].boxplot(meanofalleigens, showfliers=False)
# ax[0,0].set_title("eigen by Channel")
# ax[0,0].set_xticks(np.arange(1, chl+1), keep_channels)
# ax[0,0].set_ylim(bottom=0)

# eigenplot = ax[0,0].boxplot(meanofalleigens, showfliers=False, patch_artist=True, boxprops=dict(facecolor="red", color="red"))
papos = []

for i in range(chl-35):
    papos.append(i + 0.5)


eigenplot = ax[0,0].boxplot(eigenpa[:,35:], showfliers=True, patch_artist=True, boxprops=dict(facecolor="red", color="red"), widths=0.25)
eigenplot1 = ax[0,0].boxplot(eigenhc[:,35:], showfliers=True, patch_artist=True, positions=papos, boxprops=dict(facecolor="slateblue", color="slateblue"), widths=0.25)

ax[0,0].set_title("Local imaginary eigen Values (Red=Unhealthy, Blue=Healthy)")
ax[0,0].set_xticks(np.arange(0.75, chl-35), range(35,chl))



plt.setp(eigenplot['medians'], color='black', linewidth=3.0)
plt.setp(eigenplot1['medians'], color='black', linewidth=3.0)

# n = 0
# for box in eigenplot['boxes']:
#     if n % 2 != 0:
#         plt.setp(box, color='slateblue')
#     else:
#         plt.setp(box, color='red')
#     n+=1



# plt.setp(eigenplot['boxes'], color='slateblue')


plt.show()
##had to change number of channels
# fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "healthyvsunhealthyeigensimag" + ".png"))


n=0
value=50
for i in SVDhc[:,value]:
    if i < 1e-12:
        # print(i)
        n+=1
print(str(n) + ' ' + str(n/88))
n=0
for i in SVDpa[:,value]:
    if i < 1e-12:
        # print(i)
        n+=1
print(str(n) + ' ' + str(n/101))


hcpercents1 = []
for i in SVDhc.transpose():
    n=0
    for j in i:
        if j < 9e-11:
            n+=1
    hcpercents1.append(np.trunc(np.round(n/88*100)))

papercents1 = []
for i in SVDpa.transpose():
    n=0
    for j in i:
        if j < 9e-11:
            n+=1
    papercents1.append(np.trunc(np.round(n/101*100)))

hcpercents2 = []
for i in SVDhc.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    hcpercents2.append(np.trunc(np.round(n/88*100)))

papercents2 = []
for i in SVDpa.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    papercents2.append(np.trunc(np.round(n/101*100)))

hcpercents = []
for i in SVDhc.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    hcpercents.append(np.trunc(np.round(n/88*100)))

cbppercents = []
for i in SVDpa.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    cbppercents.append(np.trunc(np.round(n/101*100)))


# plt.plot(range(12, chl), hcpercents[12:chl], color="slateblue")
# plt.plot(range(12, chl), papercents[12:chl], color="red")

# plt.plot(range(12, chl), hcpercents1[12:chl], color="slateblue")
# plt.plot(range(12, chl), papercents1[12:chl], color="red")

plt.plot(range(12, chl), hcpercents[12:chl], color="slateblue")
plt.plot(range(12, chl), cbppercents[12:chl], color="red")

plt.ylim(bottom=0, top=101)
# plt.xlim(right=61)

plt.title("Percent of Condition = 0 at nth largest singular value")

plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "SVDpercentplotpa_e-15_10-20" + ".png"))

print(meanofallSVDs.shape)

hcpercents = []
for i in SVDhc.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    hcpercents.append(np.trunc(np.round(n/88*100)))

cbppercents = []
for i in SVDcbp.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    cbppercents.append(np.trunc(np.round(n/47*100)))

cwppercents = []
for i in SVDcwp.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    cwppercents.append(np.trunc(np.round(n/30*100)))

nppercents = []
for i in SVDnp.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    nppercents.append(np.trunc(np.round(n/18*100)))

jppercents = []
for i in SVDjp.transpose():
    n=0
    for j in i:
        if j < 1e-15:
            n+=1
    jppercents.append(np.trunc(np.round(n/6*100)))



# plt.plot(range(33, 62), hcpercents[33:62], color="slateblue")
# plt.plot(range(33, 62), papercents[33:62], color="red")

# plt.plot(range(33, 62), hcpercents1[33:62], color="slateblue")
# plt.plot(range(33, 62), papercents1[33:62], color="red")

plt.plot(range(33, 62), hcpercents[33:62], color="slateblue")
plt.plot(range(33, 62), cbppercents[33:62], color="violet")
plt.plot(range(33, 62), cwppercents[33:62], color="darkgreen")
plt.plot(range(33, 62), nppercents[33:62], color="red")
plt.plot(range(33, 62), jppercents[33:62], color="lawngreen")




plt.ylim(bottom=0, top=101)
plt.xlim(right=61)
plt.title('Blue:Healthy, Red:Neuropathic, Violet:Back, Green:Widespread, Lime:Joint')

plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "SVDpercentplotconditions" + ".png"))

percentdif = []
percentdif2 = []

for i in range(chl):
    percentdif.append(papercents[i]-hcpercents[i])
    percentdif2.append(papercents2[i]-hcpercents2[i])


plt.plot(range(33, 62), percentdif[33:62], color="slateblue")
plt.plot(range(33, 62), percentdif2[33:62], color="red")

plt.title("percent ratio: blue = original, red = only discontinuity")

# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "SVDpercentratioplot" + ".png"))

# ## Processing Data


r2 = []
allr = []

for file in r2list:
    r2values = pd.read_csv(file)
    r2values = r2values.to_numpy()[:,0]
    r2.append(np.mean(r2values))
    for i in r2values:
        allr.append(i)

allr = np.asarray(allr)


allr2 = []
for i in allr:
    if i > 0.99995:
        allr2.append(i)
allr = np.asarray(allr2)



plt.hist(allr, bins = 200)
plt.title("r^2")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "rhist" + ".png"))


prob = []
allp = []

for file in plist:
    pvalues = pd.read_csv(file)
    pvalues = pvalues.to_numpy()[:,0]
    prob.append(np.mean(pvalues))
    for i in pvalues:
        allp.append(i)

allp = np.asarray(allp)

plt.hist(allp, bins = 100)
plt.title("p")
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "phist" + ".png"))
# print(pvalues)


##Processing sink indices
##just set up rinas a matrix loop to do sink indexes instead
statSI_J = []
pvalSI_J = []
statSI_K = []
pvalSI_K = []
meanSI = []
varSI = []

for file in silist:
    #A Matrix Loading of Patient
    SI = pd.read_csv(file)
    SI = SI.to_numpy()

    #normality test on A & SI
    #1. Jarque Bera - Statstic = about 0 and pvalue for the following hypotheses (H0 - Normally Distributed, Ha - Not Normally Distributed)
    statSI, pvalSI = stats.jarque_bera(SI.flatten())
    statSI_J.append(statSI)
    pvalSI_J.append(pvalSI)

    #2 Kolmogorov Smirnov - Statstic = about 0 and pvalue for the following hypotheses (H0 - Normally Distributed, Ha - Not Normally Distributed)
    statSI, pvalSI = stats.kstest(SI.flatten(), stats.norm.cdf, N = numPatients)
    statSI_K.append(statSI)
    pvalSI_K.append(pvalSI)

    #mean and std of A & SI
    meanSI.append(np.average(SI.flatten()))
    varSI.append(np.var(SI.flatten()))

channeli = str()
for i in range(chl):
    stri = str(i)
    channeli = channeli + "Pow["+stri+"], "


print(channeli)

##power values / trying kruskal wallis
krusp = []
kruss = []

for file in powerslist:
    Pow = pd.read_csv(file)
    Pow = Pow.to_numpy()

    # Powl = []
    # i = 0
    # while i < Pow.shape[0]:
    #     Powl.append(Pow[i])
    #     i+=i

    #doesnt work with a list of arrays so I can either write a function that does this myself or just list them out manually
    s, p = stats.kruskal(Pow[0], Pow[1], Pow[2], Pow[3], Pow[4], Pow[5], Pow[6], Pow[7], Pow[8], Pow[9], Pow[10], Pow[11], Pow[12], Pow[13], Pow[14], Pow[15], Pow[16], Pow[17], Pow[18])
    # , Pow[19], Pow[20], Pow[21], Pow[22], Pow[23], Pow[24], Pow[25], Pow[26], Pow[27], Pow[28], Pow[29], Pow[30], Pow[31], Pow[32], Pow[33], Pow[34], Pow[35], Pow[36], Pow[37], Pow[38], Pow[39], Pow[40], Pow[41], Pow[42], Pow[43], Pow[44], Pow[45], Pow[46], Pow[47], Pow[48], Pow[49], Pow[50], Pow[51], Pow[52], Pow[53], Pow[54], Pow[55], Pow[56], Pow[57], Pow[58], Pow[59], Pow[60], Pow[61])
    kruss.append(s)
    krusp.append(p)

print(meanAL[0])

from math import log
meanprop = []

for i in range(numPatients):
    # print(str(meandiag[i]) + ' ' + str(meanAL[i]))
    # print(log(meandiag[i]/meanAL[i]))
    meanprop.append(log(np.abs(meandiag[i]/meanAL[i])))

# ## Writing to Excel


#format the data so I can append it to an existing excel spreasheet with openpyxl
xllocation = (mainfolder + "PopulationStats1.xlsx")
chllocation = (mainfolder + "ChannelMeans.xlsx")

xlarray = [patientids, diseasestates, statAL_J, statSI_J, meanAL, varAL, meanSI,
           varSI, mentro, mcondAL, ventro, vcondAL, sex, age, pvalAL_J, pvalSI_J,
           statAL_K, statSI_K, pvalAL_K, pvalSI_K, meanwdiag, meandiag, kruss,
           krusp, meanprop, nvclist, nvplist, currentpain, avgpain, paindur, medquants, pdisq, pdisi, mcgill]

xlarray = np.asarray(xlarray)

pd.DataFrame(xlarray.transpose()).to_csv(mainfolder + "CSV_PopulationStats.csv", header = False, index = False)

# function to make excelwriter more useable
def np_to_excel(location, x, skips, nax):
    cols = range(x.shape[0] + len(skips))
    cols = np.delete(cols, skips)

    nalist = []
    for i in range(x.shape[0]):
        nalist.append('')

    for t in nax:
        for i in t[1]:
            nalist[i] = t[0]

    for i in range(x.shape[0]):
        xlformat = pd.DataFrame(x.transpose()[:, i])
        with pd.ExcelWriter(location, mode = 'a', engine = "openpyxl", if_sheet_exists = 'overlay') as writer:
            xlformat.to_excel(writer, startrow = 1, startcol = cols[i], header = False, index = False, na_rep = nalist[i])
    return

skiplist = []
nacols = [('hc', [1]), (0, [27]), (0, [28]), (0, [29]), (0, [30]), (0, [31]), (0, [32]), (0, [33])]

np_to_excel(xllocation, xlarray, skiplist, nacols)

for file in amatlist[0:2]:
    print(np.load(file)[0])

for file in SVDlist[0:2]:
    data = pd.read_csv(file).to_numpy().transpose()
    print(np.mean(data, axis=1))

# # Plotting Relationships


powerchannel = np.zeros((chl, numPatients))
print(powerchannel.shape)
x=0
for file in silist:
    data = pd.read_csv(file).to_numpy().transpose()
    # data = np.load(file)
    chlmns = np.zeros((chl))
    # print(chlmns.shape)
    for i in range(chl):
        chlmns[i]=np.mean(data[i])
    powerchannel[:,x]=chlmns
    x+=1
print(powerchannel[:,0])



# xlarray = pd.read_csv(mainfolder + "CSV_PopulationStats.csv", header = None,
#                       names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24', 'col25', 'col26'],
#                       dtype = {'col1' : str, 'col23' : np.float64})
xlarray = pd.read_csv(mainfolder + "CSV_PopulationStats.csv", header = None).to_numpy().transpose()
xlarray2=powerchannel
# print(xlarray.dtypes)

folder = 'sipainmap'
datatype = 'sink index'
# folder = 'datapainmap'
# datatype = "data"
# folder = 'energypainmap'
# datatype = 'energy'

import numpy as np
from matplotlib import pyplot as plt

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

x = np.linspace(0, 9, 10)

coeffs = [b,m]
plt.plot(x, PolyCoefficients(x, coeffs))
plt.show()



pos = np.array([montage.get_positions()["ch_pos"][ch][:2] for ch in keep_channels])

print(pos)

dictionary = dict(zip(keep_channels, pos))

print(dictionary)

print(dict(zip(keep_channels,list(range(chl)))))


def amatconnection(slot, pain):
    pain = int(pain)

    #input manually
    if pain <= 16:
        Pain="Low"
    if pain > 16 and pain <= 30:
        Pain="Mid"
    if pain > 30:
        Pain="High"

    File = diaglist[slot]
    amatrix = np.load(File)
    m=0.05
    # while m < 1.01:
    plt.clf()
    plt.cla()
    A_Connections = nx.Graph()
    Connected_Nodes = []
    arrowsizes = []

    samples = [2,11]
    # smpc

    for i in range(chl):
        for j in range(chl):
            if np.mean(np.abs(amatrix[1:, i, j])) > m:
                if i != j:
                    # if i==samples[0] or j==samples[0] or i==samples[1] or j==samples[1]:
                    if i in samples or j in samples:
                        A_Connections.add_edges_from([(keep_channels[i], keep_channels[j])])
                        if len(arrowsizes) < len(A_Connections.edges):
                            arrowsizes.append(100*np.mean(np.abs(amatrix[1:, i, j])))
                        # print(len(arrowsizes))
                        # print(len(A_Connections.edges))
                        if keep_channels[i] not in Connected_Nodes:
                            Connected_Nodes.append(keep_channels[i])
                        if keep_channels[j] not in Connected_Nodes:
                            Connected_Nodes.append(keep_channels[j])
    # print(list(A_Connections.nodes)[0])

    colorlist = ['gray'] * len(A_Connections.nodes)

    for n in range(len(A_Connections.nodes)):
        if list(A_Connections.nodes)[n] in np.asarray(keep_channels)[samples]:
            colorlist[n] = 'red'

    options = {
        'node_color': colorlist,
        'node_size': 200,
        'font_size': 6,
        'width': 1,
        'arrowstyle': '-|>',
        'arrowsize': arrowsizes,
    }


    # print(A_Connections.nodes)

    nx.shell_layout(A_Connections, scale = 1)
    plt.figure(1,figsize=(10, 10))
    nx.draw_networkx(A_Connections, dictionary, arrows=True, **options)

    plt.title(Pain+"_Pain")
    plt.draw()
    plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "painamatinfluence", str(slot)+ "_"+ str(pain) + "_Pain"+ ".png"))
    # m+=0.01
    return

#Ok thanks I looked at the slides.
#Can you add p values to all scatter plots?

#DONE
#Also can you make the topomaps we discussed- slope, p value and

#product?

#DONE
#Finally can you add healthy topomaps- maybe average sink index over population that is demographically matched to the neuropathic group?

#plot topomaps by channel
#focus neuropathic
#different pain scores
#mean sinc index vs each pain scores for neuropathic!!!!
#show on one slide all scatterplots spacially
#for each pain score
#extra with topomap
#pvalue topomap

# pairwise cor pain cond

frontal = []
frontalnp = []
# frontalbox = []

# region = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
# regiontitle = "frontal"

# region = ['T7', 'C3', 'Cz', 'C4', 'T8']
# regiontitle = "central"

# region =['P7', 'P3', 'Cz', 'P4', 'P8', 'O1', 'O2']
# regiontitle = "occipital and parietal"


# n=2
n=-1

slopetopo = []
ptopo = []
cortopo = []
product = []
while n < chl-1:
    n+=1

    healthy = []
    unhealthy = []
    CWP = []
    CBP = []
    JP = []
    NP = []

    for i in range(numPatients):
        try:
            np.isnan(xlarray[1, i])
            healthy.append(xlarray2[n, i])
        except TypeError:
            unhealthy.append(xlarray2[n, i])
            match xlarray[1, i]:
                case "CWP":
                    CWP.append(xlarray2[n, i])
                case "CBP":
                    CBP.append(xlarray2[n, i])
                case "PNP":
                    NP.append(xlarray2[n, i])
                    # print(i)
                    # print(xlarray2[n, i])
                case "NP":
                    NP.append(xlarray2[n, i])
                    # print(i)
                    # print(xlarray2[n, i])
                case "JP":
                    JP.append(xlarray2[n, i])
                case "PHN":
                    NP.append(xlarray2[n, i])
                    # print(i)
                    # print(xlarray2[n, i])

    # hcsink.append(np.mean(healthy))
    # npsink.append(np.mean(NP))
    # healthy = removeoutliers(healthy)
    # unhealthy = removeoutliers(unhealthy)
    # CBP = removeoutliers(CBP)
    # CWP = removeoutliers(CWP)
    # NP = removeoutliers(NP)
    # JP = removeoutliers(JP)

    # plt.boxplot(x=(healthy, CBP), tick_labels = ('healthy', 'back'))

    # plt.boxplot(x=(healthy, unhealthy, CBP, CWP, JP, NP), tick_labels = ('healthy', 'unhealthy', 'back', 'widespread', 'joint', 'neuropathic'), autorange = True)
    # fig, ax = plt.subplots(1,1)
    # ax.boxplot((healthy, unhealthy, CBP, CWP, NP), 0, showfliers=True, tick_labels = ('healthy', 'unhealthy', 'back', 'widespread', 'neuropathic'))
    # ax.boxplot((healthy, unhealthy), 0, showfliers=False, tick_labels = ('healthy', 'unhealthy'))

    if keep_channels[n] in region:

        frontal.append(NP)
        frontalnp.append(NPain)




    y = n





    # NP = np.log(NP)


    # plt.scatter(CBPain, CBP)
    # plt.scatter(CWPain, CWP)
    plt.scatter(NPain, NP)
    # plt.scatter(HCain, healthy)
    plt.xlabel(heads[x])
    plt.ylabel("Mean sink of " + keep_channels[n])
    plt.title("Mean sink of " + keep_channels[n] + " vs " + heads[x])
    # plt.legend(('back', 'widespread', 'neuropathic', 'healthy'))
    # plt.legend(('back', 'widespread', 'neuropathic'))
    plt.legend(('neuropathic',))


    # plt.legend(('back', 'neuropathic'))

    # m,b=np.polyfit(CBPain, CBP, 1)
    # linspace = np.linspace(min(CBPain),max(CBPain))
    # coeffs = [b,m]
    # plt.plot(linspace, PolyCoefficients(linspace,coeffs))

    # m,b=np.polyfit(CWPain, CWP, 1)
    # linspace = np.linspace(min(CWPain),max(CWPain))
    # coeffs = [b,m]
    # plt.plot(linspace, PolyCoefficients(linspace,coeffs))

    m,b=np.polyfit(NPain, NP, 1, full = False)


    linspace = np.linspace(min(NPain),max(NPain))
    coeffs = [b,m]



    plt.plot(linspace, PolyCoefficients(linspace,coeffs))

    ptopo.append(stats.pearsonr(NPain, NP)[1])
    cortopo.append(stats.pearsonr(NPain, NP)[0])

    slopetopo.append(m)

    # m,b=np.polyfit(HCain, healthy, 1)
    # linspace = np.linspace(min(HCain),max(HCain))
    # coeffs = [b,m]
    # plt.plot(linspace, PolyCoefficients(linspace,coeffs))


    # plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "58metrics", "kruskalpsecondpart" + ".png"))
    plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, str(n) + ".png"))

    plt.cla()
    plt.clf()


frontalnp = np.asarray(frontalnp)
frontalnp = frontalnp.flatten()
frontal = np.asarray(frontal)
frontal = frontal.flatten()

frontallow=[]
frontalmid=[]
frontalhigh=[]

line=np.max(frontalnp)/3


for i in range(len(frontalnp)):
    if frontalnp[i]<=line:
        frontallow.append(frontal[i])
        # print(frontalbox)
    elif frontalnp[i]>line and frontalnp[i]<=2*line:
        frontalmid.append(frontal[i])
    elif frontalnp[i]>2*line:
        frontalhigh.append(frontal[i])

plt.boxplot((frontallow, frontalmid, frontalhigh),0,showfliers=True,tick_labels=("low", "mid", "high"))

# ax.boxplot((healthy, unhealthy, CBP, CWP, NP), 0, showfliers=True, tick_labels = ('healthy', 'unhealthy', 'back', 'widespread', 'neuropathic'))

# plt.scatter(HCain, healthy)
plt.xlabel(heads[x])


# plt.ylabel("Mean sink of frontal")
# plt.title("Mean sink of frontal" + " vs " + heads[x])

plt.ylabel("Mean sink of "+ regiontitle)
plt.title("Mean sink of "+ regiontitle + " vs " + heads[x])

# plt.ylabel("Mean sink of occipital and parietal")
# plt.title("Mean sink of occipital and parietal" + " vs " + heads[x])


# plt.xticks((0,1,2),("low", "mid", "high"))
# plt.legend(('back', 'widespread', 'neuropathic', 'healthy'))
# plt.legend(('back', 'widespread', 'neuropathic'))
# plt.legend(('neuropathic',))

# heads = (pd.read_excel(xllocation, header = None)).to_numpy()[0]

# plt.title("Singular Value " + str(n))
# x = 31
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, regiontitle + ".png"))
# plt.plot(range(len(healthy)), healthy)

pos = np.array([montage.get_positions()["ch_pos"][ch][:2] for ch in keep_channels])

fig, ax = plt.subplots()
# mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
# mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
# mne.viz.plot_topomap(sink_indices[1726], xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
# mne.viz.plot_topomap(np.mean(maxdata, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)

im,cm = mne.viz.plot_topomap(ptopo, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels)
ax_x_start = 0.85
ax_x_width = 0.02
ax_y_start = 0.1
ax_y_height = 0.8
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
ax.set_title(heads[x] + " sink index p values")
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, "pvals", "pvals" + heads[x] + ".png"))

plt.cla()
plt.clf()




fig, ax = plt.subplots()
# mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
# mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
# mne.viz.plot_topomap(sink_indices[1726], xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
# mne.viz.plot_topomap(np.mean(maxdata, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)

im,cm = mne.viz.plot_topomap(slopetopo, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels)
ax_x_start = 0.85
ax_x_width = 0.02
ax_y_start = 0.1
ax_y_height = 0.8
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
ax.set_title(heads[x] + " sink index slope")
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, "slope", "slope" + heads[x] + ".png"))

plt.cla()
plt.clf()


product = np.abs(np.asarray(slopetopo)*np.log(np.asarray(ptopo)))

fig,ax = plt.subplots(ncols=1)
im,cm   = mne.viz.plot_topomap(product, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels)


# manually fiddle the position of colorbar
ax_x_start = 0.85
ax_x_width = 0.02
ax_y_start = 0.1
ax_y_height = 0.8
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
ax.set_title(heads[x] + " sink index |slope * log(p)|")

fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder,"product", "product" + heads[x] + ".png"))



print(NPain)


def twohoursthirtymin():
    frontal = []
    frontalnp = []
    # frontalbox = []

    # region = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
    # regiontitle = "frontal"

    # region = ['T7', 'C3', 'Cz', 'C4', 'T8']
    # regiontitle = "central"

    # region =['P7', 'P3', 'Cz', 'P4', 'P8', 'O1', 'O2']
    # regiontitle = "occipital and parietal"


    # n=2
    n=-1

    slopetopo = []
    ptopo = []
    cortopo = []
    product = []
    while n < chl-1:
        n+=1

        healthy = []
        unhealthy = []
        CWP = []
        CBP = []
        JP = []
        NP = []

        for i in range(numPatients):
            try:
                np.isnan(xlarray[1, i])
                healthy.append(xlarray2[n, i])
            except TypeError:
                unhealthy.append(xlarray2[n, i])
                match xlarray[1, i]:
                    case "CWP":
                        CWP.append(xlarray2[n, i])
                    case "CBP":
                        CBP.append(xlarray2[n, i])
                    case "PNP":
                        NP.append(xlarray2[n, i])
                        # print(i)
                        # print(xlarray2[n, i])
                    case "NP":
                        NP.append(xlarray2[n, i])
                        # print(i)
                        # print(xlarray2[n, i])
                    case "JP":
                        JP.append(xlarray2[n, i])
                    case "PHN":
                        NP.append(xlarray2[n, i])
                        # print(i)
                        # print(xlarray2[n, i])

        NP = CBP

        # hcsink.append(np.mean(healthy))
        # npsink.append(np.mean(NP))
        # healthy = removeoutliers(healthy)
        # unhealthy = removeoutliers(unhealthy)
        # CBP = removeoutliers(CBP)
        # CWP = removeoutliers(CWP)
        # NP = removeoutliers(NP)
        # JP = removeoutliers(JP)

        # plt.boxplot(x=(healthy, CBP), tick_labels = ('healthy', 'back'))

        # plt.boxplot(x=(healthy, unhealthy, CBP, CWP, JP, NP), tick_labels = ('healthy', 'unhealthy', 'back', 'widespread', 'joint', 'neuropathic'), autorange = True)
        # fig, ax = plt.subplots(1,1)
        # ax.boxplot((healthy, unhealthy, CBP, CWP, NP), 0, showfliers=True, tick_labels = ('healthy', 'unhealthy', 'back', 'widespread', 'neuropathic'))
        # ax.boxplot((healthy, unhealthy), 0, showfliers=False, tick_labels = ('healthy', 'unhealthy'))

        if keep_channels[n] in region:
            if regiontitle == "frontaldenominator":

                frontald.append(NP)

            if regiontitle == "parietalnumerator":
                parietal.append(NP)

            frontal.append(NP)
            frontalnp.append(NPain)




        y = n





        # NP = np.log(NP)
        plt.cla()
        plt.clf()


        # plt.scatter(CBPain, CBP)
        # plt.scatter(CWPain, CWP)
        plt.scatter(NPain, NP)
        # plt.scatter(HCain, healthy)
        plt.xlabel(heads[x])
        plt.ylabel("Mean " + datatype + " " + "of " + keep_channels[n])
        plt.title("Mean " + datatype + " " + "of " + keep_channels[n] + " vs " + heads[x])
        # plt.legend(('back', 'widespread', 'neuropathic', 'healthy'))
        # plt.legend(('back', 'widespread', 'neuropathic'))
        plt.legend(('neuropathic',))




        # plt.legend(('back', 'neuropathic'))

        # m,b=np.polyfit(CBPain, CBP, 1)
        # linspace = np.linspace(min(CBPain),max(CBPain))
        # coeffs = [b,m]
        # plt.plot(linspace, PolyCoefficients(linspace,coeffs))

        # m,b=np.polyfit(CWPain, CWP, 1)
        # linspace = np.linspace(min(CWPain),max(CWPain))
        # coeffs = [b,m]
        # plt.plot(linspace, PolyCoefficients(linspace,coeffs))

        m,b=np.polyfit(NPain, NP, 1, full = False)


        linspace = np.linspace(min(NPain),max(NPain))
        coeffs = [b,m]



        plt.plot(linspace, PolyCoefficients(linspace,coeffs))

        ptopo.append(stats.pearsonr(NPain, NP)[1])
        cortopo.append(stats.pearsonr(NPain, NP)[0])

        slopetopo.append(m)

        # m,b=np.polyfit(HCain, healthy, 1)
        # linspace = np.linspace(min(HCain),max(HCain))
        # coeffs = [b,m]
        # plt.plot(linspace, PolyCoefficients(linspace,coeffs))


        # plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "58metrics", "kruskalpsecondpart" + ".png"))
        plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, str(n) + ".png"))

        plt.cla()
        plt.clf()


    frontalnp = np.asarray(frontalnp)
    frontalnp = frontalnp.flatten()
    frontal = np.asarray(frontal)
    frontal = frontal.flatten()

    frontallow=[]
    frontalmid=[]
    frontalhigh=[]

    line=np.max(frontalnp)/3


    for i in range(len(frontalnp)):
        if frontalnp[i]<=line:
            frontallow.append(frontal[i])
            # print(frontalbox)
        elif frontalnp[i]>line and frontalnp[i]<=2*line:
            frontalmid.append(frontal[i])
        elif frontalnp[i]>2*line:
            frontalhigh.append(frontal[i])

    plt.boxplot((frontallow, frontalmid, frontalhigh),0,showfliers=True,tick_labels=("low", "mid", "high"))

    # ax.boxplot((healthy, unhealthy, CBP, CWP, NP), 0, showfliers=True, tick_labels = ('healthy', 'unhealthy', 'back', 'widespread', 'neuropathic'))

    # plt.scatter(HCain, healthy)
    plt.xlabel(heads[x])


    # plt.ylabel("Mean " + datatype + " " + "of frontal")
    # plt.title("Mean " + datatype + " " + "of frontal" + " vs " + heads[x])

    plt.ylabel("Mean " + datatype + " " + "of "+ regiontitle)
    plt.title("Mean " + datatype + " " + "of "+ regiontitle + " vs " + heads[x])

    # plt.ylabel("Mean " + datatype + " " + "of occipital and parietal")
    # plt.title("Mean " + datatype + " " + "of occipital and parietal" + " vs " + heads[x])


    # plt.xticks((0,1,2),("low", "mid", "high"))
    # plt.legend(('back', 'widespread', 'neuropathic', 'healthy'))
    # plt.legend(('back', 'widespread', 'neuropathic'))
    # plt.legend(('neuropathic',))

    # heads = (pd.read_excel(xllocation, header = None)).to_numpy()[0]

    # plt.title("Singular Value " + str(n))
    # x = 31
    plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, regiontitle + ".png"))

    plt.cla()
    plt.clf()
    # plt.plot(range(len(healthy)), healthy)

    pos = np.array([montage.get_positions()["ch_pos"][ch][:2] for ch in keep_channels])

    fig, ax = plt.subplots()
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(sink_indices[1726], xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(np.mean(maxdata, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)

    im,cm = mne.viz.plot_topomap(ptopo, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels, vlim=(0,0.2))
    ax_x_start = 0.85
    ax_x_width = 0.02
    ax_y_start = 0.1
    ax_y_height = 0.8
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    ax.set_title(heads[x] + " " + datatype + " " + "p values")
    fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, "pvals", "pvals" + heads[x] + ".png"))

    plt.cla()
    plt.clf()




    fig, ax = plt.subplots()
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(sink_indices[1726], xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(np.mean(maxdata, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)

    im,cm = mne.viz.plot_topomap(slopetopo, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels)
    ax_x_start = 0.85
    ax_x_width = 0.02
    ax_y_start = 0.1
    ax_y_height = 0.8
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    ax.set_title(heads[x] + " " + datatype + " " + "slope")
    fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, "slope", "slope" + heads[x] + ".png"))

    plt.cla()
    plt.clf()



    fig, ax = plt.subplots()
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)
    # mne.viz.plot_topomap(np.mean(sink_indices, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(sink_indices[1726], xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True)
    # mne.viz.plot_topomap(np.mean(maxdata, axis = 0), xypos, size = 2, axes = ax, vlim=(0,1), cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = True, names = keep_channels)

    im,cm = mne.viz.plot_topomap(cortopo, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels)
    ax_x_start = 0.85
    ax_x_width = 0.02
    ax_y_start = 0.1
    ax_y_height = 0.8
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    ax.set_title(heads[x] + " " + datatype + " " + "correlation")
    fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, "correlation", "correlation" + heads[x] + ".png"))

    plt.cla()
    plt.clf()

    # manually fiddle the position of colorbar
    #no more product

    # product = np.abs(np.asarray(slopetopo)*np.log(np.asarray(ptopo)))

    # fig,ax = plt.subplots(ncols=1)
    # im,cm   = mne.viz.plot_topomap(product, pos, size = 2, axes = ax, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels)


    # ax_x_start = 0.85
    # ax_x_width = 0.02
    # ax_y_start = 0.1
    # ax_y_height = 0.8
    # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    # clb = fig.colorbar(im, cax=cbar_ax)
    # ax.set_title(heads[x] + " " + datatype + " " + "|slope * log(p)|")

    # fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder,"product", "product" + heads[x] + ".png"))

    return

# twohoursthirtymin()

frontald = []
parietal = []

# region = ['Fp1', 'Fp2']
# regiontitle = "Mid_Frontal"
# twohoursthirtymin()

# region = ['F3', 'F7']
# regiontitle = "Left_Frontal"
# twohoursthirtymin()

# region = ['F4','F8']
# regiontitle = "Right_Frontal"
# twohoursthirtymin()

# region = ['T7','C3']
# regiontitle = "Left_Central"
# twohoursthirtymin()

# region = ['T8','F4']
# regiontitle = "Right_Central"
# twohoursthirtymin()

# region = ['P7','P3']
# regiontitle = "Left_Parietal"
# twohoursthirtymin()

# region = ['P8','P4']
# regiontitle = "Right_Parietal"
# twohoursthirtymin()

# region = ['O1','O2']
# regiontitle = "Occipital"
# twohoursthirtymin()



region = ['F3']
denom=str(region)
regiontitle = "frontaldenominator"
twohoursthirtymin()


# region = ['T7', 'C3', 'Cz', 'C4', 'T8']
# regiontitle = "_central"
# twohoursthirtymin()

# region =['Pz']

region =['P4']
numer=str(region)
regiontitle = "parietalnumerator"
twohoursthirtymin()

frontald = np.asarray(frontald)
parietal = np.asarray(parietal)

print(frontald)
print()
print(parietal)


print(meanparietal)

# meanfront = np.mean(frontald, axis=0)
meanfront=frontald[0]
print(meanfront)

# meanparietal = np.mean(parietal, axis=0)
meanparietal = parietal[0]
print(meanparietal)
# for i in frontald:
#     print(i)

realtest = meanparietal/meanfront
# realtest = 1/meanfront


# realtest = meanparietal

plt.scatter(NPain,realtest)

m,b=np.polyfit(NPain, realtest, 1, full = False)


linspace = np.linspace(min(NPain),max(NPain))
coeffs = [b,m]

plt.plot(linspace, PolyCoefficients(linspace,coeffs))

m,b=np.polyfit(realtest, NPain, 1, full = False)


plt.title("p="+str(np.round(stats.pearsonr(NPain, realtest)[1],4)) + " " + str(np.round(b,3)) + "+" + str(np.round(m,3))+"m c="+str(np.round(stats.pearsonr(NPain, realtest)[0],3)))
plt.xlabel(heads[x])
# plt.ylabel(numer+"/"+denom)
plt.ylabel(numer)

plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs",heads[x] + "pvalue.png"))

# frontalnp.append(NPain)
frontal = np.asarray(frontal).flatten()
frontalnp = np.asarray(frontalnp).flatten()

plt.scatter(frontalnp, frontal)
# plt.scatter(HCain, healthy)
plt.xlabel(heads[x])
plt.ylabel("Mean sink of " + 'frontal')
plt.title("Mean sink of " + 'frontal' + " vs " + heads[x])

# plt.ylabel("Mean sink of " + 'central')
# plt.title("Mean sink of " + 'central' + " vs " + heads[x])

# plt.ylabel("Mean sink of " + 'occipital and parietal')
# plt.title("Mean sink of " + 'occipital and parietal' + " vs " + heads[x])

# plt.legend(('back', 'widespread', 'neuropathic', 'healthy'))
# plt.legend(('back', 'widespread', 'neuropathic'))
plt.legend(('neuropathic',))


# plt.legend(('back', 'neuropathic'))

# m,b=np.polyfit(CBPain, CBP, 1)
# linspace = np.linspace(min(CBPain),max(CBPain))
# coeffs = [b,m]
# plt.plot(linspace, PolyCoefficients(linspace,coeffs))

# m,b=np.polyfit(CWPain, CWP, 1)
# linspace = np.linspace(min(CWPain),max(CWPain))
# coeffs = [b,m]
# plt.plot(linspace, PolyCoefficients(linspace,coeffs))

m,b=np.polyfit(frontalnp, frontal, 1)
linspace = np.linspace(min(NPain),max(NPain))
coeffs = [b,m]
plt.plot(linspace, PolyCoefficients(linspace,coeffs))


slopetopo.append(m)

# m,b=np.polyfit(HCain, healthy, 1)
# linspace = np.linspace(min(HCain),max(HCain))
# coeffs = [b,m]
# plt.plot(linspace, PolyCoefficients(linspace,coeffs))


# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "58metrics", "kruskalpsecondpart" + ".png"))
plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, 'frontal' + ".png"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, 'central' + ".png"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, 'back' + ".png"))


plt.cla()
plt.clf()

"27-33"
'27-29-31-32'

n=33
# n+=1
x=n
# n+=1

healthy = []
unhealthy = []
CWP = []
CBP = []
JP = []
NP = []

for i in range(numPatients):
    try:
        np.isnan(xlarray[1, i])
        healthy.append(xlarray[n, i])
    except TypeError:
        unhealthy.append(xlarray[n, i])
        match xlarray[1, i]:
            case "CWP":
                CWP.append(xlarray[n, i])
            case "CBP":
                CBP.append(xlarray[n, i])
                amatconnection(i,xlarray[n, i])

            case "JP":
                JP.append(xlarray[n, i])
            case "PNP":
                NP.append(xlarray[n, i])
                # amatconnection(i,xlarray[n, i])

            case "NP":
                NP.append(xlarray[n, i])
                # amatconnection(i,xlarray[n, i])

            case "PHN":
                NP.append(xlarray[n, i])
                # amatconnection(i,xlarray[n, i])


# healthy = removeoutliers(healthy)
# unhealthy = removeoutliers(unhealthy)
# CBP = removeoutliers(CBP)
# CWP = removeoutliers(CWP)
# NP = removeoutliers(NP)
# JP = removeoutliers(JP)

# plt.boxplot(x=(healthy, CBP), tick_labels = ('healthy', 'back'))

# plt.boxplot(x=(healthy, unhealthy, CBP, CWP, JP, NP), tick_labels = ('healthy', 'unhealthy', 'back', 'widespread', 'joint', 'neuropathic'), autorange = True)
fig, ax = plt.subplots(1,1)
ax.boxplot((healthy, unhealthy, CBP, CWP, NP), 0, showfliers=True, tick_labels = ('healthy', 'unhealthy', 'back', 'widespread', 'neuropathic'))
# ax.boxplot((CBP, CWP, NP), 0, showfliers=True, tick_labels = ('back', 'widespread', 'neuropathic'))
# ax.boxplot((CBP, NP), 0, showfliers=True, tick_labels = ('back', 'neuropathic'))
#

# ax.boxplot((healthy, unhealthy), 0, showfliers=False, tick_labels = ('healthy', 'unhealthy'))



heads = (pd.read_excel(xllocation, header = None)).to_numpy()[0]

plt.title(heads[n])

# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "58metrics", "kruskalpsecondpart" + ".png"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "10-20metrics", heads[n] + ".png"))

# plt.plot(range(len(healthy)), healthy)
# plt.cla()
# plt.clf()
# x = 27
# # y += 1
# y = n

# plt.scatter(CBPain, CBP)
# plt.scatter(CWPain, CWP)
# plt.scatter(NPain, NP)
# plt.xlabel(heads[x])
# plt.ylabel(heads[y])
# plt.title(heads[x]+ " vs "+ heads[y])
# plt.legend(('back', 'widespread', 'neuropathic'))
# # plt.legend(('back', 'neuropathic'))

# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "CurpainPlots", heads[y] + ".svg"))
# plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "Durpain", heads[y] + ".svg"))

NPain = CBP
# condcor.append(NP)

condcor = []
# condcor.append(NP)

print(condcor)

condcormat = []

for i in condcor:
    condrow = []
    for j in condcor:
        condrow.append(stats.pearsonr(i, j)[0])
    condcormat.append(condrow)
condcormat = np.asarray(condcormat)


print(heads[27:])


n = len([cmap])
fig, axs = plt.subplots(1, n, figsize=(n * 2 * 4 + 2, 3 * 2),
                        layout='constrained', squeeze=False)
for [ax, cmap] in zip(axs.flat, [cmap]):
    psm = ax.pcolormesh(condcormat, cmap=cmap, rasterized=True)
    fig.colorbar(psm, ax=ax)

plt.yticks(range(7),(heads[27:]))
plt.xticks(range(7),(heads[27:]),rotation=30)


# plt.title("correlation between channels")
# amatview += 1
# plt.show()
# m+=1
plt.title("Correlation of Pain Severity for Neuropathic")
fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "corpainconditions.png"))


print(cm)

fig,(ax1,ax2) = plt.subplots(ncols=2)
im,cm   = mne.viz.plot_topomap(hcsink, pos, size = 2, axes = ax1, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels)
im,cm   = mne.viz.plot_topomap(npsink, pos, size = 2, axes = ax2, cmap = 'RdBu_r', extrapolate = 'head', outlines = 'head', show = False, names = keep_channels)

# manually fiddle the position of colorbar
ax_x_start = 0.92
ax_x_width = 0.02
ax_y_start = 0.1
ax_y_height = 0.8
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title("scale") # title on top of colorbar
ax1.set_title("healthy sink indices")
ax2.set_title("neuropathic sink indices")

fig.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", folder, "hcsink" + ".png"))

CWPain = CWP
CBPain = CBP
NPain = NP
HCain = healthy

print(n)

print(paindur)

print(CBPain)

x = 27
# y += 1
y = n

plt.scatter(CBPain, CBP)
plt.scatter(CWPain, CWP)
plt.scatter(NPain, NP)
plt.xlabel(heads[x])
plt.ylabel(heads[y])
plt.title(heads[x]+ " vs "+ heads[y])
plt.legend(('back', 'widespread', 'neuropathic'))

plt.savefig(os.path.join(mainfolder + "Archive\\Old_Graphs", "CurpainPlots", heads[y] + ".svg"))



hcstats = []
pastats = []
CWPstats = []
CBPstats = []
JPstats = []
NPstats = []

##remove unwanted data from participants

x = 0

while x < (len(participants)):
    try:
        np.isnan(participants[x][5])
        hcstats.append(participants[x])
    except TypeError:
        pastats.append(participants[x])
        match participants[x, 5]:
            case "CWP":
                CWPstats.append(participants[x])
            case "CBP":
                CBPstats.append(participants[x])
            case "PNP":
                NPstats.append(participants[x])
            case "NP":
                NPstats.append(participants[x])
            case "JP":
                JPstats.append(participants[x])
            case "PHN":
                NPstats.append(participants[x])
    x += 1


hcstats = np.asarray(hcstats).transpose()
pastats = np.asarray(pastats).transpose()
CWPstats = np.asarray(CWPstats).transpose()
CBPstats = np.asarray(CBPstats).transpose()
JPstats = np.asarray(JPstats).transpose()
NPstats = np.asarray(NPstats).transpose()

print(hcstats.shape)

print(dtypes)

def condstats(condition):
    condlist = [condition[5,0]]
    for i in range(condition.shape[0]):
        summary = ''
        if dtypes.iloc[i] == "object":
            condsum = []
            condnum = []
            for j in condition[i]:
                if j not in condsum:
                    condsum.append(j)
                    x = 0
                    for k in condition[i]:
                        if k == j:
                            x += 1
                    condnum.append(x)
            for l in range(len(condsum)):
                summary += str(condnum[l]) + " " + str(condsum[l])
                if l < len(condsum)-1:
                    summary += ", "
        else:
            condsum = []
            deletes = []
            for j in range(len(condition[i])):
                # try:
                #     np.isnan(condition[i][j])
                #     # deletes.append(j)
                #     # print(condition[i][j])
                # except TypeError:
                #     print("")
                if np.isnan(condition[i][j])==True:
                    deletes.append(j)
            condsum = np.delete(condition[i], deletes)
            if len(condsum) == 0:
                summary = "n/a"
            else:
                summary = str(round(np.mean(condsum), 2)) + "±" + str(round(np.std(condsum), 2))

        condlist.append(summary)



    # condlist = np.delete(condlist, (0,1,5))
    condlist = np.delete(condlist, (1, 2))

    return condlist

# print(condstats(hcstats))

print(condstats(participants.transpose()))

summarray = np.asarray([condstats(participants.transpose()),condstats(pastats),condstats(hcstats),condstats(CBPstats),condstats(CWPstats),condstats(NPstats),condstats(JPstats)])


summarray[0,0] = "participants"
summarray[1,0] = "patients"
summarray[2,0] = "healthy controls"

np_to_excel(mainfolder+"summary.xlsx", summarray.transpose(), [], [])

## this is the smartest jank ive ever done
# np_to_excel(mainfolder+"summary.xlsx", np.asarray((pcolumns, np.empty(20))).transpose(), [], [])


print(len(pcolumns))
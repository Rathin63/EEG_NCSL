import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import makedirs

def sec2h(t):
    ## Converts seconds to decimal
    # Inputs:
    # seconds: Time in seconds (int from 0 to 86400, which is 24 hours)

    # Outputs:
    # t: seconds

    hours = np.floor(t / 3600)
    t = t - hours * 3600
    mins = np.floor(t / 60)
    secs = t - mins * 60
    h = hours + (mins / 60) + (secs / 3600)
    return h

def sec2time(t):
    ## Converts seconds to hour HH:MM:SS

    # Inputs:
    # seconds: Time in seconds (int from 0 to 86400, which is 24 hours)

    # Outputs:
    # t: seconds

    hours = np.floor(t / 3600)
    t = t - hours * 3600
    mins = np.floor(t / 60)
    secs = t - mins * 60
    #h = hours + (mins / 60) + (secs / 3600)
    time = f"{int(hours)}:{int(mins)}:{int(secs)}"
    return time

def time2sec(hour):
    ## Converts HH:MM:SS to seconds

    # Inputs:
    # hour: Time in HH:MM:SS format(str)

    # Outputs:
    # t: seconds

    time = hour.split(':')
    h = time[0]
    m = time[1]
    s = time[2]
    h = int(float(h))
    m = int(float(m))
    s = int(float(s))
    t = h * 60 * 60 + m * 60 + s
    return t

def time_sum(time1, time2):
    ## Converts seconds to decimal

    # Inputs:
    # time1,2: time in HH:MM:SS

    # Outputs:
    # time: sum of time 1,2 in HH:MM:SS

    seconds1 = time2sec(time1)
    seconds2 = time2sec(time2)
    seconds = seconds1 + seconds2

    hours = np.floor(seconds / 3600)
    seconds = seconds - hours * 3600
    mins = np.floor(seconds / 60)
    secs = seconds - mins * 60

    hours = np.mod(hours, 24)

    time = f"{int(hours)}:{int(mins)}:{int(secs)}"
    return time

def viz_eeg(data, labels_, fs=2000, WIN_LEN_SEC=10, WIN_STEP_SEC=1, fig_out_dir=".", fig_name="Figure",
            plot_title="Title", start_time='0:0:0', detect_time='0:0:0', colorMaps=None, annotations=dict(),
            soz_channels=[], prop_channels=[], bad_channels=[], data_recon=[]):
    most_involved_channels = []

    if colorMaps is None:
        color_names = ["royalblue",
                       "mediumorchid",
                       "green",
                       "teal",
                       "dodgerblue",
                       "darkorange",
                       "olive",
                       "gray",
                       "salmon",
                       "chocolate",
                       "green",
                       "teal",
                       "dodgerblue"]
    elif type(colorMaps) == str:
        color_names = [colorMaps for i in range(13)]

    nCH = data.shape[0]

    # Plot Figures
    n_wins = int(np.floor((data.shape[1] - (WIN_LEN_SEC * fs)) / (WIN_STEP_SEC * fs))) + 1

    print(n_wins)

    for win_i in tqdm(range(n_wins + 1), desc='Plotting iEEG', initial=0, leave=False, dynamic_ncols=True, smoothing=0, disable=False, position=0):
        win_i_0 = win_i * WIN_STEP_SEC * fs
        win_i_1 = win_i_0 + WIN_LEN_SEC * fs
        # tqdm.write(f"Processing window {win_i}: win_i_0={win_i_0}, win_i_1={win_i_1}")

        fig, ax = plt.subplots(1, figsize=(17, 14))
        plt.ion()

        # if win_i_0 < (0.25*60/win_sec) * win_sec * fs:
        #     ax.set_facecolor("#fff5e6")
        # else:
        ax.set_facecolor("#fffafa")

        for i in range(WIN_LEN_SEC):
            plt.axvline(fs * i, c='silver', linestyle='-', alpha=0.5, linewidth=1)

        labi_1 = ''
        labi_2 = ''
        c_idx = 0
        for i in range(nCH):
            if labels_[i].strip() in bad_channels:
                continue

            plot_data = data[i, win_i_0:win_i_1]

            tW = np.arange(0, plot_data.shape[0], 1)

            current_label = ''.join((x for x in labels_[i] if (not x.isdigit()) and (not x == '-')))

            if labels_[i] in soz_channels:
                ax.text(-fs / 3 + fs / 20, i * 1.2 * 1e-4, labels_[i], c='red', fontsize=12, weight="bold")
            elif labels_[i] in prop_channels:
                ax.text(-fs / 3 + fs / 20, i * 1.2 * 1e-4, labels_[i], c='orange', fontsize=12, weight="bold")
            else:
                ax.text(-fs / 3 + fs / 20, i * 1.2 * 1e-4, labels_[i], c='k', fontsize=12)

            if labi_1 == '':
                labi_1 = labels_[i]
            else:
                labi_2 = labels_[i]
                ch_labi_1 = ''.join((x for x in labi_1 if (not x.isdigit()) and (not x == '-')))
                ch_labi_2 = ''.join((x for x in labi_2 if (not x.isdigit()) and (not x == '-')))

                if ch_labi_1 != ch_labi_2:
                    labi_1 = labels_[i]
                    c_idx += 1

            if colorMaps is None:
                if len(data_recon) == 0:
                    ax.plot(tW, plot_data + i * 1.2 * 1e-4, color=color_names[c_idx],  zorder=3)
                else:
                    ax.plot(tW, plot_data + i * 1.2 * 1e-4, color=color_names[c_idx], lw=2,  zorder=3)
            else:
                # ampl = plot_data.max() - plot_data.min()
                if len(most_involved_channels) == 0:  # no channel is specified to be colored
                    ax.plot(tW, plot_data + i * 1.2 * 1e-4, color=colorMaps[current_label], linewidth=1.5, alpha=1,  zorder=3)
                else:
                    if labels_[i] in soz_channels:
                        ax.plot(tW, plot_data + i * 1.2 * 1e-4, color=colorMaps[current_label], linewidth=3, alpha=0.6,  zorder=3)
                    ax.plot(tW, plot_data + i * 1.2 * 1e-4, color=colorMaps[current_label], linewidth=1.5, alpha=1,  zorder=3)
            if len(data_recon) != 0:
                plot_data_recon = data_recon[i, win_i_0:win_i_1]
                ax.plot(tW, plot_data_recon + i * 1.2 * 1e-4, color='k', zorder=5)

        tW_ticks = np.arange(0, plot_data.shape[0] + 1, fs)
        tW_tick_labels = []

        for ti in range(len(tW_ticks)):
            ticki = (tW_ticks[ti] + win_i_0) / fs
            tick = time_sum(start_time, sec2time(ticki))
            tW_tick_labels.append(tick)

            if detect_time is not None:
                h0, m0, s0 = tick.split(':')[0], tick.split(':')[1], tick.split(':')[2]
                h1, m1, s1 = detect_time.split(':')[0], detect_time.split(':')[1], detect_time.split(':')[2]
                if h0 == h1 and m0 == m1 and np.abs(int(np.floor(float(s0))) - int(np.floor(float(s1)))) < WIN_LEN_SEC // 8:
                    ax.axvline(tW_ticks[ti], linewidth=5, color='red')
                    ax.text(tW_ticks[ti], 0, "Detect ON", fontsize=20, bbox=dict(facecolor='red', alpha=1), color='gold')

        """for annoti in annotations:
            tW_ticks = np.arange(0, plot_data.shape[0] + 1, fs)
            tW_tick_labels = []
            for ti in range(len(tW_ticks)):
                annot_txt = annoti
                annot_time = annotations[annoti] # in H:M:S format
                ticki = (tW_ticks[ti]+win_i_0) / fs
                tick = time_sum(start_time, sec2time(ticki))
                tW_tick_labels.append(tick)
                h0, m0, s0 = tick.split(':')[0], tick.split(':')[1], tick.split(':')[2]
                h1, m1, s1 = annot_time.split(':')[0], annot_time.split(':')[1], annot_time.split(':')[2]
                if h0 == h1 and m0 == m1 and np.abs(int(np.floor(float(s0)))-int(np.floor(float(s1)))) < WIN_LEN_SEC//8:
                    ax.axvline(tW_ticks[ti], linewidth=5, color='limegreen')
                    ax.text(tW_ticks[ti], 5*1.2*1e-4, annot_txt, fontsize=20, bbox=dict(facecolor='springgreen', alpha=1), color='saddlebrown')"""
        for annoti in annotations:
            tW_ticks = np.arange(0, plot_data.shape[0] + 1, fs)
            tW_tick_labels = []
            for ti in range(len(tW_ticks)):
                annot_txt = annoti
                annot_time = annotations[annoti]  # in H:M:S format
                ticki = (tW_ticks[ti] + win_i_0) / fs
                tick = time_sum(start_time, sec2time(ticki))
                tW_tick_labels.append(tick)
                h0, m0, s0 = tick.split(':')[0], tick.split(':')[1], tick.split(':')[2]
                if annot_time.count(':') == 2:  # HH:MM:SS
                    h1, m1, s1 = annot_time.split(':')[0], annot_time.split(':')[1], annot_time.split(':')[2]
                    if s1.count('.') == 1:
                        s1 = s1.split('.')[0]  # for cases like 10:10:10.5678 -> 10:10:10
                    h1, m1, s1 = str(int(h1)), str(int(m1)), str(int(s1))  # reduce to 1 digit in cases like 05 -> 5
                elif annot_time.count(':') == 1:  # MM:SS.MS
                    m1, s1 = annot_time.split(':')[0], annot_time.split(':')[1]
                    if s1.count('.') == 1:
                        s1 = s1.split('.')[0]
                    m1, s1 = str(int(m1)), str(int(s1))  # reduce to 1 digit in cases like 05 -> 5
                    h1 = '0'
                if h0 == h1 and m0 == m1 and np.abs(int(np.floor(float(s0))) - int(np.floor(float(s1)))) < 1:
                    ax.axvline(tW_ticks[ti], linewidth=5, color='limegreen')
                    ax.text(tW_ticks[ti], np.random.uniform(5, 8) * 1.2 * 1e-4, annot_txt, fontsize=20,
                            bbox=dict(facecolor='springgreen', alpha=1), color='saddlebrown')

        ax.set(xlim=[-fs / 3, fs * WIN_LEN_SEC])
        ax.set(ylim=[-0.255 * 1e-3, nCH * 1.25 * 1e-4])
        ax.set_xticks(tW_ticks)
        ax.set_xticklabels(tW_tick_labels, fontsize=16, rotation=40)
        ax.axes.get_yaxis().set_visible(False)
        ax.grid(False)
        plt.title(plot_title + "\nGray vertical lines are 1 second interval", fontsize=16)
        plt.tight_layout()
        makedirs(f'{fig_out_dir}//{fig_name}', exist_ok=True)
        plt.ioff()
        plt.savefig(f'{fig_out_dir}//{fig_name}//{fig_name}_{win_i:03}.png')
        plt.close()

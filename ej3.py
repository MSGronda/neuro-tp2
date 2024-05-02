import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from common import extract_plan_spikes, train_and_test


def ej3():
    with open('./datos/datos_disparos_mono_tp2.npz', 'rb') as loadfile:
        spike_times = np.load(loadfile, allow_pickle=True)['spike_times']

    with open('./datos/metadata_mono_tp2.npy', 'rb') as loadfile:
        metadata = np.load(loadfile)
        time_touch_held = metadata['time_touch_held']
        time_go_cue = metadata['time_go_cue']
        trial_reach_target = metadata['trial_reach_target']

    repetitions = 25

    target_samples = [5, 10, 15, 20, 25, 30, 35, 40]
    precisions = []
    for target_sample_size in target_samples:
        precisions.append(test_using_trail_size(repetitions, time_touch_held, spike_times, time_go_cue, trial_reach_target, target_sample_size))

    graph_precisions_for_sample_sizes(target_samples, precisions)


def test_using_trail_size(repetitions, time_touch_held, spike_times, time_go_cue, trial_reach_target, sample_size):
    precisions = []
    for _ in range(repetitions):
        plan_spikes = extract_plan_spikes(time_touch_held, spike_times, time_go_cue, 250, 100)
        correct_targets, decoded_targets, test_trials = train_and_test(trial_reach_target, plan_spikes, sample_size)

        precisions.append(np.mean(correct_targets == decoded_targets))

    return precisions


def graph_precisions_for_sample_sizes(classes, precisions):
    fig = plt.figure(figsize=(12, 12))
    ax = sns.violinplot(data=pd.DataFrame(np.transpose(precisions), columns=classes))
    ax.set(xlabel='Cantidad de datos de entrenamiento', ylabel='Precisión decodificando')
    fig.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from common import extract_plan_spikes, train_and_test


def ej1():
    with open('./datos/datos_disparos_mono_tp2.npz', 'rb') as loadfile:
        spike_times = np.load(loadfile, allow_pickle=True)['spike_times']

    with open('./datos/metadata_mono_tp2.npy', 'rb') as loadfile:
        metadata = np.load(loadfile)
        time_touch_held = metadata['time_touch_held']
        time_go_cue = metadata['time_go_cue']
        time_target_acquired = metadata['time_target_acquired']
        trial_reach_target = metadata['trial_reach_target']
        target_locations = metadata['target_locations']
        target_angles = metadata['target_angles']

    # Ejercicio 1.a

    print(np.unique(time_go_cue - time_touch_held))
    plan_spikes = extract_plan_spikes(time_touch_held, spike_times, time_go_cue, 750)
    correct_targets, decoded_targets, test_trials = train_and_test(trial_reach_target, plan_spikes)
    print('Porcentaje correcto: ', np.mean(correct_targets == decoded_targets))
    print('({} episodios de testeo)'.format(len(test_trials)))

    # Ejercicio 1.b

    decode_perf = evaluate_window_length_performance([i for i in range(50, 800, 50)], time_touch_held, spike_times, time_go_cue, trial_reach_target)
    graph_window_length_comparison(decode_perf)

    # Ejercicio 1.c
    offsets = []
    correct_percentage = []
    for offset in range(0, 500, 10):
        plan_spikes = extract_plan_spikes(time_touch_held, spike_times, time_go_cue, 250, offset)
        correct_targets, decoded_targets, test_trials = train_and_test(trial_reach_target, plan_spikes)

        offsets.append(offset)
        correct_percentage.append(np.mean(correct_targets == decoded_targets))

    graph_offset_comparison(offsets, correct_percentage)


def evaluate_window_length_performance(window_length_values, time_touch_held, spike_times, time_go_cue, trial_reach_target):
    decode_perf = []
    for window_length in window_length_values:
        plan_spikes = extract_plan_spikes(time_touch_held, spike_times, time_go_cue, window_length)
        decode_perf.append([])

        for i in range(50):

            correct_targets, decoded_targets, test_trials = train_and_test(trial_reach_target, plan_spikes)

            decode_perf[-1].append(np.mean(correct_targets == decoded_targets))

    return np.array(decode_perf)


def graph_window_length_comparison(decode_perf):
    fig = plt.figure(figsize=(12, 12))
    ax = sns.violinplot(data=pd.DataFrame(decode_perf.T, columns=np.arange(50, 800, 50)))
    ax.set(xlabel='Duración de ventana', ylabel='Precisión decodificando')
    fig.show()


def graph_offset_comparison(offsets, correct):
    plt.scatter(offsets, correct, marker='o')

    plt.xlabel(f'Offsets')
    plt.ylabel(f'Precisión decodificando')
    plt.grid(True)

    plt.show()
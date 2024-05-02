import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from common import extract_plan_spikes, train_and_test
import seaborn as sns


def ej2():
    with open('./datos/datos_disparos_mono_tp2.npz', 'rb') as loadfile:
        spike_times = np.load(loadfile, allow_pickle=True)['spike_times']

    with open('./datos/metadata_mono_tp2.npy', 'rb') as loadfile:
        metadata = np.load(loadfile)
        time_touch_held = metadata['time_touch_held']
        time_go_cue = metadata['time_go_cue']
        trial_reach_target = metadata['trial_reach_target']

    # Ejercicio 2.a

    repetitions = 25

    precisions = []
    num_of_neurons = [30, 60, 90, 120, 150]

    for n in num_of_neurons:
        current_precisions = []

        for _ in range(repetitions):
            selected_neuron_indices = np.random.choice(190, n, replace=False)
            selected_spike_times = select_spike_times(spike_times, selected_neuron_indices)

            plan_spikes = extract_plan_spikes(time_touch_held, selected_spike_times, time_go_cue, 250, 100)
            correct_targets, decoded_targets, test_trials = train_and_test(trial_reach_target, plan_spikes)

            current_precisions.append(np.mean(correct_targets == decoded_targets))

        precisions.append(current_precisions)

    graph_precisions_for_num_neurons(num_of_neurons, precisions)

    # Ejercicio 2.b
    selected_spike_times = select_spikes_with_firings(spike_times, 30, "highest")
    precisions_highest = test_using_selected_spikes(repetitions, time_touch_held, selected_spike_times, time_go_cue, trial_reach_target)

    selected_spike_times = select_spikes_with_firings(spike_times, 30, "lowest")
    precisions_lowest = test_using_selected_spikes(repetitions, time_touch_held, selected_spike_times, time_go_cue, trial_reach_target)

    precisions_random = []
    for _ in range(repetitions):
        selected_spike_times = select_spike_times(spike_times, np.random.choice(190, 30, replace=False))
        plan_spikes = extract_plan_spikes(time_touch_held, selected_spike_times, time_go_cue, 250, 100)
        correct_targets, decoded_targets, test_trials = train_and_test(trial_reach_target, plan_spikes)

        precisions_random.append(np.mean(correct_targets == decoded_targets))

    graph_precisions_firings(["Mayor tasa", "Menor tasa", "Aleatoria"], [precisions_highest, precisions_lowest, precisions_random])


def test_using_selected_spikes(repetitions, time_touch_held, selected_spike_times, time_go_cue, trial_reach_target, ):
    precisions = []
    for _ in range(repetitions):
        plan_spikes = extract_plan_spikes(time_touch_held, selected_spike_times, time_go_cue, 250, 100)
        correct_targets, decoded_targets, test_trials = train_and_test(trial_reach_target, plan_spikes)

        precisions.append(np.mean(correct_targets == decoded_targets))

    return precisions


def select_spike_times(spike_times, selected_neuron_indices):
    resp = []
    for spike_time in spike_times:
        resp.append(spike_time[selected_neuron_indices])
    return np.array(resp)


def select_spikes_with_firings(spike_times, n, type):
    fire_rate = [0 for _ in spike_times[0]]

    for spike_time in spike_times:
        for i, neuron_firings in enumerate(spike_time):
            fire_rate[i] += len(neuron_firings)

    if type == "lowest":
        return select_spike_times(spike_times, sorted(range(len(fire_rate)), key=lambda i: fire_rate[i])[:n])
    elif type == "highest":
        return select_spike_times(spike_times, sorted(range(len(fire_rate)), key=lambda i: fire_rate[i], reverse=True)[:n])
    else:
        raise ValueError("Tonto")


def graph_precisions_for_num_neurons(num_of_neurons, precisions):
    fig = plt.figure(figsize=(12, 12))
    ax = sns.violinplot(data=pd.DataFrame(np.transpose(precisions), columns=num_of_neurons))
    ax.set(xlabel='Numero de nueronas', ylabel='Precisión decodificando')
    fig.show()

def graph_precisions_firings(classes, precisions):
    fig = plt.figure(figsize=(12, 12))
    ax = sns.violinplot(data=pd.DataFrame(np.transpose(precisions), columns=classes))
    ax.set(xlabel='Numero de nueronas', ylabel='Precisión decodificando')
    fig.show()


import numpy as np


def extract_plan_spikes(time_touch_held, spike_times, time_go_cue, window_length=None, start_offset=None):
    if start_offset:
        trial_starts = time_touch_held + start_offset
    else:
        trial_starts = time_touch_held

    plan_spikes = []
    for tx, trialSpikes in enumerate(spike_times):
        if window_length:
            trial_end = trial_starts[tx] + window_length
        else:
            trial_end = time_go_cue[tx]

        if (trial_end < trial_starts[tx]) or (trial_end > time_go_cue[tx]):
            raise ValueError("El final del episodio (trial_end) es menor que el comienzo (trial_starts) o que el final del periodo de planificación (time_go_cue)")
        else:
            plan_spikes.append(
                np.array([np.sum((st > trial_starts[tx]) &
                                 (st < trial_end)) for st in trialSpikes]))
    return np.array(plan_spikes)


def train_and_test(trial_reach_target, plan_spikes, size=25):
    training_trials = []
    test_trials = []
    for c in range(8):
        target_trials = np.argwhere((trial_reach_target == c)).squeeze()

        random_training_trials = np.random.choice(target_trials, size, replace=False)
        training_trials.append(random_training_trials)
        remaining_test_trials = np.setdiff1d(target_trials, random_training_trials)

        test_trials.extend(remaining_test_trials)

    num_neurons = plan_spikes.shape[1]
    mean_spike_counts = np.zeros((num_neurons, 8))
    for c in range(8):
        mean_spike_counts[:, c] = np.mean(plan_spikes[training_trials[c], :], axis=0)

    poisson_likelihood = np.zeros((len(test_trials), 8))
    for c in range(8):
        m = mean_spike_counts[:, c]
        poisson_likelihood[:, c] = multivariate_poisson_logpdf(m, plan_spikes[test_trials, :])

    correct_targets = trial_reach_target[test_trials]
    decoded_targets = np.argmax(poisson_likelihood, axis=1)

    return correct_targets, decoded_targets, test_trials


def multivariate_poisson_logpdf(mu, x, mean_eps=0.01):
    mu2 = mu
    mu2[np.argwhere(mu < mean_eps)] = mean_eps
    return np.sum(x * np.log(mu2) - mu2, axis=1)

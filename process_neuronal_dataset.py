import os
import numpy as np

from neuronal.utils.consts import NATURAL_MOVIES
from neuronal.utils.data import DATA_DIR


def process_session(index):
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    cache_dir = '../../../jonatham/allen_institute_data/ecephys_cache_dir/'
    assert f"session_{index}" in os.listdir(
        cache_dir), f"session {index} not in {cache_dir}, instead {os.listdir(cache_dir)}"

    manifest_path = os.path.join(cache_dir, "manifest.json")

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    print("got cache")

    session = cache.get_session_data(index)

    print("got session")

    session_dir = str(session.ecephys_session_id)

    for name in NATURAL_MOVIES:
        print(f"running {name}")
        stim = session.stimulus_presentations
        movie_stim = stim[stim.stimulus_name == name]
        starts = movie_stim.start_time[movie_stim.frame == 0]
        ends = movie_stim.stop_time[
            movie_stim.frame == (np.where(movie_stim.frame.to_numpy() == 'null', 0, movie_stim.frame.to_numpy())).max()]
        for repeat, (start, end) in enumerate(zip(starts, ends)):
            print(f"running {repeat} repeat")
            spikes_mask = {k: (v >= start) & (v <= end) for k, v in session.spike_times.items()}
            filtered_spike_times = {str(k): v[spikes_mask[k]] for k, v in session.spike_times.items()}
            filtered_spike_amplitudes = {str(k): v[spikes_mask[k]] for k, v in session.spike_amplitudes.items() if
                                         k in spikes_mask}
            filtered_running_speed = session.running_speed[
                ~ ((session.running_speed.start_time > end) | (start > session.running_speed.end_time))]
            if hasattr(session.invalid_times, 'start_time'):
                filtered_invalid_times = session.invalid_times[
                    ~ ((session.invalid_times.start_time > end) | (start > session.invalid_times.stop_time))]
            else:
                filtered_invalid_times = None

            path = [DATA_DIR, session_dir, name, str(repeat)]
            os.makedirs(os.path.join(*path), exist_ok=True)
            with open(os.path.join(*path, "spike_times.npz"), 'wb') as f:
                np.savez(f, **filtered_spike_times)
            print("saved spike times")

            with open(os.path.join(*path, "spike_amplitudes.npz"), 'wb') as f:
                np.savez(f, **filtered_spike_amplitudes)
            print("saved spike amplitudes")

            filtered_running_speed.to_csv(os.path.join(*path, "running_speed.csv"))
            print("saved running speed")

            if filtered_invalid_times is not None:
                filtered_invalid_times.to_csv(os.path.join(*path, "invalid_times.csv"))
                print("saved invalid times")

            stimulus_rows = (movie_stim.start_time >= start) & (movie_stim.start_time < end)
            frames_start = movie_stim.start_time[stimulus_rows].to_numpy()
            frames_end = movie_stim.stop_time[stimulus_rows].to_numpy()

            with open(os.path.join(*path, "frame_times.npz"), 'wb') as f:
                np.savez(f, start=frames_start, end=frames_end)
            print("printed frame times")

    session.units.to_csv(os.path.join(DATA_DIR, session_dir, "units.csv"))
    print("saved units")
    session.probes.to_csv(os.path.join(DATA_DIR, session_dir, "probes.csv"))
    print("saved probes")

    import json
    with open(os.path.join(DATA_DIR, session_dir, "metadata.json"), 'w') as f:
        json.dump({k: v for k, v in session.metadata.items() if "start_time" not in k}, f, indent=4)
    print("saved metadata")

    with open(os.path.join(DATA_DIR, session_dir, "start_time.txt"), 'w') as f:
        f.write(str(session.session_start_time))
    print("saved start time")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process allen data')
    parser.add_argument('--index', type=int, default=None, help='session index', required=True)

    args, _ = parser.parse_known_args()
    process_session(args.index)

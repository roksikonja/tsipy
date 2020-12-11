import datetime
import os


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def create_results_dir(results_dir_path, dir_name="results"):
    results_dir = make_dir(
        os.path.join(
            results_dir_path,
            datetime.datetime.now().strftime(f"%m-%d_%H-%M-%S_{dir_name}"),
        )
    )
    return results_dir

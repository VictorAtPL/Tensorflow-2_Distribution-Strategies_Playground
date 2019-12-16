import argparse
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

STARTEGIES_DICT = {
    'nostrategy': 'None',
    'multiworkermirroredstrategy': 'MultiWorkerMirroredStrategy',
    'mirroredstrategy': 'MirroredStrategy',
}

REGEX_NO_OF_DEVICES = r'Number of devices: ([0-9]+)'
REGEX_TIME = r'Epoch execution average time without 1st epoch: ([0-9\.]+)'
REGEX_ARCHITECTURE = r'Model: "([^"]+)"'


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def newline(ax, p1, p2, *args, **kwargs):
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    line = Line2D([xmin, xmax], [ymin, ymax], **kwargs)
    ax.add_line(line)
    return line


def generate_plot(args):
    path = args.path

    job_ids = []
    results_dict = {'GPUs': [], 'Time': [], 'Strategy': []}
    for filename in glob(os.path.join(path, '*output*')):
        filename_arr = filename.split("/")[-1].split(".")[0].split("_")
        job_ids.append(filename_arr[1])

        with open(filename, mode='r') as f:
            file_content = "\n".join(f.readlines())
        no_of_devices = re.search(REGEX_NO_OF_DEVICES, file_content).groups()[0]

        time_regex_result = re.search(REGEX_TIME, file_content)
        if not time_regex_result:
            continue
        time = time_regex_result.groups()[0]
        strategy = STARTEGIES_DICT[filename_arr[0]]

        architecture_regex_result = re.search(REGEX_ARCHITECTURE, file_content)
        architecture = None if not architecture_regex_result else architecture_regex_result.groups()[0]

        if (not args.architecture and args.architecture == architecture) or (architecture and args.architecture in architecture):
            results_dict['GPUs'].append(int(no_of_devices))
            results_dict['Time'].append(float(time))
            results_dict['Strategy'].append(strategy)

    df = pd.DataFrame(results_dict)
    single_gpu_time = df[df['GPUs'] == 1]['Time'].values[0]
    df['Speedup'] = single_gpu_time / df['Time']

    gpus_tested = df['GPUs'].values
    missing_gpus = [num for num in range(0, df['GPUs'].max() + 1)
                    if num not in gpus_tested]

    df = df.append(pd.DataFrame(data={"GPUs": missing_gpus}), sort=False)
    df = df.sort_values("GPUs")
    ax = sns.barplot(x="GPUs", y="Speedup", hue="Strategy", dodge=False, data=df)

    change_width(ax, .35)

    if not args.architecture:
        args.architecture = "SA-MIRI's network"
    ax.set_title("{} / {} / Marenostrum Power9-CTE / TensorFlow 2.0".format(args.dataset, args.architecture.upper()))
    ax.set_xlim(0, df['GPUs'].max() + 1)
    ax.set_xticklabels(["" if tick in missing_gpus else tick for tick in df['GPUs'].values])

    plt.legend(loc='lower right')

    newline(ax, [1, 1], [df['Speedup'].max(), df['Speedup'].max()], linewidth=1, color='gray',
            linestyle='dashed')

    plt.savefig("output/figure.png", dpi=300, quality=95, transparent=True)
    plt.show()


def main():
    sns.set_palette("bright")
    sns.set_style("whitegrid")
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('dataset', type=str, choices=["MNIST", "CIFAR"])

    parser.add_argument('--architecture', type=str, default=None)

    generate_plot(parser.parse_args())


if __name__ == '__main__':
    main()

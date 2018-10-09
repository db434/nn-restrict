import argparse
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt

import structured
from util import log


def get_fields(csv_file, fields):
    """Return a dictionary mapping each name in `fields` to the sequence of
    values seen for that field in the given file.
    """
    result = OrderedDict()
    for field in fields:
        result[field] = []
    
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in fields:
                result[field].append(row[field])
    
    return result


def plot(data, colours, line_styles):
    """Plot the given data.
    
    Expect to receive a dict mapping experiment names to results. Each result is
    itself a dict mapping a type of data to a sequence of values.
    
    Each experiment has a different colour, and each type of data has a
    different line style.
    """
    
    plt.style.use("seaborn-poster")  # See also seaborn-talk and seaborn-paper.
    plt.grid(axis="y", linestyle="-", color="#d8dcd6")
        
    for name, experiment in data.items():
        for label, series in experiment.items():
            epochs = range(1, len(series) + 1)
            line, = plt.plot(epochs, series, color=colours[name],
                             linestyle=line_styles[label])
            line.set_label(name + " " + label)
    
    plt.legend(loc="lower right")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def get_conv_type(filename):
    """Extract the convolution type from the name of the log file. Assumes the
    log file has been created using this tool."""
    for conv_type in structured.conv2d_types.keys():
        if conv_type in filename:
            return conv_type
    else:
        log.error("Couldn't detect convolution type of", filename)
        exit(1)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("filename", type=str, nargs="+",
                        help="CSV file containing training data")
    args = parser.parse_args()

    results = OrderedDict()

    for filename in args.filename:
        data = get_fields(filename, ["Val top1", "Train top1"])
        conv_type = get_conv_type(filename)
        results[conv_type] = data
    
    line_styles = {"Val top1": "-", "Train top1": ":"}
    colours = {"fc": "r", "separable": "g", "shuffle": "b", "butterfly": "c",
               "roots": "m", "shift": "y", "hadamard": "k"}
    
    plot(results, colours, line_styles)
    

if __name__ == "__main__":
    main()

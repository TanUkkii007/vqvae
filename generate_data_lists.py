"""Generate .
Usage: generate_data_lists.py [options]

Options:
    --out-dir=<path>
    --data-list-file=<path>
    --n-test-set=<n>
    --n-validation-set=<n>
    -h, --help                   Show this help message and exit
"""

from docopt import docopt
import csv
import os
from random import shuffle


def load_file_list(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            yield row[0]


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    out_dir = args["--out-dir"]
    data_list_file = args["--data-list-file"]
    n_test_set = int(args["--n-test-set"])
    n_validation_set = int(args["--n-validation-set"])

    key_list = load_file_list(data_list_file)

    shuffled_key_list = list(key_list)
    shuffle(shuffled_key_list)

    test_set = shuffled_key_list[:n_test_set]
    validation_set = shuffled_key_list[n_test_set:n_test_set + n_validation_set]
    training_set = shuffled_key_list[n_test_set + n_validation_set:]

    with open(os.path.join(out_dir, 'test.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key in test_set:
            writer.writerow([key])

    with open(os.path.join(out_dir, 'validation.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key in validation_set:
            writer.writerow([key])

    with open(os.path.join(out_dir, 'train.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key in training_set:
            writer.writerow([key])


if __name__ == '__main__':
    main()

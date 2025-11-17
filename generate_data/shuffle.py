import os
import csv
import numpy as np

os.chdir(os.path.dirname(__file__))


def save_as_csv(dataset, filename):
    """Save dataset as CSV file"""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "solution"])
        writer.writerows(dataset)
    print(f"Saved to {filename}")


def save_as_npz(dataset, filename):
    """Save dataset as NPZ file"""
    questions = np.array([row[0] for row in dataset])
    solutions = np.array([row[1] for row in dataset])
    np.savez(filename, questions=questions, solutions=solutions)
    print(f"Saved to {filename}")


def shuffle_data(input_file, output_filename):
    """Shuffle the rows of a CSV file and save to a new file

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output file
    """
    # Read the data from the input CSV file
    with open(input_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    # Shuffle the data
    np.random.shuffle(data)

    # Write the shuffled data to the output files
    save_as_csv(data, f"{output_filename}.csv")
    # save_as_npz(data, f"{output_filename}.npz")


if __name__ == "__main__":
    np.random.seed(86)
    shuffle_data("0_unshuffled_unrelated_math_testing.csv", "unrelated_math_testing")

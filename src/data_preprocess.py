import numpy as np
import os
import json

def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj

def save_to_file(data, file_path):
    with open(file_path, 'w') as f:
        for examples in data:
            for example in examples:
                input_example, output_example = example

                input_example = numpy_to_list(input_example)
                input_padding_reqd = 1024 - len(input_example)
                input_example_padded = input_example + [10] * input_padding_reqd

                output_example = numpy_to_list(output_example)
                output_padding_reqd = 1024 - len(output_example)
                output_example_padded = output_example + [10] * output_padding_reqd

                example = input_example_padded + [10] + output_example_padded
                json.dump(example, f)
                f.write('\n')
                

def flatten_with_separator(arr, separator=10):
    arr = np.array(arr)
    result = []
    for i, row in enumerate(arr):
        result.extend(row)
        if i < len(arr) - 1:
            result.append(separator)
    
    return np.array(result, dtype=np.int64)

def load_data(file_paths):
    train_data = []
    test_data = []
    for file_path in file_paths:
        rules_input = []
        test_input = []
        with open(file_path, "r") as f:
            data = json.load(f)
            for item in data["train"]:
                rules_input.append([
                    flatten_with_separator(item["input"]),
                    flatten_with_separator(item["output"])
                ])

            for item in data["test"]:
                test_input.append([
                    flatten_with_separator(item["input"]),
                    flatten_with_separator(item["output"])
                ])

        train_data.append(rules_input)
        test_data.append(test_input)

    return train_data, test_data

# Load training data
training_data_dir = "./data/training"
evaluating_data_dir = "./data/evaluation"

# get all files in training_data_dir that end with .json
training_file_paths = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith('.json')]
evaluating_file_paths = [os.path.join(evaluating_data_dir, f) for f in os.listdir(evaluating_data_dir) if f.endswith('.json')]

training_train_data, training_test_data = load_data(training_file_paths)
evaluating_train_data, evaluating_test_data = load_data(evaluating_file_paths)

save_to_file(training_train_data, "training_train_data.jsonl")
save_to_file(training_test_data, "training_test_data.jsonl")
save_to_file(evaluating_train_data, "evaluating_train_data.jsonl")
save_to_file(evaluating_test_data, "evaluating_test_data.jsonl")

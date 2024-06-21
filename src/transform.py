from hilbert import decode, encode
import numpy as np
import json
import csv
from pathlib import Path

def flatten(matrix):
    return encode(matrix, 2, 32)

def unflatten(flattened):
    return decode(flattened, 2, 32)

class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'training'
        self.eval_dir = self.data_dir / 'evaluation'
        self.max_dim = 0
        self.square_size = 0
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.token_map = {}

    def find_max_dimension_and_extrema(self):
        for directory in [self.train_dir, self.eval_dir]:
            for file in directory.glob('*.json'):
                with open(file, 'r') as f:
                    data = json.load(f)
                    for item in data.get('train', []):
                        input_array = np.array(item['input'])
                        output_array = np.array(item['output'])
                        
                        # Update max dimension
                        self.max_dim = max(self.max_dim,
                                           input_array.shape[0], input_array.shape[1],
                                           output_array.shape[0], output_array.shape[1])
                        
                        # Update global min and max
                        self.global_min = min(self.global_min, input_array.min(), output_array.min())
                        self.global_max = max(self.global_max, input_array.max(), output_array.max())

        # Find the smallest square size that can accommodate the max dimension
        self.square_size = 2 ** int(np.ceil(np.log2(self.max_dim)))
        print(f"Max dimension: {self.max_dim}")
        print(f"Square size: {self.square_size}")
        print(f"Global min: {self.global_min}")
        print(f"Global max: {self.global_max}")

        # Create token map
        self.create_token_map()

    def create_token_map(self):
        for i in range(int(self.global_min), int(self.global_max) + 1):
            self.token_map[i] = f"<TOKEN{i}>"
        self.token_map[-1] = "<BLANK>"  # Special token for padding

    def matrix_to_token_sequence(self, matrix):
        token_sequence = ["<STARTOFMATRIX>"]
        for row in matrix:
            row_tokens = [self.token_map[int(val)] for val in row]
            token_sequence.extend(row_tokens)
            token_sequence.append("\n")  # Add newline after each row
        token_sequence.append("<ENDOFMATRIX>")
        return "".join(token_sequence)

    def pad_matrix(self, matrix):
        matrix_array = np.array(matrix)
        h, w = matrix_array.shape
        padded = np.full((self.square_size, self.square_size), -1, dtype=int)
        padded[:h, :w] = matrix_array
        return padded.tolist()

    def process_data(self, input_dir, output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['prompt', 'completion'])  # CSV header

            item_count = 0
            for file in input_dir.glob('*.json'):
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get('train', []):
                        padded_input = self.pad_matrix(item['input'])
                        padded_output = self.pad_matrix(item['output'])
                        prompt = self.matrix_to_token_sequence(padded_input)
                        completion = self.matrix_to_token_sequence(padded_output)
                        writer.writerow([prompt, completion])
                        item_count += 1

        print(f"Processed {item_count} items from {input_dir} and saved to {output_file}")

    def save_processed_data(self, processed_data, output_file):
        with open(output_file, 'w') as f:
            json.dump(processed_data, f)

def main():
    data_dir = '../data'
    train_output_file = 'processed_train_data.csv'
    eval_output_file = 'processed_eval_data.csv'

    preprocessor = DataPreprocessor(data_dir)
    preprocessor.find_max_dimension_and_extrema()
    
    # Process training data
    preprocessor.process_data(preprocessor.train_dir, train_output_file)
    
    # Process evaluation data
    preprocessor.process_data(preprocessor.eval_dir, eval_output_file)

if __name__ == "__main__":
    main()
from hilbert import decode, encode
import numpy as np
import json
import os
from pathlib import Path


def flatten(matrix):
    return encode(matrix, 2, 32)

def unflatten(self, flattened):
    return decode(flattened, 2, 32)

class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'training'
        self.eval_dir = self.data_dir / 'evaluation'
        self.max_dim = 0
        self.square_size = 0

    def find_max_dimension(self):
        for directory in [self.train_dir, self.eval_dir]:
            for file in directory.glob('*.json'):
                with open(file, 'r') as f:
                    data = json.load(f)
                    for item in data.get('train', []):
                        input_shape = np.array(item['input']).shape
                        output_shape = np.array(item['output']).shape
                        self.max_dim = max(self.max_dim, 
                                           input_shape[0], input_shape[1],
                                           output_shape[0], output_shape[1])
        
        # Find the smallest square size that can accommodate the max dimension
        self.square_size = 2 ** int(np.ceil(np.log2(self.max_dim)))
        print(f"Max dimension: {self.max_dim}")
        print(f"Square size: {self.square_size}")

    def pad_matrix(self, matrix):
        matrix_array = np.array(matrix)
        h, w = matrix_array.shape
        padded = np.full((self.square_size, self.square_size), -1, dtype=int)
        padded[:h, :w] = matrix_array
        return padded.tolist()

    def process_data(self, input_dir, output_file):
        processed_data = []
        for file in input_dir.glob('*.json'):
            with open(file, 'r') as f:
                data = json.load(f)
                for item in data.get('train', []):
                    processed_item = {
                        'input': self.pad_matrix(item['input']),
                        'output': self.pad_matrix(item['output'])
                    }
                    processed_data.append(processed_item)
        
        self.save_processed_data(processed_data, output_file)
        print(f"Processed {len(processed_data)} items from {input_dir}")

    def save_processed_data(self, processed_data, output_file):
        with open(output_file, 'w') as f:
            json.dump(processed_data, f)

def main():
    data_dir = '../data'
    train_output_file = 'processed_train_data.json'
    eval_output_file = 'processed_eval_data.json'

    preprocessor = DataPreprocessor(data_dir)
    preprocessor.find_max_dimension()
    
    # Process training data
    preprocessor.process_data(preprocessor.train_dir, train_output_file)
    
    # Process evaluation data
    preprocessor.process_data(preprocessor.eval_dir, eval_output_file)

if __name__ == "__main__":
    main()
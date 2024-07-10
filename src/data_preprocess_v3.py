import json
import numpy as np
import os

def create_attention_mask(arr):
    return np.ones_like(arr, dtype=int)

def process_data(items):
    processed = []
    
    for item in items:
        input_arr = np.array(item['input'])
        output_arr = np.array(item['output'])
        
        # Determine the maximum number of rows
        max_rows = max(input_arr.shape[0], output_arr.shape[0])
        
        # Pad arrays to have the same number of rows
        input_padded = np.pad(input_arr, ((0, max_rows - input_arr.shape[0]), (0, 0)), mode='constant')
        output_padded = np.pad(output_arr, ((0, max_rows - output_arr.shape[0]), (0, 0)), mode='constant')
        
        combined = np.concatenate((input_padded, output_padded), axis=1)
        
        rows, cols = combined.shape
        x_pos = np.repeat(np.arange(cols), rows)
        y_pos = np.tile(np.arange(rows), cols)
        
        attention_mask = create_attention_mask(combined)
        
        processed.append({
            'data': combined.flatten().tolist(),
            'x_pos': x_pos.tolist(),
            'y_pos': y_pos.tolist(),
            'attention_mask': attention_mask.flatten().tolist()
        })
    
    return processed

def save_to_jsonl(data, filename, mode='w'):
    with open(filename, mode) as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    print(f"Data appended to {filename}")

# Get all JSON files in the current directory
training_data_dir = "./data/training"
json_files = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith('.json')]

# Initialize empty lists for global train and test data
global_train_data = []
global_test_data = []

# Process each file
for file in json_files:
    print(f"Processing {file}...")
    with open(file, 'r') as f:
        data = json.load(f)
    
    # Process and accumulate test data
    test_processed = process_data(data['test'])
    global_test_data.extend(test_processed)
    
    # Process and accumulate train data
    train_processed = process_data(data['train'])
    global_train_data.extend(train_processed)

# Save global test data
save_to_jsonl(global_test_data, 'global_processed_test_data.jsonl')

# Save global train data
save_to_jsonl(global_train_data, 'global_processed_train_data.jsonl')

print("Processing complete.")
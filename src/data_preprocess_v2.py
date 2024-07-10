import json
import numpy as np
import os

def pad_array(arr, target_shape=(30, 30), pad_value=10):
    result = np.full(target_shape, pad_value)
    result[:arr.shape[0], :arr.shape[1]] = arr
    return result

def create_attention_mask(arr, pad_value=10):
    return (arr != pad_value).astype(int)

def process_data(items):
    processed = []
    
    for item in items:
        input_arr = np.array(item['input'])
        output_arr = np.array(item['output'])
        
        padded_input = pad_array(input_arr)
        padded_output = pad_array(output_arr)
        
        combined = np.concatenate((padded_input, padded_output), axis=1)
        
        x_pos = np.tile(np.arange(30), 60)
        y_pos = np.repeat(np.arange(30), 60)
        
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
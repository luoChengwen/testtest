import tensorflow as tf
import time
import pandas as pd

class CsvDataset(tf.data.Dataset):
    def _generator(file_path):
        # Opening the CSV file
        csv_data = pd.read_csv(file_path)
        num_samples = len(csv_data)
        
        for sample_idx in range(num_samples):
            # Reading data from the CSV file
            # Adjust this line based on your CSV file structure
            data_point = tuple(csv_data.iloc[sample_idx, :].values)
            
            time.sleep(0.015)  # Simulating data loading delay
            
            yield data_point
    
    def __new__(cls, file_path):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Adjust dtype and shape accordingly
            args=(file_path,)
        )

# Example usage
csv_file_path = 'your_csv_file.csv'
dataset = CsvDataset(file_path=csv_file_path)

# Iterate through the dataset
for data_point in dataset:
    print(data_point)

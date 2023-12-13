import tensorflow as tf
import os

def load_data_using_tfdata(folders, dir_path, batch_size=32):
    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    data_generator = {}
    for x in folders:
        is_train = True if x == 'train' else False
        cache = x + '.tfcache'
        dir_extend = dir_path + '/' + x
        list_ds = tf.data.Dataset.list_files(str(dir_extend+'/*'))
        
        # Assuming your data is not images, you might not need a parsing function
        # Modify the lambda function accordingly based on your data format
        labeled_ds = list_ds.map(
            lambda file_path: (your_custom_data_loading_function(file_path), label_from_file_path(file_path)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        
        data_generator[x] = prepare_for_training(
            labeled_ds, cache=cache, shuffle_buffer_size=1000
        )

    return data_generator

# Example functions (replace with your actual data loading and label functions)
def your_custom_data_loading_function(file_path):
    # Implement your data loading logic for non-image data
    return tf.constant(0.0)  # Replace with actual loaded data

def label_from_file_path(file_path):
    # Implement your logic to extract labels from file path
    return tf.constant(0) 

# Create a sample DataFrame
data = {'column1': [1, 2, 3, 4, 5],
        'column2': ['a', 'b', 'c', 'd', 'e']}
df = pd.DataFrame(data)

# Add an index column to the DataFrame
df['index'] = df.index

# Convert the DataFrame to a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(df.to_dict('list'))

# Use enumerate to add indices to each element of the dataset
indexed_dataset = dataset.enumerate()

# Shuffle the dataset
shuffled_dataset = indexed_dataset.shuffle(buffer_size=len(df))

# Extract the shuffled indices and elements
shuffled_indices = shuffled_dataset.map(lambda x: x[0])
shuffled_elements = shuffled_dataset.map(lambda x: x[1])

# Iterate over 'shuffled_indices' and 'shuffled_elements' together
for index, element in zip(shuffled_indices, shuffled_elements):
    original_index = element['index'].numpy()
    original_data = {'column1': element['column1'].numpy(),
                     'column2': element['column2'].numpy()}
    
    print("Original Index:", original_index, "Original Data:", original_data)

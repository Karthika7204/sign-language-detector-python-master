import pickle
import numpy as np

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Expected number of features (42 for 21 landmarks with x and y)
expected_shape = 42
cleaned_data = []
cleaned_labels = []

valid_labels = set(range(20)) 

# Inspect and clean all samples
for i, (sample, label) in enumerate(zip(data_dict['data'], data_dict['labels'])):
    try:
        # Convert label to integer
        label_int = int(label)
        if label_int not in valid_labels:
            print(f"Skipping sample {i}: Invalid label {label}")
            continue

        # Convert sample to NumPy array
        sample_array = np.asarray(sample, dtype=np.float32)
        if sample_array.shape == (expected_shape,) and np.all(np.isfinite(sample_array)):
            cleaned_data.append(sample_array.tolist())  # Store as list for consistency
            cleaned_labels.append(label_int)
        else:
            print(f"Skipping sample {i}: Invalid shape {sample_array.shape}, Sample = {sample}")
    except (ValueError, TypeError) as e:
        print(f"Skipping sample {i}: Error = {e}, Sample = {sample}, Label = {label}")

# Save cleaned dataset
cleaned_dict = {'data': cleaned_data, 'labels': cleaned_labels}
with open('data_cleaned.pickle', 'wb') as f:
    pickle.dump(cleaned_dict, f)

print(f"Cleaned dataset saved. Samples: {len(cleaned_data)}, Labels: {len(cleaned_labels)}")
print("Unique labels in cleaned dataset:", np.unique(cleaned_labels))
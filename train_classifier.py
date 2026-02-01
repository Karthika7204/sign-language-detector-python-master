import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load cleaned dataset
data_dict = pickle.load(open('./data_cleaned.pickle', 'rb'))

# Convert to NumPy arrays
try:
    data = np.asarray(data_dict['data'], dtype=np.float32)
    labels = np.asarray(data_dict['labels'], dtype=np.int32)
except ValueError as e:
    print(f"Error converting data to array: {e}")
    raise

# Verify shapes
print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
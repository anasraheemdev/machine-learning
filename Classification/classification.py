# Step 1: Import the classifier
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Sample dataset (English, Computer Science marks) and labels (1 = Pass, 0 = Fail)
X = [
    [30, 40],   # Fail
    [35, 45],   # Fail
    [80, 85],   # Pass
    [75, 90],   # Pass
    [60, 65],   # Pass
    [20, 25],   # Fail
]

y = [0, 0, 1, 1, 1, 0]

# Step 3: Create the KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Step 4: Train the model
knn.fit(X, y)

# Step 5: Predict result for a new student
new_marks = [[70, 84]]   # English = 50, CS = 55
prediction = knn.predict(new_marks)

print("Prediction (1=Pass, 0=Fail):", prediction)
import pandas as pd 

from pathlib import Path

from sklearn.preprocessing import LabelEncoder

folder_path = Path(__file__).resolve().parent

df = pd.read_csv(folder_path / "Project_448.csv",
                 encoding = "cp1252")

df = df.dropna().reset_index(drop=True)

x = df["LLM_Output"]

y = df["LLM_Name"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print('Total Responses:', len(x))

print("LLM Classes:", encoder.classes_)

print("First Encoder Label:", y[0])


#Train-Test Split

from sklearn.model_selection import GroupShuffleSplit

groups = df["LLM_Input"].str.strip()

print("Unique Questions:", groups.nunique())

splitter = GroupShuffleSplit(n_splits=1, 
                             test_size=0.2, 
                             random_state=42)

train_idx, test_idx = next(splitter.split(x, y, groups))

x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]

y_train, y_test = y[train_idx], y[test_idx]

print("Training Responses:", len(x_train))
print("Testing Responses:", len(x_test))

#Tokenization

import re
import numpy as np
from collections import Counter

#Sentence Tokenization
def sentence_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

word_counts = Counter()

for response in x_train:
    word_counts.update(sentence_tokenize(response))

vocab = {"<PAD>": 0, "<UNK>": 1}

for word in word_counts:
    vocab[word] = len(vocab)


#Set max number of tokens in a response
max_tokens = 100

def enencoder(text):
    tokens = sentence_tokenize(text)
    numbers = [vocab.get(word, 1) for word in tokens]

#Cut off responses longer than 100 tokens
    numbers = numbers[:max_tokens]

#Pad responses shorter than 100 tokens with 0
    numbers += [0] * (max_tokens - len(numbers))

    return numbers

#Encode the training and testing responses
x_train_encoded = np.array(
    [enencoder(response) for response in x_train])

x_test_encoded = np.array(
    [enencoder(response) for response in x_test])

print("Vocabulary Size:", len(vocab))
print("Encoded Training Responses Shape:", x_train_encoded.shape)
print("Encoded Testing Responses Shape:", x_test_encoded.shape)


import torch
import torch.nn as nn

# Make results reproducible
torch.manual_seed(42)

# Define the CNN model
class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()

        # Convert token IDs into numerical vectors
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # Identify local patterns in the text
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Activation function
        self.relu = nn.ReLU()

        # Final classification layer
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):

        # Convert tokens into embeddings
        x = self.embedding(x)

        # Rearrange dimensions for CNN
        x = x.permute(0, 2, 1)

        # Apply convolution and activation
        x = self.relu(self.conv(x))

        # Extract strongest features
        x = torch.amax(x, dim=2)

        # Produce classification scores
        x = self.fc(x)

        return x


# Create CNN model
model = CNNClassifier(
    vocab_size=len(vocab),
    embedding_dim=32,
    num_classes=3
)

# Convert training data into PyTorch tensors
x_train_tensor = torch.tensor(
    x_train_encoded,
    dtype=torch.long
)

# Test whether data passes through CNN
outputs = model(x_train_tensor)

print("CNN Output Shape:", outputs.shape)
print(model)



# CNN Training

# Convert the correct LLM labels into tensors
y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Set the number of training epochs
epochs = 30

# Train the CNN
for epoch in range(epochs):

    model.train()

    # Clear previous gradients
    optimizer.zero_grad()

    # Generate predictions
    predictions = model(x_train_tensor)

    # Calculate training loss
    loss = criterion(
        predictions,
        y_train_tensor
    )

    # Calculate gradients
    loss.backward()

    # Update model weights
    optimizer.step()

    # Display loss every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {loss.item():.4f}"
        )



# CNN Evaluation

# Convert testing data into tensors
x_test_tensor = torch.tensor(
    x_test_encoded,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)

# Switch model to evaluation mode
model.eval()

# Disable gradient calculations
with torch.no_grad():

    # Generate predictions on testing data
    outputs = model(x_test_tensor)

    # Select the class with the highest score
    predicted_labels = torch.argmax(
        outputs,
        dim=1
    )

    # Calculate accuracy
    correct = (
        predicted_labels == y_test_tensor
    ).sum().item()

    accuracy = correct / len(y_test_tensor)

print("Correct Predictions:", correct)
print("Total Testing Responses:", len(y_test_tensor))
print(f"CNN Testing Accuracy: {accuracy:.2%}")


# Calculate CNN training accuracy

model.eval()

with torch.no_grad():

    # Predict labels for training data
    train_outputs = model(x_train_tensor)

    train_predictions = torch.argmax(
        train_outputs,
        dim=1
    )

    # Count correct training predictions
    train_correct = (
        train_predictions == y_train_tensor
    ).sum().item()

    # Calculate training accuracy
    train_accuracy = train_correct / len(y_train_tensor)

print(f"CNN Training Accuracy: {train_accuracy:.2%}")
print(f"CNN Testing Accuracy: {accuracy:.2%}")



from sklearn.metrics import confusion_matrix, classification_report

# Convert predicted labels to NumPy
predicted = predicted_labels.cpu().numpy()

# Display confusion matrix
print("\nCNN Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predicted,
        labels=[0, 1, 2]
    )
)

# Display detailed classification results
print("\nCNN Classification Report:")
print(
    classification_report(
        y_test,
        predicted,
        labels=[0, 1, 2],
        target_names=encoder.classes_,
        zero_division=0
    )
)

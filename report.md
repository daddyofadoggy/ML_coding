# ML Coding Project - Comprehensive Code Walkthrough & Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Classical Machine Learning](#classical-machine-learning)
   - [Linear Regression](#linear-regression)
   - [Logistic Regression](#logistic-regression)
   - [K-Means Clustering](#k-means-clustering)
   - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
   - [Decision Trees](#decision-trees)
   - [Naive Bayes (Bernoulli)](#naive-bayes-bernoulli)
3. [Statistical Methods](#statistical-methods)
   - [Covariance Matrix](#covariance-matrix)
4. [Deep Learning Fundamentals](#deep-learning-fundamentals)
   - [Single Neuron](#single-neuron)
   - [Gradient Descent](#gradient-descent)
   - [Activation Functions](#activation-functions)
   - [Loss Functions](#loss-functions)
   - [Optimizers](#optimizers)
   - [Custom Dense Layer](#custom-dense-layer)
   - [Backpropagation](#backpropagation)
5. [Advanced Deep Learning](#advanced-deep-learning)
   - [2D Convolution](#2d-convolution)
   - [Residual Blocks](#residual-blocks)
   - [Transformer Components](#transformer-components)
   - [Micrograd](#micrograd)
   - [Mixture of Experts (MoE)](#mixture-of-experts-moe)

---

## Project Overview

This repository contains from-scratch implementations of fundamental machine learning and deep learning algorithms in Python. Each implementation focuses on understanding the core mathematical concepts and algorithmic details without relying heavily on high-level libraries.

**Tech Stack:**
- Python 3.x
- NumPy (for numerical computations)
- Matplotlib (for visualizations)
- Jupyter Notebooks (for interactive development)

**Learning Philosophy:**
- Build algorithms from first principles
- Understand the mathematics behind each method
- Implement vectorized solutions for efficiency
- Test on real datasets

---

## Classical Machine Learning

### Linear Regression

**Location:** `linear_regression/linear_regression.ipynb`

**Concept:**
Linear regression models the relationship between dependent variable(s) and independent variable(s) using a linear equation: `y = mx + b` (simple) or `y = X^T·W` (multiple).

**Implementation Steps:**

#### 1. Simple Linear Regression (Least Squares Method)

```python
class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Calculate slope: m = Σ((x - x_mean) * (y - y_mean)) / Σ((x - x_mean)²)
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))

        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, X):
        return [self.slope * x + self.intercept for x in X]
```

**Key Steps:**
1. Calculate mean of X and y
2. Compute slope using covariance formula
3. Compute intercept using means
4. Predict using y = mx + b

#### 2. Vectorized Multiple Linear Regression (Normal Equation)

```python
class LinearRegression:
    def __init__(self):
        self.W = None

    def fit(self, X, y):
        # Add bias term (column of ones)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Normal equation: W = (X^T·X)^(-1)·X^T·y
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.W
```

**Key Improvements:**
- Vectorized operations (faster than loops)
- Handles multiple features
- Uses matrix operations

#### 3. Gradient Descent with Regularization

```python
class LinearRegressionGD:
    def __init__(self, regul=0):
        self.regul = regul  # L2 regularization parameter
        self.W = None

    def fit(self, X, y, lr=0.01, num_iter=1000):
        X = np.hstack([np.ones((len(X), 1)), X])
        self.W = np.zeros(X.shape[1])

        for i in range(num_iter):
            # Forward pass
            y_pred = np.dot(X, self.W)

            # Compute loss with regularization
            cost = np.sum((y_pred - y) ** 2) + self.regul * np.sum(self.W ** 2)

            # Compute gradients
            gradients = 2 * np.dot(X.T, (y_pred - y)) + 2 * self.regul * self.W

            # Update weights
            self.W = self.W - lr * gradients

    def predict(self, X):
        X = np.hstack([np.ones((len(X), 1)), X])
        return np.dot(X, self.W)
```

**Key Features:**
- Iterative optimization (good for large datasets)
- L2 regularization to prevent overfitting
- Learning rate control

**Other Implementations in Directory:**
- `lasso_regression(L1).ipynb` - L1 regularization
- `Contour_plot.ipynb` - Visualization of loss surface
- `linear-regression-assumptions.ipynb` - Statistical assumptions validation

---

### Logistic Regression

**Location:** `logistic_regression/logistic_reg_GD.ipynb`

**Concept:**
Logistic regression is used for binary classification. It uses the sigmoid function to map linear combinations to probabilities between 0 and 1.

**Implementation:**

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_logreg(X, y, learning_rate, iterations):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    losses = []

    for _ in range(iterations):
        # Linear combination
        linear_output = np.dot(X, weights) + bias

        # Apply sigmoid activation
        y_pred = sigmoid(linear_output)

        # Compute loss
        loss = binary_cross_entropy(y, y_pred)
        losses.append(round(loss, 4))

        # Compute gradients
        error = y_pred - y
        dw = np.dot(X.T, error) / n_samples
        db = np.mean(error)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, losses
```

**Step-by-Step Flow:**
1. Initialize weights and bias to zero
2. For each iteration:
   - Compute linear output: `z = X·W + b`
   - Apply sigmoid: `σ(z) = 1/(1+e^(-z))`
   - Calculate binary cross-entropy loss
   - Compute gradients via backpropagation
   - Update weights using gradient descent
3. Return optimized parameters and loss history

**Other Implementations:**
- `logistic_regression_pytorch.ipynb` - PyTorch implementation
- `logistic_reg_classification.ipynb` - Full classification pipeline

---

### K-Means Clustering

**Location:** `k-means/K-Means-Clustering.ipynb`

**Concept:**
K-Means is an unsupervised learning algorithm that partitions data into K clusters by minimizing within-cluster variance.

**Implementation:**

```python
def k_means_clustering(points, k, initial_centroids, max_iterations):

    def find_closest_centroid(points, centroids):
        m, n = len(points), len(points[0])
        idx = np.zeros(m)

        # Assign each point to nearest centroid
        for i in range(m):
            distances = []
            for j in range(len(centroids)):
                # Euclidean distance
                dist = sum((points[i][k] - centroids[j][k])**2 for k in range(n))
                distances.append(dist)
            idx[i] = np.argmin(distances)
        return idx

    def compute_centroids(points, idx, k):
        centroids = []

        # Compute mean for each cluster
        for i in range(k):
            indices = np.where(idx == i)[0]
            cluster_points = [points[j] for j in indices]
            centroids.append(tuple(np.mean(cluster_points, axis=0)))
        return centroids

    # Main K-Means loop
    for iteration in range(max_iterations):
        idx = find_closest_centroid(points, initial_centroids)
        centroids = compute_centroids(points, idx, k)
        initial_centroids = centroids

    return centroids
```

**Algorithm Steps:**
1. **Initialize** K centroids (randomly or using provided initial points)
2. **Assignment Step:** Assign each point to the nearest centroid
3. **Update Step:** Recompute centroids as the mean of assigned points
4. **Repeat** steps 2-3 until convergence or max iterations reached

**Time Complexity:** O(n·k·i·d) where:
- n = number of points
- k = number of clusters
- i = number of iterations
- d = number of dimensions

---

### K-Nearest Neighbors (KNN)

**Location:** `Nearest_Neighbor/Nearest_Neighbor.ipynb`

**Concept:**
KNN is a non-parametric classification algorithm that classifies a data point based on the majority class of its K nearest neighbors.

**Implementation:**

```python
from collections import Counter

class KNN:
    def __init__(self, k, distance='euclidean'):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = []

        for x in X_test:
            # Compute distances to all training points
            if self.distance == 'euclidean':
                distances = np.linalg.norm(self.X_train - x, axis=1)
            else:  # Manhattan distance
                distances = np.sum(np.abs(self.X_train - x), axis=1)

            # Find K nearest neighbors
            knn_indices = np.argsort(distances)[:self.k]
            knn_labels = self.y_train[knn_indices]

            # Majority voting
            label = Counter(knn_labels).most_common(1)[0][0]
            y_pred.append(label)

        return np.array(y_pred)
```

**Distance Metrics:**

| Distance Type | Formula | Use Case |
|--------------|---------|----------|
| **Euclidean** | `√(Σ(x_i - y_i)²)` | Continuous, normalized features; spherical clusters |
| **Manhattan** | `Σ|x_i - y_i|` | Sparse, high-dimensional data; grid-like movement |
| **Minkowski** | `(Σ|x_i - y_i|^p)^(1/p)` | Tunable distance (p=1: Manhattan, p=2: Euclidean) |

**Algorithm Steps:**
1. Store training data (X_train, y_train)
2. For each test point:
   - Calculate distance to all training points
   - Sort distances and select K nearest neighbors
   - Perform majority voting on neighbor labels
   - Assign the most common class label

**Important:** Always scale features before using KNN to prevent features with larger magnitudes from dominating the distance calculation.

---

### Decision Trees

**Location:** `Decision_tree/Decision_Tree_Entropy.ipynb`

**Concept:**
Decision trees recursively split data based on features that maximize information gain (or minimize impurity) to build a tree structure for classification.

**Implementation:**

```python
import math
from collections import Counter

def entropy(examples, target_attr):
    """Calculate Shannon entropy"""
    labels = [example[target_attr] for example in examples]
    label_counts = Counter(labels)
    total = len(labels)

    return -sum((count/total) * math.log2(count/total)
                for count in label_counts.values())

def information_gain(examples, attribute, target_attr):
    """Calculate information gain for an attribute"""
    total_entropy = entropy(examples, target_attr)
    total = len(examples)

    # Group by attribute values
    subsets = {}
    for example in examples:
        key = example[attribute]
        subsets.setdefault(key, []).append(example)

    # Weighted entropy of subsets
    weighted_entropy = sum(
        (len(subset)/total) * entropy(subset, target_attr)
        for subset in subsets.values()
    )

    return total_entropy - weighted_entropy

def learn_decision_tree(examples, attributes, target_attr):
    """Recursively build decision tree using ID3 algorithm"""

    # Base case: no examples
    if not examples:
        return None

    # Base case: no attributes left
    if len(attributes) == 0:
        return majority_class(examples, target_attr)

    # Base case: all examples have same label
    if all_same_class(examples, target_attr):
        return examples[0][target_attr]

    # Select best attribute (max information gain)
    best_attr = max(attributes,
                    key=lambda attr: information_gain(examples, attr, target_attr))

    # Create tree node
    tree = {best_attr: {}}

    # Get distinct values for best attribute
    attr_values = set(ex[best_attr] for ex in examples)

    # Recursively build subtrees
    for value in attr_values:
        subset = [ex for ex in examples if ex[best_attr] == value]
        remaining_attrs = [attr for attr in attributes if attr != best_attr]

        subtree = learn_decision_tree(subset, remaining_attrs, target_attr)
        tree[best_attr][value] = subtree

    return tree
```

**Algorithm Flow:**
1. **Calculate Entropy:** Measure of impurity in the dataset
   - H(S) = -Σ p_i · log₂(p_i)
2. **Calculate Information Gain:** Reduction in entropy after splitting
   - IG(S, A) = H(S) - Σ (|S_v|/|S|) · H(S_v)
3. **Select Best Attribute:** Choose attribute with highest information gain
4. **Split Dataset:** Partition data based on best attribute
5. **Recurse:** Build subtrees for each partition
6. **Stop Conditions:**
   - No examples left
   - No attributes left (return majority class)
   - All examples have same label (return that label)

**Other Implementations:**
- `Decision_tree_gini.ipynb` - Using Gini impurity instead of entropy
- `Decision_tree_gini_depth.ipynb` - With max depth constraint
- `Best_Gini_Split.ipynb` - Optimized splitting for continuous features

---

### Naive Bayes (Bernoulli)

**Location:** `Bernoulli_NaiveBase/Bernouli_NB.ipynb`

**Concept:**
Bernoulli Naive Bayes is used for binary features (e.g., word presence/absence in text classification). It applies Bayes' theorem with the "naive" assumption of feature independence.

**Mathematical Foundation:**
- P(y|X) ∝ P(y) · P(X|y)
- For binary features: P(X|y) = Π P(x_i|y)^x_i · (1-P(x_i|y))^(1-x_i)

**Implementation:**

```python
class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing  # Laplace smoothing
        self.single_class = None

    def forward(self, X, y):
        """Train the model"""
        n_emails, n_words = X.shape
        unique_classes = np.unique(y)

        # Handle single class case
        if len(unique_classes) == 1:
            self.single_class = unique_classes[0]
            return self

        # Calculate prior probabilities: P(class)
        self.classes_ = unique_classes
        n_classes = np.bincount(y)
        self.log_prior = np.log(n_classes / n_emails)

        # Count word occurrences per class
        word_present = np.array([
            X[y == 0].sum(axis=0),
            X[y == 1].sum(axis=0)
        ])

        # Calculate word probabilities with Laplace smoothing
        emails_per_class = n_classes.reshape(-1, 1)
        theta = (word_present + self.smoothing) / (emails_per_class + 2 * self.smoothing)

        # Store log probabilities
        self.log_theta_present_ = np.log(theta)      # P(word=1|class)
        self.log_theta_absent_ = np.log(1 - theta)   # P(word=0|class)

        return self

    def predict(self, X):
        """Predict class labels"""
        if self.single_class is not None:
            return np.full(X.shape[0], self.single_class)

        # Calculate log probability scores
        score = (self.log_prior +
                 X @ self.log_theta_present_.T +
                 (1 - X) @ self.log_theta_absent_.T)

        # Return class with highest score
        return self.classes_[score.argmax(axis=1)]
```

**Step-by-Step Process:**

**Training Phase:**
1. Calculate prior probabilities for each class
2. Count word occurrences in each class
3. Apply Laplace smoothing to avoid zero probabilities
   - `θ = (count + α) / (total + 2α)`
4. Store log probabilities for numerical stability

**Prediction Phase:**
1. For each test example, calculate log probability for each class:
   - `log P(y|X) = log P(y) + Σ[x_i·log P(x_i=1|y) + (1-x_i)·log P(x_i=0|y)]`
2. Select class with maximum log probability

**Why Laplace Smoothing?**
- Prevents zero probabilities for unseen features
- Adds α (typically 1.0) to all counts

---

## Statistical Methods

### Covariance Matrix

**Location:** `covariance/covariance.ipynb`

**Concept:**
Covariance measures the joint variability of two variables. A covariance matrix shows pairwise covariances between multiple variables.

**Implementation:**

```python
def calculate_covariance_matrix(vectors):
    """Calculate covariance matrix for a set of vectors"""
    vec_size = len(vectors)
    cov_mat = np.zeros((vec_size, vec_size))

    def cov(x, y):
        """Calculate covariance between two vectors"""
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        return np.sum((x - x_mean) * (y - y_mean)) / (len(x) - 1)

    # Fill covariance matrix
    for i in range(vec_size):
        for j in range(vec_size):
            cov_mat[i][j] = cov(vectors[i], vectors[j])

    return cov_mat
```

**Formula:**
- Cov(X, Y) = Σ[(x_i - x̄)(y_i - ȳ)] / (n - 1)

**Properties:**
- Diagonal elements: Variance of each variable
- Off-diagonal elements: Covariance between variable pairs
- Symmetric matrix: Cov(X,Y) = Cov(Y,X)

**Use Cases:**
- Principal Component Analysis (PCA)
- Feature correlation analysis
- Portfolio optimization in finance
- Multivariate statistical analysis

---

## Deep Learning Fundamentals

### Single Neuron

**Location:** `Deeplearning/single_neuron.ipynb`

**Concept:**
A single neuron is the basic building block of neural networks. It performs a weighted sum of inputs, adds a bias, and applies an activation function.

**Mathematical Model:**
```
output = activation(Σ(w_i · x_i) + b)
```

**Components:**
1. **Inputs:** x₁, x₂, ..., xₙ
2. **Weights:** w₁, w₂, ..., wₙ (learnable parameters)
3. **Bias:** b (learnable parameter)
4. **Activation Function:** σ (introduces non-linearity)

---

### Gradient Descent

**Location:** `Deeplearning/Gradient_Descent.ipynb`

**Concept:**
Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function by moving in the direction of steepest descent.

**Algorithm:**
```
For each iteration:
  1. Compute loss: L(θ)
  2. Compute gradient: ∇L(θ)
  3. Update parameters: θ = θ - α·∇L(θ)
```

**Types:**
- **Batch Gradient Descent:** Uses entire dataset
- **Stochastic Gradient Descent (SGD):** Uses single example
- **Mini-batch Gradient Descent:** Uses small batch of examples

**Key Parameters:**
- **Learning Rate (α):** Step size (too large → divergence, too small → slow convergence)
- **Iterations/Epochs:** Number of times to iterate through the data

---

### Activation Functions

**Location:** `Deeplearning/Activations.ipynb`

**Common Activation Functions:**

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **Sigmoid** | σ(x) = 1/(1+e^(-x)) | (0, 1) | Binary classification output |
| **Tanh** | tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) | (-1, 1) | Hidden layers (zero-centered) |
| **ReLU** | f(x) = max(0, x) | [0, ∞) | Hidden layers (most common) |
| **Leaky ReLU** | f(x) = max(αx, x), α≈0.01 | (-∞, ∞) | Fixes "dying ReLU" problem |
| **Softmax** | σ(x)ᵢ = e^(xᵢ)/Σe^(xⱼ) | (0, 1), Σ=1 | Multi-class classification |

**Purpose:**
- Introduce non-linearity (enables learning complex patterns)
- Normalize outputs to desired range
- Help with gradient flow during backpropagation

---

### Loss Functions

**Location:** `activation_n_lossfunc/MAE.ipynb`

**Common Loss Functions:**

**1. Mean Absolute Error (MAE):**
```
MAE = (1/n) Σ|y_true - y_pred|
```
- Use for regression
- Less sensitive to outliers than MSE

**2. Mean Squared Error (MSE):**
```
MSE = (1/n) Σ(y_true - y_pred)²
```
- Use for regression
- Penalizes large errors more

**3. Binary Cross-Entropy:**
```
BCE = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
- Use for binary classification
- Measures difference between probability distributions

**4. Categorical Cross-Entropy:**
```
CCE = -(1/n) Σ Σ y_ij·log(ŷ_ij)
```
- Use for multi-class classification
- Works with one-hot encoded labels

---

### Optimizers

**Location:** `Deeplearning/Adam_optimizer.ipynb`

**Adam (Adaptive Moment Estimation):**

Adam combines the benefits of two other optimizers:
- **Momentum:** Uses moving average of gradients
- **RMSprop:** Adapts learning rate for each parameter

**Algorithm:**
```python
# Hyperparameters
beta1 = 0.9      # Exponential decay for first moment
beta2 = 0.999    # Exponential decay for second moment
epsilon = 1e-8   # Small constant for numerical stability

# Initialize
m = 0  # First moment (mean)
v = 0  # Second moment (variance)

for t in 1 to iterations:
    g = compute_gradient()

    # Update biased moments
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g²

    # Bias correction
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)

    # Update parameters
    theta = theta - alpha * m_hat / (√v_hat + epsilon)
```

**Advantages:**
- Adaptive learning rates for each parameter
- Works well with sparse gradients
- Requires little tuning (default parameters work well)
- Currently one of the most popular optimizers

---

### Custom Dense Layer

**Location:** `Deeplearning/Custom_dense_layer.ipynb`

**Concept:**
A dense (fully-connected) layer where every neuron connects to every neuron in the previous layer.

**Implementation Pattern:**
```python
class DenseLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        # Z = X·W + b
        self.X = X  # Cache for backprop
        return np.dot(X, self.W) + self.b

    def backward(self, dZ, learning_rate):
        # Compute gradients
        m = self.X.shape[0]
        dW = np.dot(self.X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W.T)

        # Update parameters
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX  # Pass to previous layer
```

**Key Operations:**
1. **Forward Pass:** Linear transformation `Z = XW + b`
2. **Backward Pass:** Compute gradients and update weights
3. **Parameter Update:** Gradient descent step

---

### Backpropagation

**Location:** `Deeplearning/SingleNeuronBackprop.ipynb`

**Concept:**
Backpropagation is the algorithm for computing gradients in neural networks using the chain rule of calculus.

**Process:**
1. **Forward Pass:** Compute outputs and cache intermediate values
2. **Compute Loss:** Calculate error at output
3. **Backward Pass:** Propagate gradients from output to input
4. **Update Weights:** Apply gradient descent

**Chain Rule Example:**
For a composition f(g(h(x))):
```
df/dx = (df/dg) · (dg/dh) · (dh/dx)
```

**Gradient Flow:**
```
Loss → Output Layer → Hidden Layer(s) → Input Layer
  ↓         ↓              ↓                ↓
  dL/dL    dL/dz₂         dL/dz₁          dL/dX
```

---

## Advanced Deep Learning

### 2D Convolution

**Location:** `Deeplearning/convolution2D.ipynb`

**Concept:**
Convolutional layers apply learnable filters to input data, extracting spatial features. Essential for computer vision tasks.

**Operation:**
```
For each position (i, j) in output:
  output[i,j] = Σ Σ input[i+m, j+n] · kernel[m, n] + bias
```

**Components:**
- **Kernel/Filter:** Small matrix of learnable weights
- **Stride:** Step size for sliding the kernel
- **Padding:** Border pixels added to input
- **Channels:** Input depth (e.g., RGB = 3 channels)

**Example:**
```
Input: 32×32×3 image
Kernel: 5×5×3 filter (64 filters)
Output: 28×28×64 feature map
```

**Advantages:**
- **Parameter Sharing:** Same kernel applied across image
- **Translation Invariance:** Detects features regardless of position
- **Hierarchical Learning:** Early layers detect edges, later layers detect complex patterns

---

### Residual Blocks

**Location:** `Deeplearning/simple_residual_block.ipynb`

**Concept:**
Residual connections (skip connections) allow gradients to flow directly through the network, enabling training of very deep networks.

**Architecture:**
```
Input → [Conv → BN → ReLU → Conv → BN] → Add → ReLU → Output
  ↓                                          ↑
  └──────────────── Skip Connection ─────────┘
```

**Mathematical Form:**
```
y = F(x) + x
```
Where F(x) is the residual mapping to be learned.

**Benefits:**
- **Easier Optimization:** Network can learn identity mapping
- **Gradient Flow:** Gradients flow through skip connections
- **Deeper Networks:** Enables training of 100+ layer networks (ResNet)

---

### Transformer Components

**Location:** `Deeplearning/transformer/`

#### Self-Attention

**File:** `self-attention.ipynb`

**Concept:**
Self-attention allows each position in a sequence to attend to all positions, capturing dependencies regardless of distance.

**Mechanism:**
```python
# Input: X (sequence_length, d_model)

# 1. Project to Query, Key, Value
Q = X @ W_Q  # (seq_len, d_k)
K = X @ W_K  # (seq_len, d_k)
V = X @ W_V  # (seq_len, d_v)

# 2. Compute attention scores
scores = Q @ K.T / sqrt(d_k)  # (seq_len, seq_len)

# 3. Apply softmax
attention_weights = softmax(scores)  # (seq_len, seq_len)

# 4. Weighted sum of values
output = attention_weights @ V  # (seq_len, d_v)
```

**Intuition:**
- **Query:** What I'm looking for
- **Key:** What I have to offer
- **Value:** What I actually send
- **Attention:** How much to focus on each position

#### Multi-Head Attention

**File:** `multi-head-attention.ipynb`

**Concept:**
Run multiple attention mechanisms in parallel, allowing the model to attend to different representation subspaces.

**Process:**
```
For each head h:
  head_h = Attention(X·W_Q^h, X·W_K^h, X·W_V^h)

Concat all heads and project:
  MultiHead(X) = Concat(head_1, ..., head_h) · W_O
```

**Benefits:**
- Capture different types of relationships
- More expressive representation
- Parallel computation

#### Masked Self-Attention

**File:** `masked_self_attention.ipynb`

**Concept:**
Prevents positions from attending to future positions (for autoregressive models like GPT).

**Implementation:**
```python
# Create causal mask
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

# Apply mask before softmax
scores = Q @ K.T / sqrt(d_k) + mask
attention_weights = softmax(scores)
```

**Use Case:** Language modeling where future tokens must be hidden

#### Positional Encoding

**File:** `Pos_encoding.ipynb`

**Concept:**
Since transformers have no inherent notion of position, we add positional encodings to give the model information about token positions.

**Formula:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties:**
- Unique encoding for each position
- Relative position information
- Deterministic (not learned)

#### Layer Normalization

**File:** `Layer_Norm.ipynb`

**Concept:**
Normalize activations across features for each example independently (unlike batch norm which normalizes across batch).

**Formula:**
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```
Where:
- μ = mean across features
- σ² = variance across features
- γ, β = learnable parameters

**Benefits:**
- Stabilizes training
- Allows higher learning rates
- Works well with variable-length sequences

---

### Micrograd

**Location:** `Deeplearning/micrograd_second_half.ipynb`

**Concept:**
Micrograd is a tiny autograd engine that implements backpropagation over a dynamically built DAG (Directed Acyclic Graph).

**Core Components:**

**1. Value Object:**
```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def backward(self):
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backpropagate
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

**2. Operations:**
- Addition, Multiplication, Power
- ReLU, Tanh activation functions
- Automatic gradient computation

**Key Features:**
- **Scalar-valued autograd:** Operates on scalar values
- **Dynamic computation graph:** Built on-the-fly during forward pass
- **Educational:** Clear implementation of backpropagation
- **Minimal:** ~150 lines of code for full autograd engine

**Learning Value:**
Understanding micrograd helps you understand how PyTorch and TensorFlow work under the hood.

---

### Mixture of Experts (MoE)

**Location:** `Deeplearning/MoE/`

**Concept:**
MoE is an architecture that uses multiple specialized "expert" networks and a gating network to route inputs to the most relevant experts.

**Architecture:**
```
Input → Gating Network → Routing Weights
         ↓                      ↓
    [Expert 1]             weight_1
    [Expert 2]    →        weight_2
    [Expert 3]             weight_3
         ↓                      ↓
    Weighted Sum of Expert Outputs → Final Output
```

**Components:**

**1. Gating Network:**
```python
gate_weights = softmax(W_gate @ input)
```
Determines which experts to use for each input

**2. Expert Networks:**
```python
expert_outputs = [expert_i(input) for expert_i in experts]
```
Specialized networks trained on different aspects

**3. Combination:**
```python
output = sum(gate_weights[i] * expert_outputs[i] for i in range(n_experts))
```

**Benefits:**
- **Scalability:** Add more experts without changing architecture
- **Specialization:** Experts learn different patterns
- **Efficiency:** Only activate relevant experts (sparse MoE)
- **Capacity:** Increase model capacity without proportional compute increase

**Use Cases:**
- Large language models (GPT-4, Switch Transformer)
- Multi-task learning
- Domain adaptation

---

## Project Flow & Best Practices

### Typical ML Project Workflow

1. **Data Preparation**
   - Load and explore data
   - Handle missing values
   - Feature scaling/normalization
   - Train/test split

2. **Model Selection**
   - Choose appropriate algorithm for the task
   - Consider data size, feature types, problem complexity

3. **Implementation**
   - Implement from scratch or use library
   - Initialize parameters
   - Define forward pass

4. **Training**
   - Define loss function
   - Implement backpropagation (for neural networks)
   - Choose optimizer and learning rate
   - Train for multiple epochs

5. **Evaluation**
   - Test on unseen data
   - Calculate metrics (accuracy, precision, recall, F1, MSE, etc.)
   - Visualize results

6. **Optimization**
   - Hyperparameter tuning
   - Cross-validation
   - Regularization
   - Feature engineering

### Code Quality Practices in This Repository

1. **Vectorization:** Use NumPy operations instead of loops for efficiency
2. **Modularity:** Break code into reusable functions and classes
3. **Documentation:** Clear comments explaining mathematical concepts
4. **Testing:** Verify implementations with simple test cases
5. **Visualization:** Plot results to understand behavior

---

## Learning Path Recommendation

### Beginner Path
1. Start with **Linear Regression** - understand the basics
2. Move to **Logistic Regression** - introduce classification
3. Study **K-Means** - unsupervised learning
4. Learn **KNN** - simple but powerful classifier
5. Explore **Decision Trees** - intuitive tree-based models

### Intermediate Path
6. Understand **Naive Bayes** - probabilistic methods
7. Study **Gradient Descent** - foundation of optimization
8. Learn **Activation Functions** - building blocks of neural networks
9. Implement **Single Neuron** - basic neural network unit
10. Master **Backpropagation** - training neural networks

### Advanced Path
11. Explore **Custom Dense Layers** - build neural networks from scratch
12. Study **Adam Optimizer** - modern optimization
13. Learn **Convolution** - computer vision foundation
14. Understand **Residual Blocks** - deep networks
15. Master **Transformers** - state-of-the-art architecture
16. Dive into **Micrograd** - understand autograd engines
17. Explore **Mixture of Experts** - scaling large models

---

## Key Mathematical Concepts

### Linear Algebra
- Matrix multiplication
- Transpose operations
- Inverse matrices
- Eigenvalues and eigenvectors

### Calculus
- Derivatives and gradients
- Chain rule (essential for backpropagation)
- Partial derivatives
- Optimization

### Probability & Statistics
- Probability distributions
- Bayes' theorem
- Maximum likelihood estimation
- Expectation and variance

### Information Theory
- Entropy
- Information gain
- Cross-entropy
- KL divergence

---

## Common Patterns Across Implementations

### 1. Initialization
```python
# Random initialization for weights
W = np.random.randn(input_dim, output_dim) * 0.01
b = np.zeros((1, output_dim))
```

### 2. Forward Pass
```python
def forward(X):
    Z = X @ W + b  # Linear transformation
    A = activation(Z)  # Non-linearity
    return A
```

### 3. Loss Computation
```python
loss = loss_function(y_true, y_pred)
```

### 4. Backward Pass
```python
dZ = y_pred - y_true  # Gradient of loss
dW = X.T @ dZ / m  # Gradient of weights
db = np.mean(dZ, axis=0)  # Gradient of bias
```

### 5. Parameter Update
```python
W -= learning_rate * dW
b -= learning_rate * db
```

---

## File Organization

```
ML_coding/
├── linear_regression/          # Linear models
│   ├── linear_regression.ipynb
│   ├── lasso_regression(L1).ipynb
│   └── Contour_plot.ipynb
├── logistic_regression/        # Classification
│   ├── logistic_reg_GD.ipynb
│   └── logistic_regression_pytorch.ipynb
├── k-means/                    # Clustering
│   └── K-Means-Clustering.ipynb
├── Nearest_Neighbor/           # Instance-based learning
│   └── Nearest_Neighbor.ipynb
├── Decision_tree/              # Tree-based models
│   ├── Decision_Tree_Entropy.ipynb
│   ├── Decision_tree_gini.ipynb
│   └── Best_Gini_Split.ipynb
├── Bernoulli_NaiveBase/        # Probabilistic models
│   └── Bernouli_NB.ipynb
├── covariance/                 # Statistical methods
│   └── covariance.ipynb
├── Deeplearning/               # Neural networks
│   ├── single_neuron.ipynb
│   ├── Gradient_Descent.ipynb
│   ├── Activations.ipynb
│   ├── Adam_optimizer.ipynb
│   ├── Custom_dense_layer.ipynb
│   ├── convolution2D.ipynb
│   ├── simple_residual_block.ipynb
│   ├── micrograd_second_half.ipynb
│   ├── transformer/            # Attention mechanisms
│   │   ├── self-attention.ipynb
│   │   ├── multi-head-attention.ipynb
│   │   ├── masked_self_attention.ipynb
│   │   ├── Pos_encoding.ipynb
│   │   └── Layer_Norm.ipynb
│   └── MoE/                    # Mixture of Experts
└── activation_n_lossfunc/      # Loss functions
    └── MAE.ipynb
```

---

## Dependencies

```python
# Core libraries
numpy          # Numerical computing
matplotlib     # Visualization
jupyter        # Interactive notebooks

# Optional (for some notebooks)
sklearn        # For dataset loading and validation
torch          # PyTorch implementations
pandas         # Data manipulation
```

---

## Running the Code

### Setup
```bash
# Clone repository
git clone <repository-url>
cd ML_coding

# Install dependencies
pip install numpy matplotlib jupyter scikit-learn torch
```

### Running Notebooks
```bash
# Start Jupyter
jupyter notebook

# Navigate to desired notebook and run cells
```

### Running Python Files
```bash
python <script_name>.py
```

---

## Further Resources

### Books
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Deep Learning" by Goodfellow, Bengio, and Courville
- "Hands-On Machine Learning" by Aurélien Géron

### Online Courses
- Andrew Ng's Machine Learning (Coursera)
- Deep Learning Specialization (Coursera)
- Fast.ai Practical Deep Learning

### Papers
- "Attention is All You Need" (Transformers)
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Adam: A Method for Stochastic Optimization"

---

## Conclusion

This repository provides a comprehensive foundation in machine learning and deep learning through hands-on implementation. By building algorithms from scratch, you gain deep understanding of:

1. **Mathematical foundations** - The equations aren't just formulas, they're logic
2. **Algorithm mechanics** - How and why each step works
3. **Implementation details** - Practical considerations and optimizations
4. **Design patterns** - Common structures across different algorithms

The journey from linear regression to transformers mirrors the evolution of ML/DL, giving you both historical context and practical skills.

**Next Steps:**
- Extend implementations with more features
- Apply to real-world datasets
- Compare with library implementations (scikit-learn, PyTorch)
- Build end-to-end projects combining multiple techniques

Happy Learning! 🚀

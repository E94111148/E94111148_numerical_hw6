import numpy as np

# matrix A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

n = A.shape[0]

# Create augmented matrix [A | I]
aug = np.hstack([A, np.identity(n)])

# Perform Gauss-Jordan elimination
for i in range(n):
    # Partial pivoting
    max_row = np.argmax(abs(aug[i:, i])) + i
    if i != max_row:
        aug[[i, max_row]] = aug[[max_row, i]]
    
    # Normalize pivot row
    aug[i] = aug[i] / aug[i, i]
    
    # Eliminate column entries
    for j in range(n):
        if j != i:
            factor = aug[j, i]
            aug[j] -= factor * aug[i]

# Extract the inverse from the augmented matrix
A_inv = aug[:, n:]

# Print the result
print("Inverse of matrix A:")
print(np.round(A_inv, decimals=6))

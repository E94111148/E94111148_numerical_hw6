import numpy as np

# Define the tri-diagonal system
a = np.array([0, -1, -1, -1], dtype=float)       # sub-diagonal (starts from index 1)
b_diag = np.array([3, 3, 3, 3], dtype=float)     # main diagonal
c = np.array([-1, -1, -1, 0], dtype=float)       # super-diagonal (ends at index n-2)
d = np.array([2, 3, 4, 1], dtype=float)          # right-hand side vector

n = len(b_diag)

# Arrays for L, U, z, and x
l = np.zeros(n)
u = np.zeros(n-1)
z = np.zeros(n)
x = np.zeros(n)

# Step 1: LU Decomposition (Crout)
l[0] = b_diag[0]
u[0] = c[0] / l[0]

for i in range(1, n-1):
    l[i] = b_diag[i] - a[i] * u[i-1]
    u[i] = c[i] / l[i]

l[n-1] = b_diag[n-1] - a[n-1] * u[n-2]

# Step 2: Forward substitution to solve Lz = d
z[0] = d[0] / l[0]
for i in range(1, n):
    z[i] = (d[i] - a[i] * z[i-1]) / l[i]

# Step 3: Back substitution to solve Ux = z
x[n-1] = z[n-1]
for i in range(n-2, -1, -1):
    x[i] = z[i] - u[i] * x[i+1]

# Print the solution
print("Solution using Crout factorization for tri-diagonal matrix:")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")

# Verify A * x = d
A = np.array([
    [3, -1,  0,  0],
    [-1, 3, -1,  0],
    [0, -1,  3, -1],
    [0,  0, -1,  3]
], dtype=float)

Ax = A @ x

#

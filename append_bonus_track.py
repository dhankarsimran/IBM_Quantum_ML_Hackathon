import nbformat
import sys

notebook_path = "/Users/simran/Desktop/Folder/Start_From_Here.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Create new markdown and code cells
md_cell = nbformat.v4.new_markdown_cell("""## Bonus Track: Geometric Difference & Model Complexity

Not all datasets offer potential advantage from the use of PQKs. There are some theoretical bounds that one can use as a preliminary test to see if a particular dataset can benefit from PQKs. To quantify this, authors of Power of data in quantum machine learning define quantities referred to as classical and quantum model complexities and geometric separation of the classical and quantum models. To expect a potential quantum advantage from PQKs, the geometric separation between the classical and quantum-projected kernels should be approximately on the order of $\sqrt{N}$, where $N$ is the number of training samples.

If this condition is satisfied, we move on to checking the model complexities. If the classical model complexity is on the order of $N$ while the quantum-projected model complexity is substantially smaller than $N$, we can expect potential advantage from the PQK.
""")

geometric_separation_code = """# Gamma values used in best models above
gamma_c = svm_classical.best_params_["gamma"]
gamma_q = svm_quantum.best_params_["gamma"]

# Regularization parameter used in the best classical model above
C_c = svm_classical.best_params_["C"]
l_c = 1 / C_c

# Classical and quantum kernels used above
K_c = rbf_kernel(train_data, train_data, gamma=gamma_c)
K_q = rbf_kernel(projections_train, projections_train, gamma=gamma_q)

# Intermediate matrices in the equation
K_c_sqrt = sqrtm(K_c)
K_q_sqrt = sqrtm(K_q)
K_c_inv = inv(K_c + l_c * np.eye(K_c.shape[0]))

K_multiplication = (
    K_q_sqrt @ K_c_sqrt @ K_c_inv @ K_c_inv @ K_c_sqrt @ K_q_sqrt
)

# Geometric separation
norm = np.linalg.norm(K_multiplication, ord=np.inf)
g_cq = np.sqrt(norm)
print(f"Geometric separation between classical and quantum kernels is {g_cq:.4f}")
print(f"sqrt(N) for training set is {np.sqrt(len(train_data)):.4f}")
"""
geometric_separation_cell = nbformat.v4.new_code_cell(geometric_separation_code)

classical_complexity_code = """# Model complexity of the classical kernel
# Number of training data
N = len(train_data)

# Predicted labels
pred_labels = svm_classical.best_estimator_.predict(train_data)
pred_matrix = np.outer(pred_labels, pred_labels)

# Intermediate terms
K_c_inv = inv(K_c + l_c * np.eye(K_c.shape[0]))

# First term
first_sum = np.sum((K_c_inv @ K_c_inv) * pred_matrix)
first_term = l_c * np.sqrt(first_sum / N)

# Second term
second_sum = np.sum((K_c_inv @ K_c @ K_c_inv) * pred_matrix)
second_term = np.sqrt(second_sum / N)

# Model complexity
s_c = first_term + second_term
print(f"Classical model complexity is {s_c:.4f}")
"""
classical_complexity_cell = nbformat.v4.new_code_cell(classical_complexity_code)

quantum_complexity_code = """# Model complexity of the projected quantum kernel
# Number of training data
N = len(projections_train)

# Predicted labels
pred_labels = svm_quantum.best_estimator_.predict(projections_train)
pred_matrix = np.outer(pred_labels, pred_labels)

# Regularization parameter used in the best quantum model above
C_q = svm_quantum.best_params_["C"]
l_q = 1 / C_q

# Intermediate terms
K_q_inv = inv(K_q + l_q * np.eye(K_q.shape[0]))

# First term
first_sum = np.sum((K_q_inv @ K_q_inv) * pred_matrix)
first_term = l_q * np.sqrt(first_sum / N)

# Second term
second_sum = np.sum((K_q_inv @ K_q @ K_q_inv) * pred_matrix)
second_term = np.sqrt(second_sum / N)

# Model complexity
s_q = first_term + second_term
print(f"Quantum model complexity is {s_q:.4f}")
"""
quantum_complexity_cell = nbformat.v4.new_code_cell(quantum_complexity_code)


nb.cells.extend([md_cell, geometric_separation_cell, classical_complexity_cell, quantum_complexity_cell])

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Successfully appended bonus track cells to {notebook_path}")

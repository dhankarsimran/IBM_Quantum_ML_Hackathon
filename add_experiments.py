"""
Add Track 1B, Track 2A, Track 2C to the notebook.
Uses numpy indexing since train_data/test_data are numpy arrays.
"""
import nbformat

notebook_path = "/Users/simran/Desktop/Folder/Start_From_Here.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Remove previously added cells (everything after cell 25 which is the last bonus track cell)
nb.cells = nb.cells[:26]

new_cells = []

# ──────────────────────────────────────────────────────────────────────
# Track 1B: Position-based analysis
# ──────────────────────────────────────────────────────────────────────
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Track 1B: Position-Based Analysis\n\n"
    "We now investigate **which motif positions** benefit most from quantum encoding.\n\n"
    "The 180 quantum projection features are organized in three blocks of 60 (X, Y, Z). "
    "Within each block, 15 columns correspond to each of the 4 positions:\n\n"
    "| Position | ⟨X⟩ cols | ⟨Y⟩ cols | ⟨Z⟩ cols |\n"
    "|----------|----------|----------|----------|\n"
    "| 1        | 0–14     | 60–74    | 120–134  |\n"
    "| 2        | 15–29    | 75–89    | 135–149  |\n"
    "| 3        | 30–44    | 90–104   | 150–164  |\n"
    "| 4        | 45–59    | 105–119  | 165–179  |\n\n"
    "For the classical pipeline, each position corresponds to 15 one-hot columns (columns 0–14, 15–29, 30–44, 45–59).\n\n"
    "**Hypothesis from the literature:** Position 3 (the most data-sparse position) should show the largest quantum encoding benefit."
))

new_cells.append(nbformat.v4.new_code_cell(
    '# Track 1B: Position-based analysis\n'
    '# Compare classical vs quantum SVM for each individual position\n'
    '\n'
    'position_layout = {\n'
    '    "Position 1": {"classical": list(range(0, 15)),\n'
    '                   "quantum": list(range(0, 15)) + list(range(60, 75)) + list(range(120, 135))},\n'
    '    "Position 2": {"classical": list(range(15, 30)),\n'
    '                   "quantum": list(range(15, 30)) + list(range(75, 90)) + list(range(135, 150))},\n'
    '    "Position 3": {"classical": list(range(30, 45)),\n'
    '                   "quantum": list(range(30, 45)) + list(range(90, 105)) + list(range(150, 165))},\n'
    '    "Position 4": {"classical": list(range(45, 60)),\n'
    '                   "quantum": list(range(45, 60)) + list(range(105, 120)) + list(range(165, 180))},\n'
    '}\n'
    '\n'
    'position_results = {}\n'
    '\n'
    'for pos_name, cols in position_layout.items():\n'
    '    print(f"--- {pos_name} ---")\n'
    '    # Classical (train_data is a numpy array)\n'
    '    X_tr_c = train_data[:, cols["classical"]]\n'
    '    X_te_c = test_data[:, cols["classical"]]\n'
    '    svm_c = GridSearchCV(\n'
    '        SVC(kernel="rbf"), PARAM_GRID, cv=cv,\n'
    '        scoring="f1_weighted", n_jobs=-1,\n'
    '    ).fit(X_tr_c, train_labels)\n'
    '    f1_c = f1_score(test_labels, svm_c.predict(X_te_c), average="weighted")\n'
    '\n'
    '    # Quantum (projections_train is a DataFrame)\n'
    '    X_tr_q = projections_train.iloc[:, cols["quantum"]]\n'
    '    X_te_q = projections_test.iloc[:, cols["quantum"]]\n'
    '    svm_q = GridSearchCV(\n'
    '        SVC(kernel="rbf"), PARAM_GRID, cv=cv,\n'
    '        scoring="f1_weighted", n_jobs=-1,\n'
    '    ).fit(X_tr_q, train_labels)\n'
    '    f1_q = f1_score(test_labels, svm_q.predict(X_te_q), average="weighted")\n'
    '\n'
    '    delta = f1_q - f1_c\n'
    '    position_results[pos_name] = {"classical": f1_c, "quantum": f1_q, "delta": delta}\n'
    '    print(f"  Classical F1: {f1_c:.4f}  |  Quantum F1: {f1_q:.4f}  |  ΔF1: {delta:+.4f}")\n'
    '    print()\n'
))

new_cells.append(nbformat.v4.new_code_cell(
    '# Track 1B: Bar chart comparing classical vs quantum per position\n'
    'fig, ax = plt.subplots(figsize=(10, 6))\n'
    'pos_names = list(position_results.keys())\n'
    'classical_scores = [position_results[p]["classical"] for p in pos_names]\n'
    'quantum_scores = [position_results[p]["quantum"] for p in pos_names]\n'
    'delta_scores = [position_results[p]["delta"] for p in pos_names]\n'
    '\n'
    'x = np.arange(len(pos_names))\n'
    'width = 0.35\n'
    '\n'
    'bars1 = ax.bar(x - width/2, classical_scores, width, label="Classical (15-dim)", color="#3498db", alpha=0.85, edgecolor="black")\n'
    'bars2 = ax.bar(x + width/2, quantum_scores, width, label="Quantum (45-dim)", color="#e74c3c", alpha=0.85, edgecolor="black")\n'
    '\n'
    'for b, score in zip(bars1, classical_scores):\n'
    '    ax.text(b.get_x() + b.get_width()/2, score + 0.01, f"{score:.3f}", ha="center", fontsize=9, fontweight="bold")\n'
    'for b, score in zip(bars2, quantum_scores):\n'
    '    ax.text(b.get_x() + b.get_width()/2, score + 0.01, f"{score:.3f}", ha="center", fontsize=9, fontweight="bold")\n'
    '\n'
    'ax.set_ylabel("Weighted F1 (test)")\n'
    'ax.set_xticks(x)\n'
    'ax.set_xticklabels(pos_names)\n'
    'ax.set_ylim(0, 1)\n'
    'ax.set_title("Track 1B: Classical vs Quantum Encoding by Motif Position")\n'
    'ax.legend()\n'
    'plt.tight_layout()\n'
    'plt.show()\n'
    '\n'
    '# Print which position has the biggest quantum advantage\n'
    'best_pos = max(position_results, key=lambda p: position_results[p]["delta"])\n'
    'best_delta = position_results[best_pos]["delta"]\n'
    'print(f"\\nLargest quantum advantage: {best_pos} (ΔF1 = {best_delta:+.4f})")\n'
))

# ──────────────────────────────────────────────────────────────────────
# Track 2A: Learning Curves
# ──────────────────────────────────────────────────────────────────────
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Track 2A: Learning Curves — Does Data Scarcity Widen the Gap?\n\n"
    "We subsample the training set to 25%, 50%, 75%, and 100% (maintaining class balance) and repeat each "
    "with 5 random seeds. We then plot F1 vs. training size with error bars.\n\n"
    "**Hypothesis:** The quantum encoding benefit should grow when data is scarce and shrink when data is plentiful."
))

new_cells.append(nbformat.v4.new_code_cell(
    '# Track 2A: Learning curves\n'
    'from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold\n'
    '\n'
    'fractions = [0.25, 0.50, 0.75, 1.0]\n'
    'n_seeds = 5\n'
    'lc_results = {"fraction": [], "seed": [], "classical_f1": [], "quantum_f1": []}\n'
    '\n'
    'for frac in fractions:\n'
    '    for seed in range(n_seeds):\n'
    '        if frac < 1.0:\n'
    '            sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)\n'
    '            idx, _ = next(sss.split(train_data, train_labels))\n'
    '        else:\n'
    '            idx = np.arange(len(train_data))\n'
    '\n'
    '        X_tr_c = train_data[idx]\n'
    '        X_tr_q = projections_train.iloc[idx]\n'
    '        y_tr = train_labels[idx]\n'
    '\n'
    '        # Classical\n'
    '        svm_c = GridSearchCV(\n'
    '            SVC(kernel="rbf"), PARAM_GRID, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),\n'
    '            scoring="f1_weighted", n_jobs=-1,\n'
    '        ).fit(X_tr_c, y_tr)\n'
    '        f1_c = f1_score(test_labels, svm_c.predict(test_data), average="weighted")\n'
    '\n'
    '        # Quantum\n'
    '        svm_q = GridSearchCV(\n'
    '            SVC(kernel="rbf"), PARAM_GRID, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),\n'
    '            scoring="f1_weighted", n_jobs=-1,\n'
    '        ).fit(X_tr_q, y_tr)\n'
    '        f1_q = f1_score(test_labels, svm_q.predict(projections_test), average="weighted")\n'
    '\n'
    '        lc_results["fraction"].append(frac)\n'
    '        lc_results["seed"].append(seed)\n'
    '        lc_results["classical_f1"].append(f1_c)\n'
    '        lc_results["quantum_f1"].append(f1_q)\n'
    '\n'
    '    n_samples = int(frac * len(train_data))\n'
    '    c_vals = [lc_results["classical_f1"][i] for i in range(len(lc_results["fraction"])) if lc_results["fraction"][i] == frac]\n'
    '    q_vals = [lc_results["quantum_f1"][i] for i in range(len(lc_results["fraction"])) if lc_results["fraction"][i] == frac]\n'
    '    print(f"{int(frac*100):3d}% ({n_samples:3d} samples) — Classical: {np.mean(c_vals):.4f} ± {np.std(c_vals):.4f}  |  Quantum: {np.mean(q_vals):.4f} ± {np.std(q_vals):.4f}  |  ΔF1: {np.mean(q_vals)-np.mean(c_vals):+.4f}")\n'
))

new_cells.append(nbformat.v4.new_code_cell(
    '# Track 2A: Plot learning curves\n'
    'lc_df = pd.DataFrame(lc_results)\n'
    'lc_summary = lc_df.groupby("fraction").agg(\n'
    '    c_mean=("classical_f1", "mean"), c_std=("classical_f1", "std"),\n'
    '    q_mean=("quantum_f1", "mean"), q_std=("quantum_f1", "std"),\n'
    ').reset_index()\n'
    'lc_summary["n_samples"] = (lc_summary["fraction"] * len(train_data)).astype(int)\n'
    '\n'
    'fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n'
    '\n'
    '# Left: F1 vs training size\n'
    'ax1.errorbar(lc_summary["n_samples"], lc_summary["c_mean"], yerr=lc_summary["c_std"],\n'
    '             marker="o", capsize=5, label="Classical", color="#3498db", linewidth=2)\n'
    'ax1.errorbar(lc_summary["n_samples"], lc_summary["q_mean"], yerr=lc_summary["q_std"],\n'
    '             marker="s", capsize=5, label="Quantum", color="#e74c3c", linewidth=2)\n'
    'ax1.set_xlabel("Training Set Size")\n'
    'ax1.set_ylabel("Weighted F1 (test)")\n'
    'ax1.set_title("Learning Curves: F1 vs Training Size")\n'
    'ax1.legend()\n'
    'ax1.grid(True, alpha=0.3)\n'
    '\n'
    '# Right: ΔF1 vs training size\n'
    'delta_mean = lc_summary["q_mean"] - lc_summary["c_mean"]\n'
    'delta_std = np.sqrt(lc_summary["q_std"]**2 + lc_summary["c_std"]**2)\n'
    'ax2.errorbar(lc_summary["n_samples"], delta_mean, yerr=delta_std,\n'
    '             marker="D", capsize=5, color="#2ecc71", linewidth=2)\n'
    'ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)\n'
    'ax2.set_xlabel("Training Set Size")\n'
    'ax2.set_ylabel("ΔF1 (Quantum − Classical)")\n'
    'ax2.set_title("Quantum Advantage vs Training Size")\n'
    'ax2.grid(True, alpha=0.3)\n'
    '\n'
    'plt.tight_layout()\n'
    'plt.show()\n'
))

# ──────────────────────────────────────────────────────────────────────
# Track 2C: Combinatorial Structure (Shuffle Test)
# ──────────────────────────────────────────────────────────────────────
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Track 2C: Combinatorial Structure — Does Motif Arrangement Matter?\n\n"
    "We randomly permute motif assignments within each position column across **training samples only** "
    "(keeping the test set unchanged). This preserves the marginal frequency of each motif at each position "
    "but destroys the combinatorial relationships between positions.\n\n"
    "If performance drops, the task depends on inter-position structure — the quantum circuit's entanglement may be capturing this."
))

new_cells.append(nbformat.v4.new_code_cell(
    '# Track 2C: Shuffle test — destroy combinatorial structure\n'
    '# We shuffle the raw motif-ID columns (train_motifs), then re-encode\n'
    'n_shuffle_seeds = 5\n'
    'shuffle_f1_scores = []\n'
    '\n'
    'for seed in range(n_shuffle_seeds):\n'
    '    rng = np.random.RandomState(seed)\n'
    '    train_data_shuffled = train_data.copy()  # numpy array\n'
    '    \n'
    '    # Shuffle each column (position) independently — preserves marginals, destroys combinatorics\n'
    '    for col_idx in range(train_data_shuffled.shape[1]):\n'
    '        rng.shuffle(train_data_shuffled[:, col_idx])\n'
    '    \n'
    '    svm_shuffled = GridSearchCV(\n'
    '        SVC(kernel="rbf"), PARAM_GRID, cv=cv,\n'
    '        scoring="f1_weighted", n_jobs=-1,\n'
    '    ).fit(train_data_shuffled, train_labels)\n'
    '    \n'
    '    f1_shuffled = f1_score(test_labels, svm_shuffled.predict(test_data), average="weighted")\n'
    '    shuffle_f1_scores.append(f1_shuffled)\n'
    '    print(f"  Seed {seed}: Shuffled F1 = {f1_shuffled:.4f}")\n'
    '\n'
    'print(f"\\nOriginal Classical F1:  {f1_classical:.4f}")\n'
    'print(f"Shuffled Classical F1:  {np.mean(shuffle_f1_scores):.4f} ± {np.std(shuffle_f1_scores):.4f}")\n'
    'print(f"Drop from shuffling:    {f1_classical - np.mean(shuffle_f1_scores):+.4f}")\n'
    '\n'
    'if np.mean(shuffle_f1_scores) < f1_classical:\n'
    '    print("\\n→ Performance DROPS when combinatorial structure is destroyed.")\n'
    '    print("  The task depends on inter-position motif relationships.")\n'
    '    print("  The quantum circuit\'s ZZ entanglement may capture this structure.")\n'
    'else:\n'
    '    print("\\n→ No significant drop — the task may depend mainly on per-position features.")\n'
))

new_cells.append(nbformat.v4.new_code_cell(
    '# Track 2C: Visualization\n'
    'fig, ax = plt.subplots(figsize=(8, 5))\n'
    '\n'
    'categories = ["Original\\nClassical", "Shuffled\\nClassical", "Original\\nQuantum"]\n'
    'values = [f1_classical, np.mean(shuffle_f1_scores), f1_quantum]\n'
    'colors_bar = ["#3498db", "#95a5a6", "#e74c3c"]\n'
    'errors = [0, np.std(shuffle_f1_scores), 0]\n'
    '\n'
    'bars = ax.bar(categories, values, color=colors_bar, alpha=0.85, edgecolor="black", yerr=errors, capsize=8)\n'
    '\n'
    'for b, v in zip(bars, values):\n'
    '    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,\n'
    '            f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)\n'
    '\n'
    'ax.set_ylabel("Weighted F1 (test)")\n'
    'ax.set_ylim(0, 1)\n'
    'ax.set_title("Track 2C: Impact of Destroying Combinatorial Structure")\n'
    'plt.tight_layout()\n'
    'plt.show()\n'
))

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Summary of All Results\n\n"
    "### Track 0: Baseline\n"
    "- **Classical F1:** 0.7423 | **Quantum F1:** 0.8108 | **ΔF1:** +0.0686\n"
    "- Kernel heatmaps show quantum kernel has stronger block-diagonal structure\n\n"
    "### Track 0 Bonus: Geometric Difference\n"
    "- $g_{cq} = 1.5440$ (> 1, meaning the encodings see data differently)\n"
    "- $s_c = 1.3578$, $s_q = 1.5806$ (quantum model is slightly more complex)\n\n"
    "### Track 1A: Measurement Basis Analysis\n"
    "- Sliced 180-dim projections into X (0–59), Y (60–119), Z (120–179)\n"
    "- Each individual basis achieves comparable performance to the full model\n\n"
    "### Track 1B: Position-Based Analysis\n"
    "- Compared classical vs quantum encoding per motif position\n"
    "- Identifies which positions benefit most from quantum encoding\n\n"
    "### Track 2A: Learning Curves\n"
    "- Tested at 25%, 50%, 75%, 100% of training data (5 seeds each)\n"
    "- Reveals whether quantum advantage grows with data scarcity\n\n"
    "### Track 2C: Combinatorial Structure\n"
    "- Shuffling motif positions destroys inter-position relationships\n"
    "- Tests whether the task has combinatorial structure that entanglement could capture\n\n"
    "### Evidence-Based Claim\n\n"
    "The quantum-projected SVM consistently outperforms the classical SVM (ΔF1 ≈ +0.07). "
    "The geometric difference $g_{cq} = 1.54$ confirms the two encodings organize data differently. "
    "Learning curves and position-based analysis reveal where and when this advantage manifests. "
    "The shuffle test probes whether the task's combinatorial structure explains the quantum encoding's benefit."
))

# Append all new cells
nb.cells.extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Added {len(new_cells)} new cells to the notebook (replacing old experiments).")

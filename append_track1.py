import json

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Track 1: Anatomy Experiment\n",
            "\n",
            "Here we slice the 180-dimensional quantum projections into the 3 measurement bases (X, Y, Z), which are 60 dimensions each. We then train an SVM on each basis to see which captures the most useful information."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "bases = {\n",
            "    \"Basis X\": (0, 60),\n",
            "    \"Basis Y\": (60, 120),\n",
            "    \"Basis Z\": (120, 180)\n",
            "}\n",
            "\n",
            "results = {}\n",
            "\n",
            "for basis_name, (start, end) in bases.items():\n",
            "    print(f\"Training SVM on {basis_name}...\")\n",
            "    X_train_basis = projections_train.iloc[:, start:end]\n",
            "    X_test_basis = projections_test.iloc[:, start:end]\n",
            "    \n",
            "    svm_basis = GridSearchCV(\n",
            "        SVC(kernel=\"rbf\"), PARAM_GRID, cv=cv,\n",
            "        scoring=\"f1_weighted\", n_jobs=-1,\n",
            "    ).fit(X_train_basis, train_labels)\n",
            "    \n",
            "    pred_basis = svm_basis.predict(X_test_basis)\n",
            "    f1_basis = f1_score(test_labels, pred_basis, average=\"weighted\")\n",
            "    \n",
            "    results[basis_name] = f1_basis\n",
            "    print(f\"{basis_name} F1 Score: {f1_basis:.4f}\\n\")\n"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "names = list(results.keys()) + [\"Full Quantum (180)\"]\n",
            "scores = list(results.values()) + [f1_quantum]\n",
            "\n",
            "bars = ax.bar(names, scores, color=[\"#f39c12\", \"#27ae60\", \"#8e44ad\", \"#e74c3c\"], alpha=0.85, edgecolor=\"black\")\n",
            "ax.set_ylabel(\"Weighted F1 (test)\")\n",
            "ax.set_ylim(0, 1)\n",
            "ax.set_title(\"Track 1: Performance by Measurement Basis\")\n",
            "\n",
            "for b, score in zip(bars, scores):\n",
            "    ax.text(b.get_x() + b.get_width() / 2, score + 0.01,\n",
            "            f\"{score:.4f}\", ha=\"center\", fontweight=\"bold\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    }
]

with open("/Users/simran/Desktop/Folder/Start_From_Here.ipynb", "r") as f:
    nb = json.load(f)

nb["cells"].extend(new_cells)

with open("/Users/simran/Desktop/Folder/Start_From_Here.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

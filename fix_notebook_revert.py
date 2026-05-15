import nbformat

notebook_path = "/Users/simran/Desktop/Folder/Start_From_Here.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        
        if "bases = {" in source and "projections_train.iloc" in source:
            source = source.replace('"Basis X": 0', '"Basis X": (0, 60)')
            source = source.replace('"Basis Y": 1', '"Basis Y": (60, 120)')
            source = source.replace('"Basis Z": 2', '"Basis Z": (120, 180)')
            source = source.replace('for basis_name, offset in bases.items():', 'for basis_name, (start, end) in bases.items():')
            source = source.replace('projections_train.iloc[:, offset::3]', 'projections_train.iloc[:, start:end]')
            source = source.replace('projections_test.iloc[:, offset::3]', 'projections_test.iloc[:, start:end]')
            cell.source = source

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook slicing reverted to the correct block layout.")

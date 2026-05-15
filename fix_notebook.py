import nbformat

notebook_path = "/Users/simran/Desktop/Folder/Start_From_Here.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        
        if "API_KEY =" in source:
            new_source = []
            for line in source.split('\n'):
                if line.startswith("API_KEY ="):
                    new_source.append('API_KEY = "sOmEOV357bqTnuMc4EUsC6P5ruq5ua4NDNbHlviGhFAG"')
                else:
                    new_source.append(line)
            cell.source = '\n'.join(new_source)
            
        if "bases = {" in source and "projections_train.iloc" in source:
            source = source.replace('"Basis X": (0, 60)', '"Basis X": 0')
            source = source.replace('"Basis Y": (60, 120)', '"Basis Y": 1')
            source = source.replace('"Basis Z": (120, 180)', '"Basis Z": 2')
            source = source.replace('for basis_name, (start, end) in bases.items():', 'for basis_name, offset in bases.items():')
            source = source.replace('projections_train.iloc[:, start:end]', 'projections_train.iloc[:, offset::3]')
            source = source.replace('projections_test.iloc[:, start:end]', 'projections_test.iloc[:, offset::3]')
            cell.source = source

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook modified successfully.")

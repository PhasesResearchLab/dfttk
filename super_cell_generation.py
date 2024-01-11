from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation

# Load your primitive cell from a POSCAR file or create it manually
prim_cell = Structure.from_file("/storage/home/lam7027/work/FeSe/POSCAR.oqmd")

# Define the scaling factors for the supercell
scaling_matrix = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]

# Create the supercell transformation
supercell_transformation = SupercellTransformation(scaling_matrix=scaling_matrix)


# Apply the transformation to generate the supercell
supercell = supercell_transformation.apply_transformation(prim_cell)

# Print the supercell lattice parameters and atomic positions
print("Supercell Lattice:")
print(supercell.lattice)
print("\nSupercell Atomic Positions:")
print(supercell)

# Save the supercell to a new POSCAR file
supercell.to("/storage/home/lam7027/work/FeSe/POSCAR.oqmd_supercell")

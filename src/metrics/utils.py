from io import StringIO

import ase

def atoms_to_xyz_str(atoms: ase.Atoms):
  f = StringIO()
  atoms.write(f, format="xyz")
  return f.getvalue()
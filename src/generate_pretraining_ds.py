import ase
from ase.io import read
from ase.db import connect
import os, glob

path_tmqm = r"/home/scratch3/s222491/tmqm"
db_path = r"/home/scratch3/s222491/tmqm/tmqm.db"

all_xyz = glob.glob(os.path.join(path_tmqm, "*.xyz"))

with connect(db_path, append=False) as db:
  for xyz in all_xyz:
    atoms = read(xyz)
    if len(atoms) < 70:
      formula = atoms.get_chemical_formula()
      with open(xyz, 'r') as file:
        lines = file.readlines()
      mol_info = lines[1]
      if int(mol_info[-2]) == 4 and all(element not in formula for element in ['B', 'Si', 'Se']):
        del atoms[[atom.index for atom in atoms if atom.symbol == 'H']]
        db.write(atoms)
  
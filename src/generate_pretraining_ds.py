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
    del atoms[[atom.index for atom in atoms if atom.symbol == 'H']]
    db.write(atoms)
  
from os.path import join, splitext, basename
from xyz2mol import xyz2mol as xm
from ase import Atoms
from ase.io import write
from ase.db import connect
from ase.calculators.gaussian import Gaussian, GaussianOptimizer
from rdkit.Chem import MolToSmiles

h2_energy = 0


class VaskasPipeline:
  scratch = r'/home/scratch3/s222491/'
  db_path = r'/home/scratch3/s222491/vaskas/vaskas.db'
  
  def __init__(self, sample):
    self.barrier = None
    self.ts_atoms = None
    self.cat_xyz = None
    self.cat_atoms = None
    self.sample = sample
    self.filename = splitext(basename(self.sample))[0]
  
  def remove_h2(self) -> None:
    atoms, charge, xyz = xm.read_xyz_file(self.sample)
    
    ase_atom_H2 = Atoms(numbers=atoms, positions=xyz, pbc=False)
    ase_atom = ase_atom_H2.copy()
    
    AC, mol = xm.xyz2AC(atoms, xyz, charge)
    
    metal_connectivity = AC[atoms.index(77)]
    H_pos = [xyz[i] for i in range(len(metal_connectivity)) if metal_connectivity[i] == 1 and atoms[i] == 1]
    del ase_atom[[atom.index for atom in ase_atom if atom.position.tolist() in H_pos]]
    
    xyz_path = join(VaskasPipeline.scratch, 'generated_vaskas/generated_cat', f'{self.filename}.xyz')
    write(xyz_path, ase_atom)
    
    self.cat_xyz = xyz_path
    self.cat_atoms = ase_atom
    self.ts_atoms = ase_atom_H2
  
  def dft(self) -> None:
    calc = Gaussian(xc='PBE',
                    basis='def2SVP',
                    scf='qc',
                    command='g16 < PREFIX.com > PREFIX.log',
                    mult=1,
                    charge=0
                    )
    calc.set_label(join(VaskasPipeline.scratch, f'{self.filename}_cat.xyz'))
    self.cat_atoms.calc = calc
    
    calc.set_label(join(VaskasPipeline.scratch, f'{self.filename}_ts.xyz'))
    self.ts_atoms.calc = calc
    
    opt = GaussianOptimizer(self.cat_atoms, calc)
    opt.run(fmax='tight', steps=100)
    
    #Implement code for TS calculation here
    
    cat_energy = self.cat_atoms.get_potential_energy()
    ts_energy = self.ts_atoms.get_potential_energy()
    self.barrier = ts_energy - cat_energy - h2_energy
    
    write(join(VaskasPipeline.scratch, 'DFT_cat', f'{self.filename}.xyz'), self.cat_atoms)
    write(join(VaskasPipeline.scratch, 'DFT_ts', f'{self.filename}.xyz'), self.ts_atoms)
    
  @staticmethod
  def xyz2smi(xyz):
    atom, charge, coords = xm.read_xyz_file(xyz)
    mol = xm.xyz2mol(atoms=atom, charge=charge, coordinates=coords)
    smiles = MolToSmiles(mol[0])
    
    return smiles
  
  def add_to_training(self) -> None:
    smi = self.xyz2smi(self.cat_xyz)
    
    with connect(VaskasPipeline.db_path, append=True) as db:
      db.write(self.cat_atoms, key_value_pairs = {'barrier': self.barrier, 'smiles': smi})

from os.path import join, splitext, basename
from xyz2mol import xyz2mol as xm
from ase import Atoms
from openbabel import openbabel
from ase.io import write, read
from ase.db import connect
from ase.calculators.gaussian import Gaussian, GaussianOptimizer
from rdkit.Chem import MolToSmiles, MolFromSmiles, DetectChemistryProblems
from src.metrics.utils import atoms_to_xyz_str, remove_h2

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
  
  def validity_check(self):
    try:
      atom = read(self.sample)
      clean_atom = remove_h2(atom)
      natoms = len(clean_atom)
      for j, num in enumerate(reversed(clean_atom.numbers)):
        if num == 77:
          clean_atom.pop(natoms - j - 1)
      xyz = atoms_to_xyz_str(clean_atom)
      obConversion = openbabel.OBConversion()
      obConversion.SetInFormat('xyz')
      
      mol = openbabel.OBMol()
      obConversion.ReadString(mol, xyz)
      mol.MakeDativeBonds()
      
      obConversion.SetOutFormat("smi")
      smiles = obConversion.WriteString(mol)
      
      mol_rdk = MolFromSmiles(smiles)
      probs = DetectChemistryProblems(mol_rdk)
      
      if len(probs) > 0:
        return self.sample
    
    except Exception as e:
      print(f"Exception {e} in samples {self.sample}")
      return None
    
  
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

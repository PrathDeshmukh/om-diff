import glob, os
import argparse
import ase
import submitit
import torch
from ase.io import write, read, iread
import sys
from ase.calculators.gaussian import Gaussian, GaussianOptimizer
from rdkit.Chem import DetectChemistryProblems, rdDetermineBonds
from rdkit import Chem
from src.metrics.utils import remove_h2, atoms_to_xyz_str, write_basisfile

basis = r'/home/energy/s222491/TS_basis.gbs'

def parse_cmd():
  parser = argparse.ArgumentParser(description="Checking validty and novelty of samples and running DFT calculations")
  
  parser.add_argument(
    "--samples_path",
    type=str,
    required=True,
  )
  
  parser.add_argument(
    "--barrier_criteria",
    type=float,
    required=True,
  )
  return parser.parse_args()


cmd_args = parse_cmd()
samples_path = os.path.join(cmd_args.samples_path, 'samples.xyz')
barrier_criteria = cmd_args.barrier_criteria
predictions = os.path.join(cmd_args.samples_path, 'predictions.pt')


class RunDFT:
  scratch = r'/home/scratch3/s222491/'
  db_path = r'/home/scratch3/s222491/vaskas/vaskas.db'
  
  def __init__(self, ts: ase.Atoms, label: str):
    if not isinstance(ts, ase.Atoms):
      raise ValueError("Invalid input type.")
    
    self.ts = ts
    self.H_index = None
    self.label = label
    write_basisfile(self.ts, basisfile=basis, label=self.label)
  
  def catOpt(self):
    atoms, self.H_index = remove_h2(self.ts)
    
    calc_opt = Gaussian(mem='8GB',
                        label=os.path.join(samples_path, self.label),
                        xc='PBE',
                        chk=os.path.join(samples_path, self.label, f'cat_{self.label}.chk'),
                        basisfile=basis,
                        mult=1,
                        charge=0
                        )
    opt = GaussianOptimizer(atoms, calc_opt)
    opt.run(fmax='tight')
    
    write(os.path.join(samples_path, self.label, f'cat_{self.label}.xyz'), atoms)
    energy = atoms.get_potential_energy()
    
    return energy
  
  def tsOpt(self, atoms):
    calc_opt = Gaussian(mem='8GB',
                        label=os.path.join(samples_path, self.label),
                        xc='PBE',
                        chk=os.path.join(samples_path, self.label, f'ts_{self.label}.chk'),
                        basisfile=basis,
                        mult=1,
                        charge=0,
                        extra="nosymm EmpiricalDispersion=GD3 pseudo=read"
                        )
    opt = GaussianOptimizer(atoms, calc_opt)
    opt.run(opt='calcfc,ts')
    
    energy = atoms.get_potential_energy()
    write(os.path.join(samples_path, self.label, f'ts_{self.label}.xyz'), atoms)
    return energy
  
  def tsFreeze(self):
    atoms = self.ts
    addsec = f"B {self.H_index[0] + 1} {self.H_index[1] + 1} F"
    
    calc_opt = Gaussian(mem='8GB',
                        label=os.path.join(samples_path, self.label),
                        xc='PBE',
                        chk=os.path.join(samples_path, self.label, f'ts_freeze_{self.label}.chk'),
                        basisfile=basis,
                        addsec=addsec,
                        mult=1,
                        charge=0,
                        extra="nosymm EmpiricalDispersion=GD3 pseudo=read"
                        )
    opt = GaussianOptimizer(atoms, calc_opt)
    opt.run(opt='addredun')
    
    energy = self.tsOpt(atoms=atoms)
    return energy


def compute_dft(sample, label):
  dft = RunDFT(sample, label)
  try:
    cat_energy = dft.catOpt()
  except Exception as e:
    print(f"Exception {e} in complex no. {label}")
    return None
  else:
    try:
      ts_energy = dft.tsOpt(sample)
      barrier = ts_energy - cat_energy - H_energy
      return barrier
    except:
      try:
        ts_energy = dft.tsFreeze()
        barrier = ts_energy - cat_energy - H_energy
        return label, barrier
      
      except:
        print(f"No transition state for sample {label}")
        return None


class ValidityCheck:
  def __init__(self, ts: ase.Atoms):
    if not isinstance(ts, ase.Atoms):
      print("Invalid input type.")
    self.charges = [-2, -1, 0, 1, 2]
    self.ts = ts
  
  def checkvalidity(self):
    try:
      atom = self.ts
      clean_atom, H_ind = remove_h2(atom)
      natoms = len(clean_atom)
      for j, num in enumerate(reversed(clean_atom.numbers)):
        if num == 77:
          clean_atom.pop(natoms - j - 1)
      
      As_index = [id for id, atom in enumerate(clean_atom) if atom.symbol == 'As']
      for id in As_index:
        clean_atom[id].symbol = 'P'
      
      ind_samples = []
      for chg in self.charges:
        try:
          xyz = atoms_to_xyz_str(clean_atom)
          raw_mol = Chem.MolFromXYZBlock(xyz)
          mol = Chem.Mol(raw_mol)
          rdDetermineBonds.DetermineBonds(
            mol,
            charge=chg,
          )
          mol.UpdatePropertyCache(strict=False)
          probs = DetectChemistryProblems(mol)
          smi = Chem.MolToSmiles(mol)
          frags = smi.split(".")
          if len(probs) == 0 and len(frags) <= 4:
            ind_samples.append(smi)
        except Exception as e:
          print(e)
          continue
      if len(ind_samples) > 0:
        return self.ts
    
    except Exception as e:
      print(f"Exception {e} in samples {self.ts}")  # Replace self.ts with a label/identifier
      return None


ds_path = r"/home/scratch3/s222491/vaskas/coordinates_complex"
all_ds_xyz = glob.glob(os.path.join(ds_path, "*.xyz"))
all_ds_atoms = [read(xyz) for xyz in all_ds_xyz]


def atoms2fp(clean_atom, charges):
  natoms = len(clean_atom)
  for j, num in enumerate(reversed(clean_atom.numbers)):
    if num == 77:
      clean_atom.pop(natoms - j - 1)
  
  # Replace As with P because rdkit does not work with As
  As_index = [id for id, atom in enumerate(clean_atom) if atom.symbol == 'As']
  for l in As_index:
    clean_atom[l].symbol = 'P'
  fps_arr = []
  
  for chg in charges:
    try:
      xyz = atoms_to_xyz_str(clean_atom)
      
      raw_mol = Chem.MolFromXYZBlock(xyz)
      mol = Chem.Mol(raw_mol)
      rdDetermineBonds.DetermineBonds(
        mol,
        charge=chg,
      )
      fp = Chem.RDKFingerprint(mol)
      if fp:
        fps_arr.append(fp)
    except:
      pass
  
  return fps_arr


all_ds_fps = [atoms2fp(ds_atoms, charges=[-1]) for ds_atoms in all_ds_atoms]

charges = [-2,-1,0,1,2]
def novelty(sample):
  sim_arr = []
  try:
    clean_atoms, H_ind = remove_h2(sample)
    gen_fps = atoms2fp(clean_atoms, charges=charges)
    
    for fps in gen_fps:
      for j, ds_fps in enumerate(all_ds_fps):
        try:
          for ds_fp in ds_fps:
            sim = Chem.DataStructs.TanimotoSimilarity(ds_fp, fps)
            if sim >= 0.90:
              sim_arr.append([sim, j])
        except:
          pass
  except:
    pass
  
  if len(sim_arr) > 0:
    print(sim_arr)
    return None
  else:
    return sample


all_samples = iread(samples_path)
all_samples_list = list(all_samples)
H_energy = -1.1389  # Write H energy

novel_samples = []
valid_samples = []
labels = []

for i, sample in enumerate(all_samples_list):
  novel_sample = novelty(sample)
  if novel_sample:
    novel_samples.append(sample)
    val_chk = ValidityCheck(novel_sample)
    valid_sample = val_chk.checkvalidity()
    if valid_sample:
      valid_samples.append(valid_sample)
      labels.append(i)

print(f"No. of novel samples: {len(novel_samples)}")
print(f"No. of valid samples: {len(valid_samples)}")
print(f"Total number of samples: {i}")

preds = torch.load(predictions)

samples_for_dft = [all_samples_list[k] for k in labels if preds['barrier'][k] <= barrier_criteria + 1]

for dft_sample in samples_for_dft:
  write('samples_for_dft2.xyz', dft_sample, append=True)
print(labels)
sys.exit()
executor = submitit.AutoExecutor(folder="sublogs/")

if executor._executor.__class__.__name__ == 'SlurmExecutor':
  executor.update_parameters(
    slurm_partition="xeon24",
    cpus_per_task=24,
    tasks_per_node=1,
    slurm_nodes=1,
    slurm_time="2-00:00:00",
  )

jobs = executor.map_array(compute_dft, samples_for_dft,
                          labels)  # submit_fn = function that runs DFT calculation, list_of_params=list of atoms object

print(jobs.result())
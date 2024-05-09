import glob, os
import argparse
import ase
import submitit
import torch
import csv
from ase.io import write, read, iread
import sys
from ase.calculators.gaussian import Gaussian, GaussianOptimizer
from rdkit.Chem import DetectChemistryProblems, rdDetermineBonds
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
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

header_row = ['label', 'barrier']
with open(os.path.join(samples_path, 'barrier_energies.csv'), 'w') as f:
  writer = csv.writer(f)
  writer.writerow(header_row)
  f.close()
  
class RunDFT:
  scratch = r'/home/scratch3/s222491/'
  db_path = r'/home/scratch3/s222491/vaskas/vaskas.db'
  
  def __init__(self, ts: ase.Atoms, label: str, charge: int):
    if not isinstance(ts, ase.Atoms):
      raise ValueError("Invalid input type.")
    
    self.ts = ts
    self.charge = charge
    self.H_index = None
    self.label = label
    write_basisfile(self.ts, basisfile=basis, label=self.label)
  
  def catOpt(self):
    atoms, self.H_index = remove_h2(self.ts)
    
    calc_opt = Gaussian(mem='8GB',
                        label=os.path.join(samples_path, 'gaussian/', self.label),
                        xc='PBE',
                        chk=os.path.join(samples_path, 'gaussian/', f'cat_{self.label}.chk'),
                        basisfile=basis,
                        mult=1,
                        charge=self.charge
                        )
    opt = GaussianOptimizer(atoms, calc_opt)
    opt.run(fmax='tight')
    
    write(os.path.join(samples_path, 'gaussian/', f'cat_{self.label}.xyz'), atoms)
    energy = atoms.get_potential_energy()
    
    return energy
  
  def tsOpt(self, atoms):
    calc_opt = Gaussian(mem='8GB',
                        label=os.path.join(samples_path,'gaussian/', self.label),
                        xc='PBE',
                        chk=os.path.join(samples_path,'gaussian/', f'ts_{self.label}.chk'),
                        basisfile=basis,
                        mult=1,
                        charge=self.charge,
                        extra="nosymm EmpiricalDispersion=GD3 pseudo=read"
                        )
    opt = GaussianOptimizer(atoms, calc_opt)
    opt.run(opt='calcfc,ts')
    
    energy = atoms.get_potential_energy()
    write(os.path.join(samples_path, 'TS_optimized', f'ts_{self.label}.xyz'), atoms)
    return energy
  
  def tsFreeze(self):
    atoms = self.ts
    addsec = f"B {self.H_index[0] + 1} {self.H_index[1] + 1} F"
    
    calc_opt = Gaussian(mem='8GB',
                        label=os.path.join(samples_path,'gaussian/', self.label),
                        xc='PBE',
                        chk=os.path.join(samples_path,'gaussian/', f'ts_freeze_{self.label}.chk'),
                        basisfile=basis,
                        addsec=addsec,
                        mult=1,
                        charge=self.charge,
                        extra="nosymm EmpiricalDispersion=GD3 pseudo=read"
                        )
    opt = GaussianOptimizer(atoms, calc_opt)
    opt.run(opt='addredun')
    
    energy = self.tsOpt(atoms=atoms)
    return energy


def compute_dft(sample, label, charge):
  dft = RunDFT(sample, label, charge)
  try:
    cat_energy = dft.catOpt()
  except Exception as e:
    print(f"Exception {e} in complex no. {label}")
    return None
  else:
    try:
      ts_energy = dft.tsOpt(sample)
      barrier = ts_energy - cat_energy - H_energy
      with open(os.path.join(samples_path, 'barrier_energies.csv'), 'a') as file:
        writer = csv.writer(file)
        writer.writerow([label, barrier])
      file.close()
      return barrier
    except:
      try:
        ts_energy = dft.tsFreeze()
        barrier = ts_energy - cat_energy - H_energy
        with open(os.path.join(samples_path, 'barrier_energies.csv'), 'a') as file:
          writer = csv.writer(file)
          writer.writerow([label,barrier])
        file.close()
        return label, barrier
      
      except:
        print(f"No transition state for sample {label}")
        return None


class ValidityCheck:
  def __init__(self, ts: ase.Atoms):
    if not isinstance(ts, ase.Atoms):
      print("Invalid input type.")
    self.charges = [-1, 0, -2]
    self.ts = ts
  
  def checkvalidity(self):
    try:
      atom = sample
      clean_atom, H_ind = remove_h2(atom)
      natoms = len(clean_atom)
      for j, num in enumerate(reversed(clean_atom.numbers)):
        if num == 77:
          clean_atom.pop(natoms - j - 1)
      
      As_index = [id for id, atom in enumerate(clean_atom) if atom.symbol == 'As']
      for id in As_index:
        clean_atom[id].symbol = 'P'
      return_chg = []
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
            return_chg.append(chg)
            break
          else:
            continue
        except Exception:
          pass
      if len(return_chg) > 0:
        return self.ts, return_chg[0]
      else:
        return None
    except Exception as e:
      print(f"Exception {e} in sample")  # Replace self.ts with a label/identifier


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

charges = [-2,-1,0]
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
valid_charges = []
labels = []

for i, sample in enumerate(all_samples_list):
  novel_sample = novelty(sample)
  if novel_sample:
    novel_samples.append(sample)
    val_chk = ValidityCheck(novel_sample)
    sample_validity = val_chk.checkvalidity()
    if sample_validity is not None:
      valid_sample, valid_chg = sample_validity
      valid_samples.append(valid_sample)
      valid_charges.append(valid_chg)
      labels.append(i)

assert len(valid_samples) == len(valid_charges), "No. of samples and charges is not equal"

novel_count = len(novel_samples)
valid_count = len(valid_samples)
total_count = len(all_samples_list)

print(f"No. of novel samples: {novel_count}")
print(f"Percentage of novel samples: {float(novel_count/total_count)}%")
print(f"No. of valid samples: {valid_count}")
print(f"Percentage of valid samples: {float(valid_count/total_count)}%")
print(f"Total number of samples: {total_count}")

with open(os.path.join(samples_path,'stats.txt'), 'w') as f:
  f.write(f"No. of novel samples: {novel_count}" + '\n')
  f.write(f"Percentage of novel samples: {float(novel_count/total_count)}%" + '\n')
  f.write(f"No. of valid samples: {valid_count}" + '\n')
  f.write(f"Percentage of valid samples: {float(valid_count/total_count)}%" + '\n')
  f.write(f"Total number of samples: {total_count}" + '\n')
  f.close()

preds = torch.load(predictions)

barrier = preds['barrier']
barrier_np = barrier.detach().cpu().numpy()

sns.histplot(barrier_np,
             kde=True)
plt.xlabel("Barrier energy")
plt.savefig(os.path.join(samples_path, 'pre-screening_energy_dist.png'))
plt.clf()

samples_for_dft = [all_samples_list[k] for k in labels if preds['barrier'][k] <= barrier_criteria + 1]

for dft_sample in samples_for_dft:
  write('samples_for_dft2.xyz', dft_sample, append=True)
print(labels)

post_screening_energies = [preds['barrier'][k] for k in labels]
sns.histplot(barrier_np,
             kde=True)
plt.xlabel("Barrier energy")
plt.savefig(os.path.join(samples_path, 'post-screening_energy_dist.png'))

executor = submitit.AutoExecutor(folder=os.path.join(samples_path, "sublogs/"))

if executor._executor.__class__.__name__ == 'SlurmExecutor':
  executor.update_parameters(
    slurm_partition="xeon24",
    cpus_per_task=24,
    tasks_per_node=1,
    slurm_nodes=1,
    slurm_time="18:00:00",
  )

jobs = executor.map_array(compute_dft, samples_for_dft,
                          labels, valid_charges)  # submit_fn = function that runs DFT calculation, list_of_params=list of atoms object
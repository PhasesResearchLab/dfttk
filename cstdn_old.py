import os
import shutil

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler
from custodian.vasp.jobs import VaspJob
from pymatgen.core import structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

subset = list(VaspErrorHandler.error_msgs.keys())
subset.remove("algo_tet")

handlers = [VaspErrorHandler(errors_subset_to_catch = subset)]
vasp_cmd = ["srun", "vasp_std"]

# Full volume relaxation
os.makedirs('vol_0')
current_directory = os.path.abspath('')
files_to_copy = ['INCAR', 'KPOINTS', 'POSCAR', 'POTCAR']

for file_name in files_to_copy:
    if os.path.isfile(file_name):
        shutil.copy2(os.path.join(current_directory, file_name), os.path.join('vol_0', file_name))

os.chdir('vol_0')

# Step 1
step1 = VaspJob(
vasp_cmd = vasp_cmd,
copy_magmom = True,
final = False,
suffix = '.1relax'
        )

# Step 2
step2 = VaspJob(
vasp_cmd = vasp_cmd,
copy_magmom = True,
final = False,
suffix = '.2relax',
settings_override = [
    {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
    ]
        )

# Step 3
step3 = VaspJob(
vasp_cmd = vasp_cmd,
copy_magmom = True,
final = True,
suffix = '.3static',
settings_override = [
    {"dict": "INCAR", "action": {"_set": {
        "IBRION": -1,
	"NSW": 0,
        "ISMEAR": -5
        }}},
    {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
    ]
        )

jobs = [step1, step2, step3]

c = Custodian(handlers, jobs, max_errors=3)
c.run()


# Relax at other fixed volumes
new_volumes = []
volume_scale = [6, 4, 2, -2, -4, -6] #percent change in volume
struct = structure.Structure.from_file("CONTCAR.3static")
volume = struct.volume

for i in volume_scale:
    new_volumes.append(volume * (1 + i/100))

j = 0
for num in volume_scale:
        folder_name = 'vol_' + str(num)
        folder_path = '../' + folder_name
        os.makedirs(folder_path)
        files_to_copy = ['CONTCAR.3static', 'INCAR.2relax', 'KPOINTS.1relax', 'POTCAR', 'WAVECAR.3static', 'CHGCAR.3static']
        current_directory = os.path.abspath('')
        for file_name in files_to_copy:
            if os.path.isfile(file_name):
                shutil.copy2(os.path.join(current_directory, file_name), os.path.join(folder_path, file_name))

        os.chdir(folder_path)
        os.rename('CONTCAR.3static', 'POSCAR')
        os.rename('INCAR.2relax', 'INCAR')
        os.rename('KPOINTS.1relax', 'KPOINTS')
        os.rename('WAVECAR.3static', 'WAVECAR')
        os.rename('CHGCAR.3static', 'CHGCAR')

        struct = structure.Structure.from_file("POSCAR")
        new_volume = new_volumes[j]
        struct.scale_lattice(new_volume)
        struct.to_file("POSCAR", "POSCAR")

        # Step 1
        step1 = VaspJob(
        vasp_cmd = vasp_cmd,
        copy_magmom = True,
        final = False,
        suffix = '.1relax',
        settings_override = [
            {"dict": "INCAR", "action": {"_set": {
                "ISIF": 4,
                }}},
            ]
            	)

        # Step 2
        step2 = VaspJob(
        vasp_cmd = vasp_cmd,
        copy_magmom = True,
        final = False,
        suffix = '.2relax',
        settings_override = [
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
            ]
                )

        # Step 3
        step3 = VaspJob(
        vasp_cmd = vasp_cmd,
        copy_magmom = True,
        final = True,
        suffix = '.3static',
        settings_override = [
            {"dict": "INCAR", "action": {"_set": {
                "IBRION": -1,
        	"NSW": 0,
                "ISMEAR": -5
                }}},
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
            ]
                )

        jobs = [step1, step2, step3]

        c = Custodian(handlers, jobs, max_errors=3)
        c.run()
        j += 1

os.chdir('..')

# Function to extract the last occurrence of volume from OUTCAR files
def extract_volume(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'volume' in line:
                volume = float(line.split()[-1])
                break  # Stop searching after finding the last occurrence
    return volume

# Function to extract the last occurrence of pressure from OUTCAR files
def extract_pressure(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'pressure' in line:
                pressure = float(line.split()[3])
                break  # Stop searching after finding the last occurrence
    return pressure

# Function to extract energy from OSZICAR files
def extract_energy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'F=' in line:
                energy = float(line.split()[4])
                break  # Stop searching after finding the last occurrence
    return energy

# Directory containing OUTCAR and OSZICAR files
parent_directory = ''

# Lists to store extracted data
volume_list = []
energy_list = []
pressure_list = []
magmom_list = []

# Iterate through each folder
for num in range(-6, 7, 2):
    folder_name = 'vol_' + str(num)
    folder_path = os.path.join(parent_directory, folder_name)

    # Iterate through files in each folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Extract volume and magmom from OUTCAR files
        if 'OUTCAR.3static' in filename:
            volume = extract_volume(file_path)
            volume_list.append(volume)
            outcar = Outcar(file_path)
            magmom = [m["tot"] for m in outcar.magnetization]
            magmom_list.append(magmom)

        # Extract pressure from OUTCAR files
        if 'OUTCAR.3static' in filename:
            pressure = extract_pressure(file_path)
            pressure_list.append(pressure)

        # Extract energy from OSZICAR files
        elif 'OSZICAR.3static' in filename:
            energy = extract_energy(file_path)
            energy_list.append(energy)

# Write extracted data into the output file with volume in the first column and energy in the second column
with open('energy-volume', 'w') as output_file:
    for volume, energy in zip(volume_list, energy_list):
        output_file.write(f"{volume} {energy}\n")

# Write extracted data into the output file with pressure in the first column
with open('pressure', 'w') as output_file:
    for pressure in pressure_list:
        output_file.write(f"{pressure}\n")

# Write magmom_list to file
with open('magmom', 'w') as output_file:
    for magmom in magmom_list:
        output_file.write(f"{magmom}\n")

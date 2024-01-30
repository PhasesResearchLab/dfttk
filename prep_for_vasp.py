import os
import sys
import pandas as pd
from ase import io
import shutil
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints

def write_to_file(filename, lines):
    with open(filename, 'w') as file:
        file.writelines(lines)

def parse(ywoutput, directory='strs'):
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the 'strs' directory if it doesn't exist
    
    with open(ywoutput, 'r') as file:
        lines = file.readlines()

    current_lines = []
    file_number = 1

    for line in lines:
        current_lines.append(line)

        # Check if the line contains the "end" marker
        if 'end' in line:
            output_filename = os.path.join(directory, f'str.{file_number}')
            write_to_file(output_filename, current_lines)
            current_lines = []  # Reset the current_lines list for the next section
            file_number += 1

def convert_strs_to_poscars(directory='strs', configurations_directory='configurations'):
    # Get a list of files in the specified directory
    str_files = [filename for filename in os.listdir(directory) if filename.startswith('str')]

    # Create the configurations directory if it doesn't exist
    if not os.path.exists(configurations_directory):
        os.makedirs(configurations_directory)

    for structure_from in str_files:
        output_directory = os.path.join(configurations_directory, 'config_' + structure_from[4:])
        structure_to = os.path.join(output_directory, 'POSCAR')

        if not os.path.exists(f'{directory}/{structure_from}'):
            print(f'EEEE ATAT file {structure_from} does not exist in the path!')
            print('EEEE You have to specify the ATAT file in this Python script.')
            print('EEEE exiting ...')
            sys.exit(1)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Convert str.out to CIF format first
        tmp = 'tmp.cif'
        cmd = f'str2cif < {directory}/{structure_from} > {tmp}'
        os.system(cmd)

        # Then convert CIF to POSCAR using ASE
        atoms = io.read(tmp)
        atoms.write(structure_to, format='vasp')

        # Clean up
        os.remove(tmp)

def count_atoms(strs_dir='strs'):
    kept = []
    data_list = []
    for file_name in os.listdir(strs_dir):
        frequency_dict = {'file_number': file_name[4:]}
        with open(os.path.join(strs_dir, file_name), 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    parts = line.split()
                    if len(parts) == 4:
                        value = parts[3]
                        if value in frequency_dict:
                            frequency_dict[value] += 1
                        else:
                            frequency_dict[value] = 1
        data_list.append(frequency_dict)
    df = pd.DataFrame(data_list) 
    df = df.fillna(0) 
    return df

def remove_spin_up_less_than_down(atom_count_df, atom_up, atom_down): #takes the dataframe from count_atoms and removes all files where the number of spin up atoms is less than the number of spin down
    for index, row in atom_count_df.iterrows():
        parity = [] #when the number of spin up atoms is equal to the number of spin down atoms we need to keep one of the files but not the other. This list keeps track of whether we have kept a parity file or not.
        if row[atom_up] < row[atom_down]:
            os.remove(f'strs/str.{row["file_number"]}')


def make_incars(atom_1, atom_2, atom_3, spin_1, spin_2, spin_3, incar='INCAR', configurations_directory='configurations', strs_dir='strs'): #takes an INCAR file and adds a line for the magnetic moment in accordance with the str file
    str_files = [filename for filename in os.listdir(strs_dir) if filename.startswith('str')]
    for str_file in str_files:
        mag_mom_list = []
        output_directory = os.path.join(configurations_directory, 'config_' + str_file[4:])
        incar_to = os.path.join(output_directory, 'INCAR')
        with open(os.path.join(strs_dir, str_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                if atom_1 in line:
                    mag_mom_list.append(spin_1)
                elif atom_2 in line:
                    mag_mom_list.append(spin_2)
                elif atom_3 in line:
                    mag_mom_list.append(spin_3)
        mag_mom_string = ' '.join(mag_mom_list)
        shutil.copy(incar, incar_to)
        with open(incar_to, 'a') as file:
            file.write('\nMAGMOM = %s' % mag_mom_string)


"""
This function differs from make_incars and convert_strs_to_poscars in that it
requires the config_dirs to already exist. it also assume that there is a
poscar inside that directory.

kppa (float) – Grid density

force_gamma (bool) – Force a gamma centered mesh (default is to use gamma only for hexagonal cells or odd meshes)
configurations_directory (str) – Path to the configurations directory
"""
def make_kpoints(kppa, force_gamma=False, configurations_directory='configurations'):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    for config_dir in config_dirs:
        structure = Structure.from_file(os.path.join(configurations_directory, config_dir, 'POSCAR'))
        kpoints = Kpoints.automatic_density(structure, kppa, force_gamma=force_gamma)
        kpoints.write_file(os.path.join(configurations_directory, config_dir, 'KPOINTS'))

# def prep_for_vasp(ywoutput, magnetic_configurations=False):
#     parse(ywoutput)

#     if magnetic_configurations == False:
#         for file in os.listdir():
#             convert_strs_to_poscars()

#     if magnetic_configurations == True:
#         remove_spin_up_less_than_down()
#         convert_strs_to_poscars()
#         make_incars(str_file)


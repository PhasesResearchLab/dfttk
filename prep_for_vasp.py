import os
import sys
import pandas as pd
from ase import io
import shutil
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Poscar

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


"""
magmoms is a dictionary that looks like:
magmoms = {'Fe+': 5, 'Fe-': -5, 'O': 0}
"""

def make_incars(magmoms, incar='INCAR', configurations_directory='configurations', strs_dir='strs'): #takes an INCAR file and adds a line for the magnetic moment in accordance with the str file
    str_files = [filename for filename in os.listdir(strs_dir) if filename.startswith('str')]
    for str_file in str_files:
        mag_mom_list = []
        output_directory = os.path.join(configurations_directory, 'config_' + str_file[4:])
        incar_to = os.path.join(output_directory, 'INCAR')
        atoms = list(magmoms.keys())
        with open(os.path.join(strs_dir, str_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                count = 0
                for atom in atoms:
                    if atom in line:
                        mag_mom_list.append(magmoms[atom])
                        count += 1
                if count not in [0, 1]:
                    raise ValueError(f'Error: {count} atoms found in line {line} in file {str_file}')
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


"""
function that creates a submit script for each configuration directory. It
also changes the job name to be the same as the configuration name. that is,
everything after config_ in the directory name.
"""
def create_submit_scripts(configurations_directory='configurations', submit_script='submit.sh'):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    with open(submit_script, 'r') as submit_file:
        lines = submit_file.readlines()
    for config_dir in config_dirs:
        new_job_name = config_dir.split('config_')[-1]
        submit_script_path = os.path.join(configurations_directory, config_dir, submit_script)
        with open(submit_script_path, 'w') as file:
            for line in lines:
                if line.startswith('#SBATCH --job-name='):
                    file.write(f'#SBATCH --job-name={new_job_name}\n')
                else:
                    file.write(line)


def read_magmom_line(incar_file):
    with open(incar_file, 'r') as file:
        for line in file:
            if line.startswith('MAGMOM ='):
                magmom_line = line.strip()
                numeric_part = magmom_line.split('=')[1].strip()
                numeric_list = numeric_part.split()
                numeric_values = [int(value) for value in numeric_list]
                return numeric_values
    return None

"""
this function is a patch to rearrange the sites and magmoms in the POSCAR and
INCAR files. If the sites are not grouped by specie, VASP will look for more
potentials than supplied/necessary.
"""
def rearrage_sites_and_magmoms(config_dir):
    incar_file = os.path.join(config_dir, 'INCAR')
    poscar_file = os.path.join(config_dir, 'POSCAR')
    struct = Structure.from_file(poscar_file) # read poscar
    orig_magmoms = read_magmom_line(incar_file) # read magmom from incar
    struct.add_site_property("magmom", orig_magmoms) # add magmom to structure
    struct = struct.get_sorted_structure() # sort structure with the magmoms
    rearranged_magmoms = struct.site_properties['magmom'] # get the rearranged magmoms
    numeric_strings = [str(value) for value in rearranged_magmoms] # convert values to strings
    result_string = ' '.join(numeric_strings) # join the strings
    result_string = "MAGMOM = " + result_string # add the MAGMOM = part

    # Write the result_string to the INCAR file
    with open(incar_file, 'r') as file:
        lines = file.readlines()
    with open(incar_file, 'w') as file:
        for line in lines:
            if line.startswith('MAGMOM ='):
                file.write(result_string + '\n')
            else:
                file.write(line)
    
    # Write the rearranged structure to the POSCAR file
    poscar = Poscar(struct)
    poscar.write_file(poscar_file)
    return None

"""
This function loops through all the config directories and scales
the POSCAR files to the specified volume per atom. It does not return
anything, but it changes the POSCAR files in place.
volums_per_atom (float or int) – The volume per atom in Å^3/atom
"""
def scale_poscars(vol_per_atom, configurations_directory='configurations'):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    for config_dir in config_dirs:
        poscar_file = os.path.join(configurations_directory, config_dir, 'POSCAR')
        struct = Structure.from_file(poscar_file)
        number_of_atoms = struct.num_sites
        volume = vol_per_atom * number_of_atoms
        struct.scale_lattice(volume)
        poscar = Poscar(struct)
        poscar.write_file(poscar_file)
    return None

"""
Changes lreal to false in the incar if it has less than or equal to max_atoms.
This is useful for small systems where the real space projection is not
recommended.
"""
def lreal_to_false(configurations_directory='configurations', max_atoms=10):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    for config_dir in config_dirs:
        poscar_file = os.path.join(configurations_directory, config_dir, 'POSCAR')
        struct = Structure.from_file(poscar_file)
        if struct.num_sites <= max_atoms:
            incar_file = os.path.join(configurations_directory, config_dir, 'INCAR')
            with open(incar_file, 'r') as file:
                lines = file.readlines()
            with open(incar_file, 'w') as file:
                lreal_found = False
                for line in lines:
                    if line.startswith('LREAL ='):
                        file.write('LREAL = .FALSE.\n')
                        lreal_found = True
                    else:
                        file.write(line)
                if not lreal_found:
                    file.write('LREAL = .FALSE.\n')
    return None

def change_incar_tag(tag, value, configurations_directory='configurations'):
    config_dirs = [dir for dir in os.listdir(configurations_directory) if os.path.isdir(os.path.join(configurations_directory, dir))]
    for config_dir in config_dirs:
        incar_file = os.path.join(configurations_directory, config_dir, 'INCAR')
        with open(incar_file, 'r') as file:
            lines = file.readlines()
        with open(incar_file, 'w') as file:
            tag_found = False
            for line in lines:
                if line.startswith(tag):
                    file.write(f'{tag} = {value}\n')
                    tag_found = True
                else:
                    file.write(line)
            if not tag_found:
                file.write(f'{tag} = {value}\n')
    return None


"""
df here is a dataframe that contains a 'config' column of the configs you want to
set up the ev calculation for.
"""
def set_up_ev_from_fixed_volume_calculations(df, path_to_fixed_volume_configurations, dest_configurations_path):
    try:
        os.makedirs(dest_configurations_path, exist_ok=True)
    except FileExistsError:
        print(f"Destination directory '{dest_configurations_path}' already exists.")
        return None

    for config in df['config']:
        subdirectories = [d for d in os.listdir(os.path.join(path_to_fixed_volume_configurations, f'config_{config}')) if os.path.isdir(os.path.join(path_to_fixed_volume_configurations, f'config_{config}', d))]        
        if len(subdirectories) != 1:
            print(f"ERROR: There should be exactly one subdirectory (e.g., 'vol_21') for each config '{config}' in the dataframe.\nBut there are {len(subdirectories)} subdirectories.")
            return None
        
        subdirectory = subdirectories[0]
        fixed_vol_contcar = os.path.join(path_to_fixed_volume_configurations, f'config_{config}', subdirectory, "CONTCAR")
        fixed_vol_potcar = os.path.join(path_to_fixed_volume_configurations, f'config_{config}', subdirectory, "POTCAR")
        fixed_vol_incar = os.path.join(path_to_fixed_volume_configurations, f'config_{config}', subdirectory, "INCAR")
        
        dest_poscar = os.path.join(dest_configurations_path, f'config_{config}', "POSCAR")
        dest_potcar = os.path.join(dest_configurations_path, f'config_{config}', "POTCAR")
        dest_incar = os.path.join(dest_configurations_path, f'config_{config}', "INCAR")
        
        try:
            os.makedirs(os.path.join(dest_configurations_path, f'config_{config}'), exist_ok=False)
            shutil.copy(fixed_vol_contcar, dest_poscar)
            shutil.copy(fixed_vol_potcar, dest_potcar)
            shutil.copy(fixed_vol_incar, dest_incar)
        except Exception as e:
            print(f"Error processing configuration '{config}': {str(e)}")    
    return None


# def prep_for_vasp(ywoutput, magnetic_configurations=False):
#     parse(ywoutput)

#     if magnetic_configurations == False:
#         for file in os.listdir():
#             convert_strs_to_poscars()

#     if magnetic_configurations == True:
#         remove_spin_up_less_than_down()
#         convert_strs_to_poscars()
#         make_incars(str_file)


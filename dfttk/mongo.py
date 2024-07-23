"""
Store results on MongoDB.
"""

# Standard library imports
import json
import os

# Third-party library imports
import numpy as np
from natsort import natsorted
from pymongo import MongoClient
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar

# DFTTK imports
from dfttk.aggregate_extraction import extract_configuration_data
from dfttk.eos_fit import fit_to_all_eos
from dfttk.workflows import custodian_errors_location

class MongoDBStorage:

    # Establish connection to MongoDB
    def __init__(self, path: str, connection_string: str):
        self.cluster = MongoClient(connection_string)
        self.db = self.cluster["DFTTK"]
        self.collection = self.db["community"]
        self.path = path

    def store_ev_curve(self, eos: str = "BM4"):
        vol_folders = [
            f
            for f in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, f)) and f.startswith("vol")
        ]
        vol_folders = natsorted(vol_folders)
        
        # material
        poscar = Poscar.from_file(
            os.path.join(self.path, vol_folders[0], "POSCAR.1relax")
        )
        formula = (poscar.structure.composition.formula)
        reduced_formula = (poscar.structure.composition.reduced_formula)
        number_of_atoms = (poscar.structure.composition.num_atoms)

        # properties
        ## EV curve
        #TODO: If there are errors in all vol_folders, this will not work
        vol_folders_errors = custodian_errors_location(self.path)
        vol_folders_no_errors = [f for f in vol_folders if f not in vol_folders_errors]
        
        incar_relax = Incar.from_file(
            os.path.join(self.path, vol_folders_no_errors[0], "INCAR.1relax")
        )
        incar_static = Incar.from_file(
            os.path.join(self.path, vol_folders_no_errors[0], "INCAR.3static")
        )
        kpoints = Kpoints.from_file(
            os.path.join(self.path, vol_folders_no_errors[0], "KPOINTS.1relax")
        )
        potcar = Potcar.from_file(os.path.join(self.path, vol_folders_no_errors[0], "POTCAR"))

        df = extract_configuration_data(
            self.path,
            outcar_name="OUTCAR.3static",
            contcar_name="CONTCAR.3static",
            oszicar_name="OSZICAR.3static",
        )
        
        eos_df, eos_parameters_df = fit_to_all_eos(df)
        eos_df = eos_df.drop(columns=["config", "a", "b", "c", "d", "e", "number_of_atoms"])
        eos_df = eos_df[eos_parameters_df['EOS'] == "BM4"]
        eos_df['volumes'] = eos_df['volumes'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        eos_df['energies'] = eos_df['energies'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        eos_df['pressures'] = eos_df['pressures'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        eos_dict = eos_df.to_dict(orient="records")[0]
        
        df = df.drop(columns=["config", "number_of_atoms"])
        df_dict = df.to_dict(orient="records")

        error_dict = {}
        for f in vol_folders_errors:
            file_path = os.path.join(self.path, f, "custodian.json")
            with open(file_path, 'r') as file:
                error_dict[f] = json.load(file)
        
        ev_curve_properties = {}
        ev_curve_properties["VASP input"] = {
            "INCAR relax": incar_relax,
            "INCAR static": incar_static,
            "KPOINTS": kpoints.as_dict(),
            "POTCAR": potcar.as_dict(),
            "errors": error_dict
        }
        
        i = 0
        for vol_folder in vol_folders:
            poscar = Poscar.from_file(os.path.join(self.path, vol_folder, "POSCAR.3static"))
            df_dict[i]["POSCAR"] = poscar.as_dict()
            ev_curve_properties[vol_folder] = df_dict[i]
            i += 1
        
        ev_curve_properties["EOS fit"] = eos_dict

        document = {
            "material": {
                "formula": formula,
                "reduced formula": reduced_formula,
                "number of atoms": number_of_atoms,
            },
            "properties": {"EV curve": ev_curve_properties},
        }
        self.collection.insert_one(document)
        return document
    
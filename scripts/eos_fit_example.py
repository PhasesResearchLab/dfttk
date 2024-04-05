import os
from src.eos_fit import ev_fit, pv_fit
os.chdir('FeNi')
#[results, energy_eos, pressure_eos, volume_range] = eos_fit('single', '90DW', eos_index=[1,2], plot_ev=True, plot_pv=False)

# input_files = ['FEG', '90DW', '180DW']
input_files = ['str_0', 'str_1', 'str_2', 'str_3', 'str_4', 'str_5', 'str_6', 'str_7', 'str_8', 'str_9', 'str_10',
               'str_11', 'str_12', 'str_13', 'str_14', 'str_15', 'str_16', 'str_17', 'str_18', 'str_19', 'str_20',
               'str_21', 'str_22', 'str_23', 'str_24', 'str_25', 'str_26', 'str_27', 'str_28', 'str_29', 'str_30',
               'str_31', 'str_32', 'str_33', 'str_34', 'str_35', 'str_36', 'str_37', 'str_38', 'str_39', 'str_40',
               'str_41', 'str_42', 'str_43', 'str_44', 'str_45', 'str_46', 'str_47', 'str_48', 'str_49', 'str_50',
               'str_51', 'str_52', 'str_53', 'str_54', 'str_55', 'str_56', 'str_57', 'str_58', 'str_59', 'str_60',
               'str_61', 'str_62', 'str_63', 'str_64', 'str_65', 'str_66', 'str_67', 'str_68', 'str_69', 'str_70',
               'str_71', 'str_72', 'str_73', 'str_74', 'str_75', 'str_76', 'str_77', 'str_78', 'str_79', 'str_80',
               'str_82', 'str_83', 'str_85', 'str_86', 'str_87', 'str_88', 'str_89', 'str_91', 'str_92', 'str_97']
[results, energy_eos, pressure_eos] = ev_fit('multiple', *input_files, eos_index=[3], plot_ev=True, plot_pv=False)
#input_files = ['pressure_FEG', 'pressure_90DW', 'pressure_180DW']
#[results, pressure_eos] = pv_fit('multiple', *input_files, eos_index=[4], plot_pv=True)
#print(results)
#print(pressure_eos)
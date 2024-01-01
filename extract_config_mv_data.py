import sys
import os
sys.path.append('/storage/home/lam7027/bin/vasp-job-automation')
import cstdn

 
ion_list = [i for i in range(1, 9)] #recall that range(n, m) = [n, n+1, n+2, ..., m-1]

df = cstdn.extract_config_mv_data(os.getcwd(), ion_list, outcar_name='OUTCAR.2relax')

plot_mv_data()


import time
import numpy as np                        # for manipulating arrays
from specutils import Spectrum1D          # spectrum data model
import json                               # to pretty-print dicts
from astropy import units as u            # astropy utilities
from astropy.coordinates import SkyCoord
from astropy.table import Table
from matplotlib import pyplot as plt      # visualization libs
from IPython.display import display
from getpass import getpass
from dl import authClient as ac, queryClient as qc #, storeClient as sc
from dl.helpers.utils import convert
from simulator import sim_event, sim_rubin_event, build_mu_rel_pairs

from pathlib import Path
script_dir = Path.cwd()
token = ac.login(input("Enter user name: (+ENTER) "),getpass("Enter password: (+ENTER) "))
ac.whoAmI()
if not ac.isValidToken(token):
    print('Error: invalid login for user %s (%s)' % (username,token))
else:
    print("Login token:   %s" % token)
    # print("Login token:   %s" % token)


cfg_path =script_dir / "config_file.json"
with cfg_path.open() as f:
    params = json.load(f)

ra_center = params["ra"]
dec_center = params["dec"]
radius = params["radius"]   # degrees
Ds_max = params["Ds_max"]
N =params["N"]
print("Generate ",N, " events in (ra,dec) = (",ra_center,",", dec_center,")")

query = f"""
    SELECT *
    FROM lsst_sim.simdr2
    WHERE q3c_radial_query(ra, dec, {ra_center}, {dec_center}, {radius})
      AND mu0 < (5*LOG10({Ds_max})-5)
    LIMIT {N+1000}
    """


t0_range = params["t0_range"]

model = params["model"]
system_type = params["system_type"]
path_to_save_model = params["path_save"]

print("Requesting data from AstroDataLab ...")
res = qc.query(sql=query,format='csv')
df_raw_trilegal = convert(res,'pandas')
df  = build_mu_rel_pairs(df_raw_trilegal, N, offset=0.1, min_D=1.0, random_state=None)
del df_raw_trilegal

print("Data downloaded")


for i in range(N):
    
    TRILEGAL_data = {"D_S":df["D_S"].iloc[i],
                     "D_L":df["D_L"].iloc[i],
                     "mu_rel":df["mu_rel"].iloc[i],
                     "logL":df["logl"].iloc[i],
                     "logTe":df["logte"].iloc[i],
                     "ra":df["ra"].iloc[i],
                     "dec":df["dec"].iloc[i],
                     "u":df["umag"].iloc[i],
                     "g":df["gmag"].iloc[i],
                     "r":df["rmag"].iloc[i],
                     "i":df["imag"].iloc[i],
                     "z":df["zmag"].iloc[i],
                     "Y":df["ymag"].iloc[i]                
    }
    TRILEGAL_row = df
    print("Generating event", i )
    my_own_model, pyLIMA_parameters, decision = sim_rubin_event(i, system_type, model, TRILEGAL_data ,path_to_save_model, t0_range)
    

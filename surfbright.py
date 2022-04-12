from ast import parse
from os import read
import sys
import numpy as np
import astropy
import pandas
from astropy import units as u
from astropy.io import fits

wave_scale = {"m":1e4,"A":1}    
RydE = 13.605662285137
RydA = 911.2700
RydE_x_RydA = RydE * RydA

def label_Ryd(lable):
    label = lable.split()[-1]
    wave_Ang = float(label[:-1])*wave_scale[label[-1]]
    return(RydE_x_RydA / wave_Ang)
    

def is_float(string):
    try:
        float(string)
        return(True)
    except:
        return(False)

def read_infile(m):
    with open(m["name"]+".in", 'r') as fh:
        infile = fh.read().split("\n")
    # x = [i for i, x in enumerate(infile) if x.startswith('radius')]
    start_radius_command = [x for x in infile if x.startswith('radius')]
    assert start_radius_command[0].endswith("linear parsec"), "input files are assumed to specify r0 with linear parsecs, but that is not contained in this input file"
    start_radius_command = start_radius_command[0].split()
    assert len(start_radius_command) == 4, "input should contain 'radius', float value, and 'linear parsec'. Only that format is accepted"
    m.update({"r0":float(start_radius_command[1])*u.pc.to("cm")})

def read_mol(m):
    mol = pandas.read_csv(m["name"]+".mol", delimiter="\t")
    nH = mol['H'] + mol['H+']+ mol['H-']+ mol['H2']
    
    m.update({'depth': mol['#depth'].to_numpy(), 'nH':nH.to_numpy()})

def read_ems(m):
    ems = pandas.read_csv(m["name"]+".ems", delimiter="\t")
    del(ems["#depth"])
    m.update({'ems':ems})

def make_shells(m):
    R0 = np.zeros_like(m['depth'])
    R0[0] = m['r0']
    R0[1:] = m['r0'] + m['depth'][:-1]

    R1 = m['depth'] + m['r0']
    dr = R1 - R0
    sub_dr = dr/(m['raster']+1)
    N_zones = len(R0)

    r_proj = np.zeros(m["N_inner_divisions"]+m['raster']*(1+N_zones))
    r_proj[:m["N_inner_divisions"]] = np.linspace(m["r0"]/m["N_inner_divisions"],m["r0"] ,m["N_inner_divisions"], endpoint=False)

    for i_raster in range(m['raster']):
        # m["N_inner_divisions"] -1 + m['raster'] is the last index of the inner zones before we start slicing shells
        # +1 is the first time we get into the shells
        r_proj[m["N_inner_divisions"] + m['raster'] + i_raster::m['raster']] = R0 + i_raster * sub_dr
    
    tmp_dr = ( r_proj[m["N_inner_divisions"]+m['raster']] - r_proj[m["N_inner_divisions"]-1] ) / (m['raster'] + 1)
    for i in range(1,m['raster']+1):
        r_proj[m["N_inner_divisions"] + i -1] = r_proj[m["N_inner_divisions"]-1] + tmp_dr * i


    m.update({"R0":R0, "R1":R1, "N_zones":N_zones,"r_proj":r_proj})

def read_opc(m):
    assert "N_zones" in m, "N_zones is not in model info when reading opacity. Run make shells first..."
    opc = pandas.read_csv(m["name"]+".opc", delimiter="\t",skip_blank_lines=True)

    # BUG BUG BUG
    # The opacity files from cloudy repeat the last zone, don't include it in spec_i0
    # I assume this will eventually be fixed
    # For now we see if there is on extra read in the opacity file, compared to the number of zones of the ouput
    if opc.shape[0]%(m['N_zones']+1) == 0:
        opc = opc.iloc[0:-opc.shape[0]//(m['N_zones']+1)]

    E_bin_zone = opc["#nu/Ryd"].to_numpy()
    spec_i0 = np.argwhere(E_bin_zone == E_bin_zone[0])

    spec_i0 = spec_i0.reshape(spec_i0.shape[0])
    N_spec = spec_i0[1] -spec_i0[0]
    m.update({'opc':opc, "spec_i0":spec_i0, "N_spec":N_spec, "E_bin_zone":E_bin_zone})

def trim(m):
    new_opc = np.zeros((m["N_zones"], len(m['ems'].columns) ))
    for em_label_i, em_lable in enumerate(m['ems'].columns):
        E = label_Ryd(em_lable)
        first_i = np.argwhere(abs(E - m["E_bin_zone"][:m["spec_i0"][1]-1]) == np.min(abs(E - m["E_bin_zone"][:m["spec_i0"][1]-1])))[0]
        all_i = first_i + m["spec_i0"]
        new_opc[:,em_label_i] = m['opc']['Tot opac'][all_i]
    m['opc'] = new_opc

    m['lines'] = m['ems'].columns.to_list()
    m['N_lines'] = len(m['lines'])
    m['ems'] = m['ems'].to_numpy()
    m['nH'] = np.tile(m['nH'],(m['N_lines'],1)).T

    pass

def ray_trace(m, weight=None, save_name=None):
    # Start at the bottom work up
    # Work from inside out.
    # In the inner radius, everything is valid

    if weight != None:
        weight = np.tile(weight,(m['N_lines'],1)).T
        m['ems'] *= weight
    
    Fout = np.zeros((m["r_proj"].shape[0], m['N_lines']))

    for r_proj_i, r_proj in enumerate(m['r_proj'][:m["N_inner_divisions"]+m['raster']]):

        shell_z0 = np.sqrt(np.square(m['R0']) - np.square(r_proj))
        shell_z1 = np.sqrt(np.square(m['R1']) - np.square(r_proj))
        shell_dz = np.tile(abs(shell_z1-shell_z0),(m['N_lines'],1)).T
        
        # calculate the kappa of each ray segment, from inner shell to outter shell
        shell_dtau = m['opc']*shell_dz

        # integrate a cumulative sum in starting from the inner shell  of kappa to get total kappa to the bottom shell 
        shell_net_tau = np.cumsum(shell_dtau, 0)

        # Now get the total exinction to each zone
        shell_tot_ext_zone = np.exp(-shell_net_tau)

        # and sum net flux from each zone
        Fout[r_proj_i,:] = np.sum(m['ems']*shell_dz*shell_tot_ext_zone, axis=0)

        # Repeat, but reverse the prior extinction
        shell_tot_ext_zone = np.exp(-np.cumsum(np.flip(shell_dtau,0),0))
        Fout[r_proj_i,:]  += np.sum(m['ems']*shell_dz*shell_tot_ext_zone, axis=0)


    # For projected radii outside of the inner zone, we will "slice" into one zone
    # The result is a path length much longer than the cloudy dr of the zone, and we will need to subdivide it to get the 
    # Opacity right.
    # Zones less than r_proj fall outside the ray.
    # Zones greater than follow the prior algorithm.
    # Note: Because we index things at zero we don't need to add 1 to the r_proj_i in the enumeration
    for r_proj_i, r_proj in enumerate(m['r_proj'][m["N_inner_divisions"]+m['raster']:], start=m["N_inner_divisions"]+m['raster']):
        # Find the shell we are intersecting.
        # Get the index/arg where the current r_proj is greater than R0 and less than R1
        i_slice = np.argwhere((r_proj >= m['R0']) * (r_proj <= m['R1']))[0][0]

        if i_slice < m["N_zones"] - 1:
            # Process all shells larger than the intersecting cells, which whe hit at a gradual angle.
            # We do not intersect shell less than shell_intersect. 
            shell_z0 = np.sqrt(np.square(m['R0'][i_slice+1:]) - np.square(r_proj))
            shell_z1 = np.sqrt(np.square(m['R1'][i_slice+1:]) - np.square(r_proj))
            shell_dz = np.tile(abs(shell_z1-shell_z0),(m['N_lines'],1)).T
            
            # calculate the capa of each ray segment, from inner shell to outter shell
            shell_dtau = m['opc'][i_slice+1:]*shell_dz

            # integrate a cumulative sum in starting from the inner shell  of kappa to get total kappa to the bottom shell 
            shell_net_tau = np.cumsum(shell_dtau, 0)

            # Now get the total exinction to each zone
            shell_tot_ext_zone = np.exp(-shell_net_tau)

            # and sum net flux from each zone
            Fout[r_proj_i,:] = np.sum(m['ems'][i_slice+1:]*shell_dz*shell_tot_ext_zone, axis=0)

        # Now we deal with the long path length by subdividing it.
        # First we calculate the full length throught the current sliced shell.
        slice_dz = 2* np.sqrt(m['R1'][i_slice]**2 - r_proj**2)
        slice_dr = m['R1'][i_slice] - m['R0'][i_slice]
        # This will always be at least 1. In the event it's a very ... fat zone close to the star we could have it not hit 2. So we pick a minimum of 10 bins.
        # Divide the pathlength dl of this zone into smaller pieces delta - delta_dl, so ddl. Cloudy said that the dr was safe, so we use that many dr to divide it.

        slice_ndl = min(max(10, int(slice_dz /slice_dr)),100)

        slice_ddl = slice_dz/slice_ndl

        slice_dtau = np.linspace(slice_dz/slice_ndl, slice_dz, slice_ndl)*np.tile(m['opc'][i_slice],(slice_ndl,1)).T
        slice_dtau = slice_dtau.T
        slice_tot_ext_per_ddz = np.exp(-slice_dtau)
        Fout[r_proj_i,:] = Fout[r_proj_i,:] * slice_tot_ext_per_ddz[-1,:] + np.sum(m['ems'][i_slice] * slice_ddl * slice_tot_ext_per_ddz, axis=0)

        if i_slice < m["N_zones"] - 1:

            # Repeat the other shells, but now inside out, and reverse the calculation of exctinction from out to in
            # This provides the net extinction coefficent for the top of the shell.
            shell_net_tau = np.cumsum(np.flip(shell_dtau,0),0)
            shell_tot_ext_zone = np.flip(np.exp(-shell_net_tau),0)

            # Attenuate the flux of the lower shell with the total opacity i.e. the opacity of the last upper shell.
            Fout[r_proj_i,:] = Fout[r_proj_i,:] * shell_tot_ext_zone[-1,:]
            
            # Add the net flux of the top part of the shell to the attenuated lower
            Fout[r_proj_i,:] += np.sum(m['ems'][i_slice+1:]*shell_dz*shell_tot_ext_zone, axis=0)

    hdr = fits.Header()
    hdr["r_proj_u"] = "cm"
    hdr["flux_u"] = "erg/s/cm2"

    for i_line, line in enumerate(m['lines']):
        unit = line[-1]
        line = line.replace(" ","").replace("BLND","BL").replace(".","_").replace("+","p")
        if len(line) > 8:
            line = line[:7]+unit
        hdr[line] = i_line

    primary_hdu = fits.PrimaryHDU( Fout, header=hdr)
    r_proj_hdu = fits.ImageHDU(m['r_proj'])

    c1 = fits.Column(name='lines', array=m['lines'], format='20A')
    table_hdu = fits.BinTableHDU.from_columns([c1,])
    hdul = fits.HDUList([primary_hdu, r_proj_hdu, table_hdu])
    if save_name == None:
        save_name = m["name"]+".fits"
    hdul.writeto(save_name,overwrite=True)
    pass

def surfbright(out_file, raster=3, N_inner_divisions=10):
    m = {"name":out_file.replace(".out",""), 'raster':raster, "N_inner_divisions":N_inner_divisions}
    print("Processing ", m['name'])
    # Each read routine creates dictionary entries for model data. If you want to see what those are print the dictionary keys
    read_infile(m)
    read_mol(m)
    read_ems(m)
    make_shells(m)
    read_opc(m)
    trim(m)
    ray_trace(m)

if __name__ == "__main__":
    import os
    import glob
    import sys
    import argparse
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(prog='SurfBright')
    parser.add_argument('model_search_paths', metavar='model_search_paths', type=str, nargs='+', default=["./"],
                    help='list of paths to calculate')
    parser.add_argument('--ending', metavar='model ending suffix', type=str, default=".out", help='model suffix/file ending to search for models. Default: \".out\"')
    parser.add_argument('--nproc', type=int, default=os.cpu_count(), help='number of processors %(prog)s uses to process models')

    if len(sys.argv) == 1:
        args = parser.parse_args(["./"])
    else:
        args = parser.parse_args()

    for model_search_paths in args.model_search_paths:
        files = glob.iglob(model_search_paths+'/**/*'+args.ending, recursive=True)
        # myfiles = myfiles = [file for file in files]
        # surfbright(myfiles[0])
        with Pool(args.nproc) as p:
            p.map(surfbright, files)
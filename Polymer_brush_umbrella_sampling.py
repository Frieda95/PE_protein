#obtain an estimate for the z dependence of the charge distribution.
from __future__ import print_function
import espressomd
from espressomd import code_info
from espressomd import analyze
from espressomd.interactions import *
from espressomd import electrostatics
from espressomd.io.writer import h5md  # pylint: disable=import-error
from espressomd import shapes
#from espressomd import reaction_ensemble
import espressomd.reaction_methods
import espressomd.accumulators
import espressomd.observables
import espressomd.polymer
import espressomd.visualization
import math 
import matplotlib.pyplot as plt
import random

import numpy as np
from scipy.optimize import newton
import h5py
from scipy import interpolate

#from statistic import *
#from espressomd import visualization

#import matplotlib.pyplot as plt

import sys
import gzip
import pickle
import os
import time

"""

from __future__ import print_function
import espressomd
import espressomd.analyze
import espressomd.electrostatics
import espressomd.observables
import espressomd.accumulators
import espressomd.math
import espressomd.polymer
import espressomd.reaction_methods
from scipy import interpolate

espressomd.assert_features(['ELECTROSTATICS', 'P3M', 'WCA'])

import tqdm
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

"""

#conversion_factor_from_1_per_sigma_3_to_mol_per_l=37.11711304995093
sigma = 3.55e-10 # Sigma in SI units
avo = 6.022e+23 # Avogadro's number in SI units
conversion_factor_from_1_per_sigma_3_to_mol_per_l = 1/(10**3 * avo * sigma**3) # Prefactor to mol/L
print("conversion_factor_from_1_per_sigma_3_to_mol_per_l", conversion_factor_from_1_per_sigma_3_to_mol_per_l )
temperature = 1.0
beta=1.0/temperature
pH_desired=7
pH=pH_desired
#>>>>>>>>>>>>>>>>>>>>>.Problems in this code that msut clarify with David<<<<<<<<<<<<


#1. Why for pH=7 I am not getting cH/cOH order  of 10^-14
#2. Imposed cHA  infinetesimally  low , so one of H2A and A is infinetesimally low too
#3. ONes again confirm if all concentrations and other units taken properly in bulk units not mol_per_l
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#cs_bulk=0.000016
c_poly_mol_per_L=0.01#0.15 #   Why some reaction runs are depending on this check 
c_poly=c_poly_mol_per_L/conversion_factor_from_1_per_sigma_3_to_mol_per_l
cs_mol_per_L=0.001
cs_bulk=cs_mol_per_L/conversion_factor_from_1_per_sigma_3_to_mol_per_l
Kw=10**-14 #dimensionless dissociation constant Kw=relative_activity(H)*relative_activity(OH)
cref_in_mol_per_l=1.0 #in mol/l
cH_bulk_in_mol_per_l=10**(-pH)*cref_in_mol_per_l #this is a guess, which is used as starting point of the self consistent optimization, will be modified for a desired pH at a certain salt concentration anyway. 
#cs_bulk=input_from_file



pKA =4.0 #8.3-13.3
KA = np.exp((np.log(10))*-pKA)#pKA = log10(KA)
Kcideal_in_mol_per_l=KA*cref_in_mol_per_l
#cA_bulk = 0.00016# in units of 1/sigma3

pka=8.0
#8.3-13.3
ka = np.exp((np.log(10))*-pka)#pKA = log10(KA)
pkb=-7.0 
#8.3-13.3
kb = np.exp((np.log(10))*-pkb)#pKA = log10(KA)
c0_in_mol_per_l =0#0.001#(10*cs_mol_per_L)#0.01


#box_length = disputed 
"""
|__________________________________________________Determine Bulk concentrations self consistently____________________________|
|_____________________________________________________________________________________________________________________________|

Form a bulk ionic composition constituting {Na+, Cl-,H+, OH-} imposing electroneutrality of the reservoir. Take [H+] and an 
initial salt concentration as inputs then from this Kw=[H+][OH-] get [OH-]; electroneutrality gives [Na+] and [Cl-] . Since inputs
are in mol/L. We want all in 1/sigma3. 
"""

#conversion_factor_from_1_per_sigma_3_to_mol_per_l=37.1
    
    

def determine_bulk_concentrations_selfconsistently(arg_cH_bulk , arg_gamma_res):
    #Globally read
    global Kw, cref_in_mol_per_l, conversion_factor_from_1_per_sigma_3_to_mol_per_l, pH, pka, pkb
    global cref_sim, c0_sim
    
    # Globally edited 
    global cHA_bulk, cH2A_bulk, cA_bulk, cNa_bulk, cCl_bulk, cOH_bulk, cH_bulk
    global cH_bulk_in_mol_per_l, gamma_res

    gamma_res = arg_gamma_res
    cH_bulk = arg_cH_bulk
    cH_bulk_in_mol_per_l = cH_bulk*conversion_factor_from_1_per_sigma_3_to_mol_per_l
    cOH_bulk=(Kw/(cH_bulk_in_mol_per_l/cref_in_mol_per_l))*cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l
    
    FAC1 = ((10**(-pkb))*(cH_bulk/cref_sim)) + 1
    cA_bulk = c0_sim / ((gamma_res*FAC1*(cH_bulk/cref_sim)*(10**pka))  + 1)
    cH2A_bulk = c0_sim - (cA_bulk * ((1.0/((10**(-pka)) * (cref_sim/cH_bulk) * (gamma_res**(-1))))+1))
    cHA_bulk = c0_sim - cH2A_bulk - cA_bulk
    
#    print("\n After >> \n FAC - ", FAC1, "\n cH_bulk - ",cH_bulk,  "\n gamma_res -", gamma_res, "\n c0_sim - ", c0_sim, "\n cA_bulk = (ka*cHA_bulk)/(gamma_res*cH_bulk) - ", cA_bulk, "\n cH2A_bulk = kb*cHA_bulk*cH_bulk - ", cH2A_bulk,  cA_bulk * ((1.0/((10**(-pka)) * (cref_sim/cH_bulk) * (gamma_res**(-1))))+1),   ((gamma_res*FAC1*(cH_bulk/cref_sim)*(10**pka))  + 1))

    if((cOH_bulk+cA_bulk)>=(cH_bulk+cH2A_bulk)):
        cNa_bulk=cs_bulk+(cOH_bulk+cA_bulk-cH_bulk-cH2A_bulk)
        cCl_bulk=cs_bulk
    else:
        cCl_bulk=cs_bulk+(cH_bulk+cH2A_bulk-cOH_bulk-cA_bulk)
        cNa_bulk=cs_bulk



    return np.array([cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk, cH2A_bulk, cHA_bulk, cA_bulk])

"""____________________________________________________________________________________________________________________________|
_______________________________________________________________________________________________________________________________|
"""
"""
|_______________________________________Recursion to converge pH of reservoir to the desired value_____________________________|
|______________________________________________________________________________________________________________________________|
pH = -(1/ln10). ln(C_H+/C_ref) - (1/ln10).(\bta\mu_{H+}/2)
pH = -log_10{(C_H+/C_ref)*exp(\bta\mu_{H+}/2)}
here;
\mu_{H+} is a function of ionic strength I_res(current_concentrations) of the reservoir 
"""
#/tikhome/keerthirk/keerthi_ICP/worksheets_david/scripts/widom_insertion


print('entered')

#       Here,  ionic strength is in bulk units (1/sigma^3) and so is other values  
ionic_strength, excess_chemical_potential_monovalent_pairs_in_bulk_data ,value_of_lB ,excess_chemical_potential_monovalent_pairs_in_bulk_data_error =np.loadtxt("/tikhome/keerthirk/espresso/build_new/test-for-PE_brush/new/excess_chemical_potential_david_lB2_uncommented.dat", unpack=True)


excess_chemical_potential_monovalent_pairs_in_bulk=interpolate.interp1d(ionic_strength, excess_chemical_potential_monovalent_pairs_in_bulk_data)

#                   (((((((    IDEAL CASE CHANGE   )))))))))))


#def excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk):

#    return 0



max_self_consistent_runs=200
self_consistent_run=0

def calculate_alphas_self_consistently(arg_pH, arg_cs_mol_per_L, arg_c0_in_mol_per_l, arg_pka, arg_pkb):
    global cs_mol_per_L, cs_bulk,  c0_in_mol_per_L, c0_sim,  pH, pka, pkb
    global cH_bulk_in_mol_per_l
    global cNa_bulk, cCl_bulk, cOH_bulk, cH_bulk, cH2A_bulk, cHA_bulk, cA_bulk, ionic_strength_bulk
    global Kw, cref_in_mol_per_l, cref_sim,conversion_factor_from_1_per_sigma_3_to_mol_per_l
    global determined_pH

    cs_mol_per_L = arg_cs_mol_per_L
    c0_in_mol_per_l = arg_c0_in_mol_per_l
    pH = arg_pH
    pka = arg_pka
    pkb = arg_pkb

    cs_bulk=cs_mol_per_L/conversion_factor_from_1_per_sigma_3_to_mol_per_l
    cref_sim = cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l
    c0_sim = c0_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l

    cH_bulk_in_mol_per_l=10**(-pH)*cref_in_mol_per_l #this is a guess, which is used as starting point of the self consistent optimization
    cH_bulk=cH_bulk_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l

    gamma_res = 1.0 #initially considering the ideal case 
    cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk, cH2A_bulk, cHA_bulk, cA_bulk =determine_bulk_concentrations_selfconsistently(cH_bulk, gamma_res )

    ionic_strength_bulk=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk+cH2A_bulk+cA_bulk) #in units of 1/sigma^3
    determined_pH=-np.log10(cH_bulk*conversion_factor_from_1_per_sigma_3_to_mol_per_l/cref_in_mol_per_l*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/(2.0*temperature)))

    while abs(determined_pH-pH)>1e-6:
        if(determined_pH)>pH:
            cH_bulk=cH_bulk*1.005
        else:
            cH_bulk=cH_bulk/1.003
        gamma_res = np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/(1.0*temperature))
        cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk, cH2A_bulk, cHA_bulk, cA_bulk =determine_bulk_concentrations_selfconsistently(cH_bulk, gamma_res)

        ionic_strength_bulk=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk+cH2A_bulk+cA_bulk) #in units of 1/sigma^3
    
        determined_pH=-np.log10(cH_bulk*conversion_factor_from_1_per_sigma_3_to_mol_per_l*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/(2.0*temperature)))
#        print(pH, determined_pH, gamma_res, ionic_strength_bulk )
#    print("\n Before exiting Loop >> \n cH_bulk - ", cH_bulk, "\n cHA_bulk - ",cHA_bulk,  "\n cH2A_bulk -", cH2A_bulk, "\n cOH_bulk - ", cOH_bulk, "\n cA_bulk - ", cA_bulk, "\n cNa_bulk - ", cNa_bulk, "\n cCl_bulk - ", cCl_bulk)
#    print('Determined pH in NonIdeal case ', determined_pH)
    return 0,0,0
#    return cH2A_bulk/c0_sim, cA_bulk/c0_sim, (cA_bulk-cH2A_bulk)/(cA_bulk+cHA_bulk+cH2A_bulk)

def ideal_alpha_acid(pH, pKA):
    return 1. / (1 + 10**(pKA - pH))
"""
pH_range = np.linspace(0.0, 14.0, 500)
alphas_positive = []
alphas_negative = []
alphas_net = []
for pH in pH_range:
    alphas_temp = calculate_alphas_self_consistently(pH, cs_mol_per_L, c0_in_mol_per_l, pka, pkb)
    alphas_positive.append(alphas_temp[0])
    alphas_negative.append(alphas_temp[1])
    alphas_net.append(alphas_temp[2])

#plt('font', **{'family':'serif','serif':['Helvetica']})
#plt('text', usetex=True)
plt.plot(pH_range, alphas_positive, label="base")
plt.plot(pH_range, alphas_negative, label="acid")
plt.plot(pH_range, alphas_net, label="effective")
plt.plot(pH_range, ideal_alpha_acid(pH_range, 10.0), label="acid (HH)", linestyle="dotted")
plt.xlabel('pH')
plt.ylabel('Degree of ionization ')
plt.legend()
#plt.show()
plt.savefig('alpha_ampholyte_ion_calc_self_consistent.png')


#f = open("sunil.mpc", "a")
"""

pH= pH_desired
alphas_temp = calculate_alphas_self_consistently(pH, cs_mol_per_L, c0_in_mol_per_l, pka, pkb)
print('Determined pH in NonIdeal case ', determined_pH)


print( "pH - ",pH, "\n cs_mol_per_L - ",cs_mol_per_L,"\n  c0_in_mol_per_l -", c0_in_mol_per_l,"\n pka -",pka,"\n pkb - ",pkb, "\n cs_bulk - ",cs_bulk ,"\n c0_sim -",c0_sim, "\n cH_bulk - ", cH_bulk, "\n cHA_bulk - ",cHA_bulk,  "\n cH2A_bulk -", cH2A_bulk, "\n cOH_bulk - ", cOH_bulk, "\n cA_bulk - ", cA_bulk, "\n cNa_bulk - ", cNa_bulk, "\n cCl_bulk - ", cCl_bulk)
#f.write("pH - ",pH, "\n cs_mol_per_L - ",cs_mol_per_L,"\n  c0_in_mol_per_l -", c0_in_mol_per_l,"\n pka -",pka,"\n pkb - ",pkb, "\n cs_bulk - ",cs_bulk ,"\n c0_sim -",c0_sim, "\n cH_bulk - ", cH_bulk, "\n cHA_bulk - ",cHA_bulk,  "\n cH2A_bulk -", cH2A_bulk, "\n cOH_bulk - ", cOH_bulk, "\n cA_bulk - ", cA_bulk, "\n cNa_bulk - ", cNa_bulk, "\n cCl_bulk - ", cCl_bulk)
#f.close()



#print("\n After exiting Loop >> \n cH_bulk - ", cH_bulk, "\n cHA_bulk - ",cHA_bulk,  "\n cH2A_bulk -", cH2A_bulk, "\n cOH_bulk - ", cOH_bulk, "\n cA_bulk - ", cA_bulk, "\n cNa_bulk - ", cNa_bulk, "\n cCl_bulk - ", cCl_bulk)
#print(np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )))
"""____________________________________________________________________________________________________________________________|
_______________________________________________________________________________________________________________________________|
"""


"""
|_______________________________________Basic concentration checks_____________________________________________________________|
|______________________________________________________________________________________________________________________________|
Checks for kw-((c_H+/c_ref).(c_OH-/c_ref).exp(2\bta\mu/2))=0 ; 
           sum{z_i c_i}=0 ;
           pH_desired-pH_determined=0 ; 
           cs_salt_fixed = min(c_Na, c_cl) ;
           At pH 7 conce. of H+ and OH- same ; 


"""
print('Ionic strength', ionic_strength_bulk, 'value ', excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk))


#def check_concentrations():
"""
#    if(abs(Kw-cOH_bulk*cH_bulk*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/temperature)*conversion_factor_from_1_per_sigma_3_to_mol_per_l**2/cref_in_mol_per_l**2)>1e-15):
#        raise RuntimeError("Kw incorrect")
# UNCOMMENT THIS AND CONFIRM WHY 10^-15
"""
def check_concentrations():

    if(abs(cNa_bulk+cH_bulk+cH2A_bulk-cOH_bulk-cCl_bulk-cA_bulk)>1e-14):
        raise RuntimeError("bulk is not electroneutral")
    if(abs(pH-determined_pH)>1e-6):
        raise RuntimeError("pH is not compatible with ionic strength and bulk H+ concentration")
    if(abs(cs_bulk-min(cNa_bulk, cCl_bulk))>1e-14):
        raise RuntimeError("bulk salt concentration is not correct")
    if(abs(pH-7)<1e-14):
        if((cH_bulk/cOH_bulk-1)>1e-14):
            print(cH_bulk, cOH_bulk)
#            raise RuntimeError("cH and cOH need to be symmetric at pH 7")

#            print(cH_bulk*50*50*50, cOH_bulk*50*50*50)
#IMPORTANT COMMENTED            raise RuntimeError("cH and cOH need to be symmetric at pH 7")



#print("after self consistent concentration calculation: cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk, cH2A_bulk, cA_bulk, cHA_bulk", cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk, cH2A_bulk, cA_bulk, cHA_bulk)
print ("after self consistent concentration calculation: ionic strength - ",  ionic_strength_bulk,"cNa_bulk, cCl_bulk, cOH_bulk, cH_bulk, cH2A_bulk, cHA_bulk, cA_bulk: ", cNa_bulk, cCl_bulk, cOH_bulk, cH_bulk, cH2A_bulk, cHA_bulk, cA_bulk)
print("check KW: ",Kw, cOH_bulk*cH_bulk*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/temperature)*conversion_factor_from_1_per_sigma_3_to_mol_per_l**2/cref_in_mol_per_l**2)
print((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk)))
print("check electro neutrality bulk after", cNa_bulk+cH_bulk+cH2A_bulk-cOH_bulk-cCl_bulk-cA_bulk) #note that charges are neutral up to numerical precision. femto molar charge inequalities are not important in the bulk.
print("check pH: input", pH, "determined pH", determined_pH)
print("check cs bulk: input", cs_bulk, "determinde cs_bulk", min(cNa_bulk, cCl_bulk))
#print("check cH_bulk/cOH_bulk:", cH_bulk/cOH_bulk)
print("ka, kb, cHA_in_mol_per_l :", ka, kb, (cHA_bulk*conversion_factor_from_1_per_sigma_3_to_mol_per_l))

check_concentrations()


print ("Uptill here !!!!!!!!!!!!!")


def MC_swap_A_HA_particles(type_HA, type_A):
    As=system.part.select(type=type_A)
    HAs=system.part.select(type=type_HA)
    ids_A=As.id
    ids_HA=HAs.id
    if(len(ids_A)>0 and len(ids_HA)>0):
        #choose random_id_A, choose_random_id_HA
        random_id_A=ids_A[np.random.randint(0,len(ids_A))]
        random_id_HA=ids_HA[np.random.randint(0,len(ids_HA))]

        old_energy=system.analysis.energy()["total"]
        #adapt type and charge
        system.part[random_id_A].type=type_HA
        system.part[random_id_A].q=0
        system.part[random_id_HA].type=type_A
        system.part[random_id_HA].q=-1
        new_energy=system.analysis.energy()["total"]
        #apply metropolis criterion, accept or reject based on energetic change
        if(np.random.random()<min(1,np.exp(-(new_energy-old_energy)/temperature))):
            #accept
            pass
        else:
            #reject
            system.part[random_id_A].type=type_A
            system.part[random_id_A].q=-1
            system.part[random_id_HA].type=type_HA
            system.part[random_id_HA].q=0

"""
def rescale_system_with_changing_grafting_density(graf_dens_sim, Num_polymers_in_brush):
    
    global box_l_x, box_l_y, Volume
    sqrt_Num_polymers_in_brush = (np.sqrt(Num_polymers_in_brush))
    if isinstance(sqrt_Num_polymers_in_brush, float):
        raise RuntimeError("Num of polymers in brush has to be a a proper root")


    unit_step = np.sqrt((1.0/graf_dens_sim))
    new_length = unit_step*sqrt_Num_polymers_in_brush
    box_l_x=new_length
    box_l_y=new_length

    Volume = box_l_x*box_l_y*box_l_z_tube # in units of 1/sigma3

    # rescale box
    system.change_volume_and_rescale_particles(d_new=box_length)
    print("Rescaled the simulation box.", flush=True)

    p3m = espressomd.electrostatics.P3M(**P3M_PARAMS, **CI_PARAMS)
    system.actors.add(p3m)



def system_setup(c_salt_SI):
    print(f"Salt concentration: {c_salt_SI:.12g} mol/l", flush=True)
    c_salt_sim = c_salt_SI /conversion_factor_from_1_per_sigma_3_to_mol_per_l
    box_length = np.cbrt(N_ion_pairs / c_salt_sim)

    # rescale box
    system.change_volume_and_rescale_particles(d_new=box_length)
    print("Rescaled the simulation box.", flush=True)

    p3m = espressomd.electrostatics.P3M(**P3M_PARAMS, **CI_PARAMS)
    system.actors.add(p3m)
"""

"""________________________________________________DEFINING REACTIONS__________________________________________________________|
_______________________________________________________________________________________________________________________________|
"""
type_H=0
type_A=1
type_HA=2
type_OH=3
#type_constraint=4
type_Na=4
type_Cl=5
type_constraint_A=6
type_constraint_B=7
type_HA_zwit=8
type_A_zwit=9
type_H2A_zwit=10
type_Usampling_wall=11
type_probe=[type_HA_zwit, type_A_zwit, type_H2A_zwit]


z_H=+1
z_A=-1
z_HA=0
z_OH=-1
#z_constraint=
z_Na=+1
z_Cl=-1
z_constraint=0
z_HA_zwit=0
z_H2A_zwit=+1
z_A_zwit=-1
#z_probe=+1
z_Usampling_wall=0


types_neutral_polymer=[type_HA, type_constraint_A, type_constraint_B]
type_ions=[type_H ,type_OH, type_Na, type_Cl , type_HA_zwit, type_A_zwit, type_H2A_zwit]
types=[type_H, type_A, type_HA,type_OH, type_Na, type_Cl, type_constraint_A, type_constraint_B , type_HA_zwit, type_A_zwit, type_H2A_zwit]
types_without_zwitterion=[type_H, type_A, type_HA,type_OH, type_Na, type_Cl, type_constraint_A, type_constraint_B ]
types_without_polymer=[type_H,type_OH, type_Na, type_Cl, type_constraint_A, type_constraint_B , type_HA_zwit, type_A_zwit, type_H2A_zwit]
types_polymer=[type_A, type_HA]



#types_without_wall=[type_H, type_A, type_HA,type_OH, type_Na, type_Cl,  type_HA_zwit, type_A_zwit, type_H2A_zwit]
#types=[type_H, type_A, type_HA,type_OH, type_Na, type_Cl, type_HA_zwit, type_A_zwit, type_H2A_zwit]
charges=[z_H, z_A, z_HA, z_OH, z_Na, z_Cl, z_HA_zwit, z_H2A_zwit, z_A_zwit]
#Nions=[N_H, N_A, N_HA, N_OH, N_Na, N_Cl]





"""
bead_start_pos_arr_1D = np.array([1,2,3])#([123]) A list 
bead_start_pos_arr_2D = np.array([ [1,2,3],[4,5,6],[7,8,9]  ])#([ [123],[123],[123],...] )A list of lists
bead_start_pos_arr_3D = np.array( [ [[1,2,3],[4,5,6],[7,8,9]]  ,  [['a','s','d'],['f','g','h'],['a','s','d']]  ,  [['b','h','m'],['q','r','r'],['o','l','p']]  ]  )#([ [[],[],[]] , [[],[],[]] , [[],[],[]]  ] )A list of lists of lists
print ('bead_start_pos_arr_1D.shape', bead_start_pos_arr_1D.shape)
print ('bead_start_pos_arr_2D.shape', bead_start_pos_arr_2D.shape)
print ('bead_start_pos_arr_3D.shape', bead_start_pos_arr_3D.shape)
"""

#______________________Grafting density and setting up box dimensions________
graf_dens_sim=0.1

Num_polymers_in_brush=100
#Num_beads_each_polymer=
Num_monomers_maximum=100
Num_beads_each_polymer=Num_monomers_maximum
sqrt_Num_polymers_in_brush = (int)(np.sqrt(Num_polymers_in_brush))
#sqrt_check_Num_polymers_in_brush = (np.sqrt(Num_polymers_in_brush))
#if isinstance(sqrt_check_Num_polymers_in_brush, float):
#    raise RuntimeError("Num of polymers in brush has to be a a proper root")


unit_step = np.sqrt((1.0/graf_dens_sim))
new_length = unit_step*sqrt_Num_polymers_in_brush
box_l_x=new_length
box_l_y=new_length
box_l_z_tube=150
elc_gap=30
box_l_z=box_l_z_tube+elc_gap
Volume = box_l_x*box_l_y*box_l_z_tube # in units of 1/sigma3
print("Box DImensions ")
print(box_l_x, box_l_y, box_l_z, unit_step, sqrt_Num_polymers_in_brush, Num_polymers_in_brush)
slab_width=box_l_z_tube#(wall_offset_from_box_boundary - (box_l_z-wall_offset_from_box_boundary) )
wall_offset=0#box_l_z - ((box_l_z/2.0)+(slab_width/2.0))

system = espressomd.System(box_l=[box_l_x, box_l_y, box_l_z])
TIME_STEPS = 0.01
system.time_step = TIME_STEPS
system.cell_system.skin = 0.4

LJ_EPSILON = 1.0
LJ_SIGMA = 1.0
print("-------------- Box DImensions---------------------")
print(box_l_x, box_l_y, box_l_z)
#===============================Setting up the polymer brush===================================

# Setup an array of initial adhered beads 
#=======================================

bead_start_pos_arr = np.empty((Num_polymers_in_brush, 3), dtype=float)

z_offset=wall_offset+LJ_SIGMA
particle_count =0;


if(particle_count < Num_polymers_in_brush):

    for i in range (sqrt_Num_polymers_in_brush ):
        for j in range (sqrt_Num_polymers_in_brush):
            bead_start_pos_arr[particle_count][0] = (i+1)*unit_step
            bead_start_pos_arr[particle_count][1] = (j+1)*unit_step
            bead_start_pos_arr[particle_count][2] = z_offset#system.box_l[0]/2.0
            particle_count = particle_count+1


# Check if the correct grafting density is implemented
#=====================================================


min_x = +10000000
max_x = -10000000
min_y = +10000000
max_y = -10000000

for i in range (Num_polymers_in_brush):
    if(bead_start_pos_arr[i][0]<min_x):
        min_x = bead_start_pos_arr[i][0]
    if(bead_start_pos_arr[i][0]>max_x):
        max_x = bead_start_pos_arr[i][0]
    if(bead_start_pos_arr[i][1]<min_y):
        min_y = bead_start_pos_arr[i][1]
    if(bead_start_pos_arr[i][1]>max_y):
        max_y = bead_start_pos_arr[i][1]

max_x = max_x
min_x = min_x-unit_step
max_y = max_y
min_y = min_y-unit_step
print(min_x, max_x, min_y, max_y)
graf_dens_calc = Num_polymers_in_brush/((max_x-min_x)*(max_y-min_y))
print("Calculated grafting density is :  ", graf_dens_calc )

Num_polymer_ls_seq=[]
#Num_poly_read= np.zeros((20,20))

total_num_beads = 0
Num_poly_read = np.loadtxt("/work/keerthirk/spherical_brush/attractive_LJ/PolydispersityInd_1/PolymerList.txt" )
lets_start=0
for j in range (len(Num_poly_read)):
    total_num_beads = total_num_beads + (Num_poly_read[j][0]*Num_poly_read[j][1])
#    Num_monomers = int(Num_poly_read[j][1])
    for i in range (int(Num_poly_read[j][1])):
        lets_start= lets_start +1
        print(lets_start , Num_poly_read[j][0])
        Num_polymer_ls_seq.append(Num_poly_read[j][0])

random.shuffle(Num_polymer_ls_seq)


list_of_linear_positions = np.zeros((Num_polymers_in_brush, Num_monomers_maximum, 3), dtype=float)
for i in range (len(Num_polymer_ls_seq)):
    Count_monomers_in_this_poly=Num_polymer_ls_seq[i]
    for j in range(int(Count_monomers_in_this_poly)):
        list_of_linear_positions[i][j][0] = bead_start_pos_arr[i][0]
        list_of_linear_positions[i][j][1] = bead_start_pos_arr[i][1]
        list_of_linear_positions[i][j][2] = bead_start_pos_arr[i][2]+(1.0*j)
        
        
     



'''
list_of_linear_positions = np.empty((Num_polymers_in_brush,Num_beads_each_polymer, 3), dtype=float)
#print('list_of_linear_positions:  Shape: ', list_of_linear_positions.shape )
#print( list_of_linear_positions )


for i_poly in range (Num_polymers_in_brush):
    for i_monomer in range (Num_beads_each_polymer):
#        if(i_monomer==0):
#            list_of_linear_positions[i_poly][i_monomer] = bead_start_pos_arr[i_poly]
#        else: 
            list_of_linear_positions[i_poly][i_monomer][0] = bead_start_pos_arr[i_poly][0]
            list_of_linear_positions[i_poly][i_monomer][1] = bead_start_pos_arr[i_poly][1]
            list_of_linear_positions[i_poly][i_monomer][2] = bead_start_pos_arr[i_poly][2]+(1*i_monomer)    
   
'''


pi =np.pi
print("Piiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii", pi)
#================================Setting up the walls at Lz=0 and Lz=box_l_z====================

if (wall_offset!=0):
    raise RuntimeError("Wall offset must be at zero to Lz when using ELC gap explicitly")
# bottom wall, normal pointing in the +z direction, laid at z=0.1

floor = espressomd.shapes.Wall(normal=[0, 0, 1], dist=wall_offset)
c1 = system.constraints.add(
    particle_type=type_constraint_A, penetrable=False, only_positive=False, shape=floor)

steps = 1
wall_pr_in_x = (int)(box_l_x/steps)
wall_pr_in_y = (int)(box_l_y/steps)
particle_pos_floor=np.empty(((wall_pr_in_x*wall_pr_in_y),3), dtype=float)
particle_pos_ceil=np.empty(((wall_pr_in_x*wall_pr_in_y),3), dtype=float)

wall_pr_id=0

for i in range (wall_pr_in_x):
    for j in range (wall_pr_in_y):

        particle_pos_floor[wall_pr_id][0] =(i*steps)
        particle_pos_floor[wall_pr_id][1] =(j*steps)
        particle_pos_floor[wall_pr_id][2] =wall_offset
        wall_pr_id=wall_pr_id+1;

#print("!!!!!!!!!!!!!!!!!Printing particle floor positions......\n")
#print(particle_pos_floor.shape, "\n")
#print(particle_pos_floor)




# top wall, normal pointing in the -z direction, laid at z=49.9, since the
# normal direction points down, dist is -49.9
ceil = espressomd.shapes.Wall(normal=[0, 0, -1], dist=-(box_l_z_tube - wall_offset))  #WHY NEGATIVE 
c2 = system.constraints.add(
    particle_type=type_constraint_B, penetrable=False, only_positive=False, shape=ceil)

wall_pr_id=0

for i in range (wall_pr_in_x):
    for j in range (wall_pr_in_y):

        particle_pos_ceil[wall_pr_id][0] =(i*steps)
        particle_pos_ceil[wall_pr_id][1] =(j*steps)
        particle_pos_ceil[wall_pr_id][2] =(box_l_z_tube-wall_offset)
        wall_pr_id=wall_pr_id+1;

print("!!!!!!!!!!!!!!!!!Printing particle ceil positions......\n")
#print(particle_pos_ceil.shape, "\n")
#print(particle_pos_ceil)


#David I figured a small technicality related to the equiliberation without break of bonds . Actually as you suggested if we only call 10^6 times integrator without electrostatics and then switch on electrostatics . The system still blew up. Because in my case my system initially starts with no counterions , so the presence of huge intra chain repulsion still blew the system even after warmup.  So I had to make reaction calls as well when equiliberating without electrostatics , so that by the time electrostatics is switched on there is enough screening by counterions within the brush to avoid huge repulsive forces
#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()
#visualizer.screenshot(f'screenshot_wall.png')


#POLYMER_PARAMS = {'n_polymers': Num_polymers_in_brush, 'beads_per_chain':25, 'bond_length': 1, 'seed': 42, 'min_distance': 0.9, 'start_positions':}
#POLYMER_PARAMS = {'n_polymers': Num_polymers_in_brush, 'beads_per_chain': Num_beads_each_polymer, 'bond_length': 1, 'seed': 42, 'min_distance': 0.9,'start_positions':bead_start_pos_arr}#, 'bond_angle':pi/4}
bjerrum=2.0
#system.seed=np.random.randint(np.random.randint(1000000))#   DOUBT WHY SHOWING PROBLEM
#Ensure Charge neutrality with the int used rounding off it can go wrong

#N_POLY = Num_polymers_in_brush*Num_beads_each_polymer#int(c_poly*Volume)#N_A + N_HA = N_POLY
N_POLY = total_num_beads
N_HA = N_POLY
Na_cpoly_vol=(N_HA-N_POLY)
N_A=np.abs(Na_cpoly_vol)
N_H = 0
N_OH = 0# int((cOH_bulk*Volume))
N_Na = 0#int((cNa_bulk*Volume))
N_Cl = 0#int((cCl_bulk*Volume))
N_HA_zwit = 0
N_A_zwit = 0
N_H2A_zwit=0
Nions=[N_H, N_A, N_HA, N_OH, N_Na, N_Cl, N_HA_zwit, N_A_zwit, N_H2A_zwit]


print('N_H-',N_H, "N_HA-", N_HA,  "  N_OH-",N_OH, "  N_Na-",N_Na, "  N_Cl-",N_Cl, "  N_A-",N_A, "N_HA_zwit", N_HA_zwit,"N_A_zwit",  N_A_zwit,"N_H2A_zwit",  N_H2A_zwit)
print("----------------------------->Ionic strength: ",ionic_strength_bulk)
#print ("---------------------------->Concentration of A- ions: ", (N_A/Volume))


#===========================================================================================



#===============================SET UP INTERACTIONS==========================================
#LJ_EPSILON = 1.0
#LJ_SIGMA = 1.0


#_______________________introduce fene to system as a bonded interaction_________________
fene = espressomd.interactions.FeneBond(k=30, d_r_max=2.0)
system.bonded_inter.add(fene)

#harmonic = HarmonicBond(k=30.0, r_0=0.0)
#system.bonded_inter.add(harmonic)


#__________________________LJ intercation between counterions, monomers and salt_______________
#                   (((((((    IDEAL CASE CHANGE   )))))))))))
#for i in types:
#    system.non_bonded_inter[i, type_constraint_A].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)
#    system.non_bonded_inter[i, type_constraint_B].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)



# ion-ion interaction
for i in types_without_zwitterion:
    for j in types_without_zwitterion:
        system.non_bonded_inter[i, j].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)

for i in types_without_polymer:# Goto line 1036 to correct for raspberry ions
    system.non_bonded_inter[i, type_HA_zwit].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)
    system.non_bonded_inter[i, type_H2A_zwit].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)
    system.non_bonded_inter[i, type_A_zwit].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)

for i in types_polymer:# Goto line 1036 to correct for raspberry ions
    system.non_bonded_inter[i, type_HA_zwit].lennard_jones.set_params(epsilon=LJ_EPSILON+1, sigma=LJ_SIGMA, cutoff=LJ_SIGMA * 2.5, shift="auto")
    system.non_bonded_inter[i, type_H2A_zwit].lennard_jones.set_params(epsilon=LJ_EPSILON+1, sigma=LJ_SIGMA, cutoff=LJ_SIGMA * 2.5, shift="auto")
    system.non_bonded_inter[i, type_A_zwit].lennard_jones.set_params(epsilon=LJ_EPSILON+1, sigma=LJ_SIGMA, cutoff=LJ_SIGMA * 2.5, shift="auto")

#for i in types_polymer:
#    system.non_bonded_inter[i, type_HA_zwit].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)
#    system.non_bonded_inter[i, type_H2A_zwit].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)
#    system.non_bonded_inter[i, type_A_zwit].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)




#        system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA, cutoff=LJ_SIGMA * 2**(1.0/6.0), shift="auto")




#_________________________Build polymer with fene potential________________________________
#Note: espressomd.polymer.linear_polymer_positions returns a 3D numpy array . SInce we have a 
#list conatining N entries (y cols) each with 3 positional values (z cols) its essentially 2D . SO we have to invoke 
#by default the list_of_linear_positions[0] whcih is of the size list_of_linear_positions[0][N][3]
# Here list_of_linear_positions is the self defined linear array of polymers in a PE brush

start_part_id_list=[]
track_brush_particle_id=[]

def build_polymer( type_of_bond_interaction):
    global track_brush_particle_id
#    list_of_linear_positions = espressomd.polymer.linear_polymer_positions( **polymer_params)
#    print ("Building a polymer with parameters:  ", polymer_params)

#    print(list_of_linear_positions)
#    print(list_of_linear_positions.shape)
#    print (list_of_linear_positions[:][0])
    



    for N_pol in range (len(Num_polymer_ls_seq)):
        Count_monomers_in_this_poly=Num_polymer_ls_seq[N_pol]
#        print("len(Num_polymer_ls_seq)", len(Num_polymer_ls_seq))
        par_prev=None
        start_par = 1
        for i in range(len(list_of_linear_positions[N_pol])): # here since positions is a list of list of lists shape(10,50,3) ,positions[N_poly] will give a list of 50 monomer.pos of the (N_polyth-1) chain.
         
            if(i<Count_monomers_in_this_poly):
                par=system.part.add(pos=list_of_linear_positions[N_pol][i], type=type_HA, q=charges[type_HA])
                track_brush_particle_id.append(par.id)
                print(N_pol , i ,  par.pos)

                if start_par==1:
                    start_par=0
                    par.fix = [True, True, True]
                    start_part_id_list.append(par.id)

                if par_prev is not None:
                    par.add_bond((type_of_bond_interaction, par_prev))
                par_prev = par

    


"""
    previous_part = None
    for i in list_of_linear_positions[0]:
        part=system.part.add(pos=i, type=type_A, q=charges[type_A])
        
        if previous_part:
            part.add_bond((type_of_bond_interaction, previous_part))
        previous_part = part
"""
print("Building polymer brush .....")
#build_polymer( fene)# We are initializing a chain with all ions as HA ions, with zero A-



#particle_id = system.part.all().id
#particle_type = system.part.all().type
#particle_pos = system.part.all().pos


#print("POlymer", system.part.all().pos)
#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()
#visualizer.screenshot(f'polymer_brush_1.png')

#vmd_file = open("movie.out", "a")
num_ls = []
#num_ls.append(Num_polymers_in_brush*Num_beads_each_polymer)

#np.savetxt("movie.out" , np.column_stack([particle_pos]), fmt='%.7f\t', delimiter='\t')

#np.savetxt("movie_ls.out" , particle_pos, fmt='%.7f\t', delimiter='\t')




#____________________________Add counterions and salt in system_____________________________

#N_ions=[10]
#counter_ions=[]
"""
ion_positions=np.empty((3),dtype=float)
print ("Setting counterions H+ at t=0....")
for i in range(N_H):
    ion_positions[0] = np.random.random(1) * system.box_l[0]
    ion_positions[1] = np.random.random(1) * system.box_l[1]
#    ion_positions[2] = (np.random.random(1) * box_l_z_tube) + wall_offset

    rg1=wall_offset+5   
    rg2=box_l_z_tube-wall_offset-5
    ion_positions[2] = ((rg2-rg1)*np.random.random(1)) + rg1   #rg1-rg2 desired (rg2-rg1)*a + rg1

    par = system.part.add(pos=ion_positions, type=type_H  , q=charges[type_H])
#    print ("Particle_id:  ",par, "  ; Particle Pos: ", par.pos, "  ; Particle Type: ", par.type,"  ; Particle Charge: ", par.q)

print ("Setting counterions HA at t=0....")
for i in range(N_HA):
#    ion_positions = np.random.random(3) * system.box_l
    ion_positions[0] = np.random.random(1) * system.box_l[0]
    ion_positions[1] = np.random.random(1) * system.box_l[1]
#    ion_positions[2] = (np.random.random(1) * box_l_z_tube) + wall_offset
    rg1=wall_offset+5   
    rg2=box_l_z_tube-wall_offset-5
    ion_positions[2] = ((rg2-rg1)*np.random.random(1)) + rg1   #rg1-rg2 desired (rg2-rg1)*a + rg1

    par = system.part.add(pos=ion_positions, type=type_HA  , q=charges[type_HA])
#    print ("Particle_id:  ",par, "  ; Particle Pos: ", par.pos, "  ; Particle Type: ", par.type,"  ; Particle Charge: ", par.q)


"""
part_positions = system.part.all().pos
for itr1 in part_positions:
    if(itr1[2]<wall_offset or itr1[2]>(box_l_z_tube-wall_offset)):
        print("!!!!!!!!!!!!!!Particle going outside in z-direction !!!!!!!", itr1)




#visualizer.screenshot(f'screenshot_2.png')

#If want to start a system with all ions both res and sys within the system then;
#particle_id=[]
#print(N_H,"  ",N_OH,"  ",N_Na,"  ",N_Cl,"  ",N_A,"  ",N_HA,"  ",N_H+N_OH+N_Na+N_Cl+N_A+N_HA)

#particle_id = []

particle_id = system.part.all().id
particle_type = system.part.all().type

sum_charge=0
for i in particle_id:

    p=system.part.by_id(i)
    sum_charge = sum_charge+p.q
#    print(p.type,"  ", p.q)
if(abs(sum_charge)>1e-3):
    raise ValueError("System is not neutral. Charge is "+ str(sum_charge))


if(abs(cNa_bulk+cH_bulk-cOH_bulk-cCl_bulk+cH2A_bulk-cA_bulk)>1e-14):
        raise RuntimeError("bulk is not electroneutral")

print('Bulk:  ','N_H-',(cH_bulk*Volume),  "  N_OH-",(cOH_bulk*Volume), "  N_Na-",(cNa_bulk*Volume), "  N_Cl-",(cCl_bulk*Volume), 'N_HA_zwit', (cHA_bulk*Volume), 'N_H2A_zwit', (cH2A_bulk*Volume), 'N_A_zwit', (cA_bulk*Volume))

print('System started with:  ','N_H-',N_H, "N_HA-", N_HA,  "  N_OH-",N_OH, "  N_Na-",N_Na, "  N_Cl-",N_Cl, "  N_A-",N_A, "N_HA_zwit", N_HA_zwit,"N_A_zwit",  N_A_zwit,"N_H2A_zwit",  N_H2A_zwit)

"""
#____________________________Set up electrostatic interactions______________________________

#                   (((((((    IDEAL CASE CHANGE   )))))))))))

p3m = electrostatics.P3M(prefactor=bjerrum*temperature, accuracy=1e-3)
elc = espressomd.electrostatics.ELC(
    actor=p3m, maxPWerror=1e-3, gap_size=elc_gap)
system.actors.add(elc)

#system.actors.add(p3m)
print("P3M parameters:\n")
p3m_params = p3m.get_params()
for key in list(p3m_params.keys()):
    print("{} = {}".format(key, p3m_params[key]))
"""

#============================================================================================

#============================SYSTEM WARMING AND OVERLAP REMOVES==============================

def remove_overlap(system, sd_params):
    system.integrator.set_steepest_descent(f_max=0,
                                        gamma=sd_params['damping'],
                                    max_displacement=sd_params['max_displacement'])
    system.integrator.run(0)
    maxforce = np.max(np.linalg.norm(system.part.all().f, axis=1))
    energy = system.analysis.energy()['total']

    i = 0
    while i < sd_params['max_steps'] // sd_params['emstep']:
        prev_maxforce = maxforce
        prev_energy = energy
        system.integrator.run(sd_params['emstep'])
        maxforce = np.max(np.linalg.norm(system.part.all().f, axis=1))
        relforce = np.abs((maxforce - prev_maxforce) / prev_maxforce)
        energy = system.analysis.energy()['total']
        relener = np.abs((energy - prev_energy) / prev_energy)
        if i > 1 and (i + 1) % 4 == 0:
            print(f"minimization step: {(i+1)*sd_params['emstep']:4.0f}"
                    f"    max. rel. force change:{relforce:+3.3e}"
                    f"    rel. energy change:{relener:+3.3e}")
        if relforce < sd_params['f_tol'] or relener < sd_params['e_tol']:
            break
        i += 1
    system.integrator.set_vv()



STEEPEST_DESCENT_PARAMS = {'f_tol': 1e-2,
                           'e_tol': 1e-5,
                           'damping': 30,
                            'max_steps': 30000,
                           'max_displacement': 0.0001,
                           'emstep': 50}

'''
**********************************************************
                  Umbrella Sampling 
**********************************************************
'''
z_min=0
z_max = 50
z_min_init=z_min
bin_size=0.5#1.0#(z_max-z_min)
num_of_bins=(int)((z_max-z_min)/bin_size)

k_harmonic=3.0

#For metafile
file_name_ls=[]
minimum_ls=[]
kspring_ls=[]

#===========================================Umbrella Sampling==========================================

K_har = k_harmonic
r_min=-5# THis sometimes causes error if taken too large like > 50   ...why , also too large like 30 was causing issue with missing peaks too
r_max=5
energy_width_r=0.2
N_points = (int)(((r_max-r_min)/energy_width_r)+1)
tabulated_force=[]
tabulated_energy=[]
for tab in range (N_points):
    r_p = r_min + (tab*energy_width_r)
    energy_rp = 0.5*K_har*((r_p-0)**2)
    force_rp = -K_har*(r_p-0)
    tabulated_energy.append(energy_rp)
    tabulated_force.append(force_rp)

#*************************************************
#       Setting up a raspberry  particle  
#*************************************************
 # Ion types RASPBERRY
    #############################################################
TYPE_CENTRAL = len(types)
TYPE_SURFACE = len(types)+1

# Interaction parameters (Lennard-Jones for raspberry)
radius_col = 3.
harmonic_radius = 3.0

# the subscript c is for colloid and s is for salt (also used for the surface beads)
eps_ss = 1.   # LJ epsilon between the colloid's surface particles.
sig_ss = 1.   # LJ sigma between the colloid's surface particles.
eps_cs = 48.  # LJ epsilon between the colloid's central particle and surface particles.
sig_cs = radius_col  # LJ sigma between the colloid's central particle and surface particles (colloid's radius).
a_eff = 0.32  # effective hydrodynamic radius of a bead due to the discreteness of LB.

# the LJ potential with the central bead keeps all the beads from simply collapsing into the center
system.non_bonded_inter[TYPE_SURFACE, TYPE_CENTRAL].wca.set_params(epsilon=eps_cs, sigma=sig_cs)
system.non_bonded_inter[type_HA_zwit, TYPE_CENTRAL].wca.set_params(epsilon=eps_cs, sigma=sig_cs)
system.non_bonded_inter[type_H2A_zwit, TYPE_CENTRAL].wca.set_params(epsilon=eps_cs, sigma=sig_cs)
system.non_bonded_inter[type_A_zwit, TYPE_CENTRAL].wca.set_params(epsilon=eps_cs, sigma=sig_cs)
# the LJ potential (WCA potential) between surface beads causes them to be roughly equidistant on the
# colloid surface
type_on_surface = [TYPE_SURFACE, type_HA_zwit, type_H2A_zwit, type_A_zwit]
for i in type_on_surface:
    for j in type_on_surface:
        system.non_bonded_inter[i, j].wca.set_params(epsilon=eps_ss, sigma=sig_ss)

# the harmonic potential pulls surface beads towards the central colloid bead
col_center_surface_bond = espressomd.interactions.HarmonicBond(k=3000., r_0=harmonic_radius)
system.bonded_inter.add(col_center_surface_bond)

center = system.box_l / 2
colPos = center

# Charge of the colloid
q_col = 0# IMPORTANT <<***********
# Number of particles making up the raspberry (surface particles + the central particle).
n_col_part = int(4 * np.pi * np.power(radius_col, 2) + 1)
print(f"Number of colloid beads = {n_col_part}")

# Place the central particle
central_part = system.part.add(pos=colPos, type=TYPE_CENTRAL, q=q_col,
                               fix=(True, True, True), rotation=(True, True, True))
probe_id = central_part.id

# Create surface beads uniformly distributed over the surface of the central particle
colSurfPos = np.random.uniform(low=-1, high=1, size=(n_col_part - 1, 3))
colSurfPos = colSurfPos / np.linalg.norm(colSurfPos, axis=1)[:, np.newaxis] * radius_col + colPos
#colSurfTypes = np.full(n_col_part - 1, TYPE_SURFACE)
Num_H2A_surface = 10
Num_A_surface = 10
Num_neutral = n_col_part - 1 - Num_H2A_surface - Num_A_surface

# Bonded interactions are between ids , while non-bonded are between types 
for i in range (len(colSurfPos)):
    if(i <Num_H2A_surface):
        surface_parts = system.part.add(pos=colSurfPos[i], type=type_H2A_zwit, q=z_H2A_zwit )
        surface_parts.add_bond((col_center_surface_bond, central_part))
    if(i>=Num_H2A_surface and i<(Num_H2A_surface+Num_A_surface)):
        surface_parts = system.part.add(pos=colSurfPos[i], type=type_A_zwit , q=z_A_zwit)
        surface_parts.add_bond((col_center_surface_bond, central_part))

    if(i>=(Num_H2A_surface+Num_A_surface)):
        surface_parts = system.part.add(pos=colSurfPos[i], type=TYPE_SURFACE , q=0)
        surface_parts.add_bond((col_center_surface_bond, central_part))

for i in types_without_polymer:# Goto line  704 to correct with  other ions too
    system.non_bonded_inter[i, TYPE_SURFACE].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)

for i in types_polymer:# Goto line  704 to correct with  other ions too
    system.non_bonded_inter[i, TYPE_SURFACE].lennard_jones.set_params(epsilon=LJ_EPSILON+1, sigma=LJ_SIGMA, cutoff=LJ_SIGMA * 2.5, shift="auto")


surface_parts = system.part.select(lambda p: p.type == type_H2A_zwit or  p.type == type_A_zwit or p.type == TYPE_SURFACE )
print(len(surface_parts))

#surface_parts = system.part.add(pos=colSurfPos, type=colSurfTypes)
char=[]
pos_x=[]
pos_y=[]
pos_z=[]
for p in surface_parts:
    if(p.type==type_H2A_zwit):
        char.append('a')
        pos_x.append(p.pos[0])
        pos_y.append(p.pos[1])
        pos_z.append(p.pos[2])
    if(p.type==type_A_zwit):
        char.append('b')
        pos_x.append(p.pos[0])
        pos_y.append(p.pos[1])
        pos_z.append(p.pos[2])
    if(p.type==TYPE_SURFACE):
        char.append('c')
        pos_x.append(p.pos[0])
        pos_y.append(p.pos[1])
        pos_z.append(p.pos[2])

np.savetxt("movie_init.xyz" , np.column_stack([char, pos_x, pos_y, pos_z]), fmt='%s\t', delimiter='\t')
print("System particle length: ", len(system.part.all()))

#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()

#visualizer.screenshot(f'screenshot_2.png')

system.integrator.set_steepest_descent(f_max=0, gamma=30, max_displacement=0.01 * sig_ss)

def constrain_surface_particles():
    # This loop moves the surface beads such that they are once again exactly radius_col
    # away from the center. For the scalar distance, we use system.distance() which
    # considers periodic boundaries and the minimum image convention
    colPos = central_part.pos
    for p in surface_parts:
#        print(p.pos)
        p.pos = (p.pos - colPos) / np.linalg.norm(system.distance(p, central_part)) * radius_col + colPos
        p.pos = (p.pos - colPos) / np.linalg.norm(p.pos - colPos) * radius_col + colPos

system.time_step = 0.005

print("Relaxation of the raspberry surface particles")
for j in range(1000):
    system.integrator.run(50)
    constrain_surface_particles()
    force_max = np.max(np.linalg.norm(system.part.all().f, axis=1))
#    print(f"maximal force: {force_max:.1f}")
    if force_max < 10.:
        break

system.time_step = 0.01
thermostat_seed = np.random.randint(np.random.randint(1000000))
system.thermostat.set_langevin(kT=temperature, gamma=1.0, seed=thermostat_seed)
system.integrator.set_vv()

print("Relaxation of the raspberry surface particles done ")

# (1 Working ) visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()


#visualizer.screenshot(f'screenshot_2.png')

# Select the desired implementation for virtual sites
system.virtual_sites = espressomd.virtual_sites.VirtualSitesRelative()

# Setting min_global_cut is necessary when there is no interaction defined with a range larger than
# the colloid such that the virtual particles are able to communicate their forces to the real particle
# at the center of the colloid
system.min_global_cut = radius_col#       ?????

# Calculate the center of mass position (com) and the moment of inertia (momI) of the colloid
com = np.average(surface_parts.pos, 0)  # surface_parts.pos returns an n-by-3 array
momI = 0
for p in surface_parts:
    momI += np.power(np.linalg.norm(com - p.pos), 2)


# note that the real particle must be at the center of mass of the colloid because of the integrator
print(f"Moving central particle from {central_part.pos} to {com}")
central_part.fix = [False, False, False]
central_part.pos = com
central_part.mass = n_col_part
central_part.rinertia = np.ones(3) * momI

# Convert the surface particles to virtual sites related to the central particle
# The id of the central particles is 0, the ids of the surface particles start at 1.
for p in surface_parts:
    p.vs_auto_relate_to(central_part)


#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()


'''

#*************************************************
#             Umbrella Sampling Module 
#*************************************************

#par_position=((box_l_x/2.0),(box_l_y/2.0), (box_l_z/2.0))
#par = system.part.add(pos=par_position , type=type_H2A_zwit  , q=z_H2A_zwit, ext_force=[0,0, 0])
#probe_id = par.id


#position=np.empty((3),dtype=float)
#rg1=wall_offset+5
#rg2=box_l_z_tube-wall_offset-5
#pos_z = ((rg2-rg1)*np.random.random()) + rg1   #rg1-rg2 desired (rg2-rg1)*a + rg1
#pos_x = np.random.random() * system.box_l[0]
#pos_y = np.random.random() * system.box_l[1]
#print("----------------------------> ", pos_x, pos_y )
#pos=(pos_x, pos_y, pos_z)
#counter = system.part.add(pos=pos, type=type_A_zwit  , q=z_A_zwit)

bin_id=20
#z_min=0
#bin_size=1.0
bin_equib_pt = (z_min + (bin_id*bin_size)) + (bin_size/2.0)
system.part.by_id(probe_id).pos=(box_l_x/2.0, box_l_y/2.0, bin_equib_pt-1)

#system.non_bonded_inter[type_HA, type_probe].lennard_jones.set_params(epsilon=LJ_EPSILON+2, sigma=LJ_SIGMA, cutoff=LJ_SIGMA * 2.5, shift="auto")
#system.non_bonded_inter[types_polymer, type_probe].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)
#system.non_bonded_inter[types_without_polymer, type_probe].wca.set_params(epsilon=LJ_EPSILON, sigma=LJ_SIGMA)
LJ_NP=1.0
#for i in range (len(types_polymer)):
#    system.non_bonded_inter[i,type_probe].lennard_jones.set_params(epsilon=LJ_EPSILON+1, sigma=(LJ_SIGMA+LJ_NP)*0.5, cutoff=(LJ_SIGMA+LJ_NP)*0.5 * 2.5, shift="auto")
#    system.non_bonded_inter[i, type_probe].wca.set_params(epsilon=LJ_EPSILON, sigma=(LJ_SIGMA+LJ_NP)*0.5)
#for i in range (len(types_without_polymer)):
#    system.non_bonded_inter[i, type_probe].wca.set_params(epsilon=LJ_EPSILON, sigma=(LJ_SIGMA+LJ_NP)*0.5)



Usampling_const_shape = espressomd.shapes.Wall(normal=[0, 0, 1], dist=bin_equib_pt)
Usampling_constraint = system.constraints.add(
particle_type=type_Usampling_wall, penetrable=True, only_positive=False, shape=Usampling_const_shape)

type_probe=[TYPE_CENTRAL]#type_HA_zwit, type_A_zwit, type_H2A_zwit]
for k in type_probe: 
    system.non_bonded_inter[k, type_Usampling_wall].tabulated.set_params(
    min=r_min, max=r_max, energy=tabulated_energy, force=tabulated_force)


#**************************************************************
#              Umbrella Sampling Ends
#**************************************************************
'''
# ------> Decide whether to bring near the build polymer 
#------>print("Started energy minimization...\n")
#------>remove_overlap(system, STEEPEST_DESCENT_PARAMS)
#------->print("MD runs done until energy minimization converges after removing overlaps .")

#system.integrator.run(3000)
#print("Energy minimzation finished.\n")

#thermostat_seed = np.random.randint(np.random.randint(1000000))
#system.thermostat.set_langevin(kT=temperature, gamma=1.0, seed=thermostat_seed)
#system.integrator.set_vv()

#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()



#for ir in range (100):
#    system.integrator.run()
#    for jr in range (10):
#        print(jr, system.part.by_id(start_part_id_list[jr]).pos)


#visualizer.screenshot(f'screenshot_final.png')

"""
#======================================Save evolved PE brush structure =================================
particle_pos_1=[]
#particle_pos_floor=[]
#particle_pos_ceil=[]
type_1=[]
type_floor=[]
type_ceil=[]


for p in system.part.select(type=type_A):
    particle_pos_1.append(p.pos)
    type_1.append('a')
for p in range(wall_pr_in_x*wall_pr_in_y):
#    particle_pos_floor.append(p.pos)
#    print("Wall particle:", p.pos)
    type_floor.append('b')
for p in range(wall_pr_in_x*wall_pr_in_y):
#    particle_pos_ceil.append(p.pos)
    type_ceil.append('c')


#np.savetxt("movie_ls_evolved.out" , particle_pos_1, fmt='%.7f\t', delimiter='\t')
#print("!!!!!!!!!!!!Movie evolved is saved after 100 MD runs before the reactions are set")

with open("movie_ls_evolved.out", "a") as numpy_file1:
    np.savetxt(numpy_file1 , np.column_stack([type_1,particle_pos_1]), fmt='%.7s\t', delimiter='\t')
    np.savetxt(numpy_file1 , np.column_stack([type_floor,particle_pos_floor]), fmt='%.7s\t', delimiter='\t')
    np.savetxt(numpy_file1 , np.column_stack([type_ceil,particle_pos_ceil]), fmt='%.7s\t', delimiter='\t')

print("!!!!!!!!!!!!Movie evolved is saved after 100 MD runs before the reactions are set")
"""


#=======================================================================================================

#=============================================================================================
#                                REACTIONS
#=============================================================================================


# Originally \mu =kBT ln(c \lambda^3)+ \mu_{ex}
# Write in terms of a reference state cref
# \mu = kBT ln(c_ref \lmbda^3) + \mu_{ex}{ref} + kBT ln(c/c_ref) + \mu_{ex}-\mu{ex}{ref}
# (\mu-\mu_ref) = kBT ln(c/c_ref) + \mu_{ex}-\mu{ex}{ref}
# We \mu throughout the code is wrt the ref state: (\mu  = \mu-\mu_ref)
# and excess chemical potential also is :          \mu_{ex} = \mu_{ex}-\mu{ex}{ref}


reaction_seed = np.random.randint(np.random.randint(1000000))
RE = espressomd.reaction_methods.ReactionEnsemble(kT=temperature, exclusion_range=0.9, seed=reaction_seed)
RE.set_wall_constraints_in_z_direction(slab_start_z =wall_offset ,slab_end_z= box_l_z_tube-wall_offset )

RE.set_volume(volume = box_l_x*box_l_y*box_l_z_tube)

# (1) Coupling to the Reservoir
#        ==================
#        |()  -> Na+ + Cl- |  (1.1)
#        ==================
# Reaction coefficient:   KbT lnK_{NaCl} = summ_{\nu_i\mu_i}  
#                         K_{NaCl} = C_{Na+}*c_{Cl-}*exp(\beta\mu_{ex})
#                         exp taken on both sides: hence same stoichemetry gives product of concentrations and exp of both \mu_{Na+}, \mu{Cl-} terms too
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=cNa_bulk*cCl_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature), 
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_Na, type_Cl],
    product_coefficients=[+1, +1],
    default_charges={type_Na:+1, type_Cl:-1})

#!!!!!!!!!!!!!!!!!!!!!!!!!!1Ambiguous must ask David!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Why not Kw , instead of this 


# (1) Coupling to the Reservoir
#        ==================
#        |()  -> H+ + OH- |  (1.1)
#        ==================

RE.add_reaction(
    gamma=cH_bulk*cOH_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature),
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_H, type_OH],
    product_coefficients=[+1, +1],
    default_charges={type_H:+1, type_OH:-1})
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# (1) Coupling to the Reservoir
#        ==================
#        |()  -> Na+ + OH- |  (1.1)
#        ==================

RE.add_reaction(
    gamma=cNa_bulk*cOH_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature),
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_Na, type_OH],
    product_coefficients=[+1, +1],
    default_charges={type_Na:+1, type_OH:-1})

# (1) Coupling to the Reservoir
#        ==================
#        |()  -> H+ + Cl- |  (1.1)
#        ==================

RE.add_reaction(
    gamma=cH_bulk*cCl_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature),
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_H, type_Cl],
    product_coefficients=[+1, +1],
    default_charges={type_H:+1, type_Cl:-1})

"""
# (1) Coupling to the Reservoir
#        ==================
#        |()  -> H2A+ + A- |  (1.1)
#        ==================

RE.add_reaction(
    gamma=cH2A_bulk*cA_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature),
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_H2A_zwit, type_A_zwit],
    product_coefficients=[+1, +1],
    default_charges={type_H2A_zwit:+1, type_A_zwit:-1})



# (1) Coupling to the Reservoir
#        ==================
#        |()  -> H2A+ + Cl- |  (1.1)
#        ==================

RE.add_reaction(
    gamma=cH2A_bulk*cCl_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature),
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_H2A_zwit, type_Cl],
    product_coefficients=[+1, +1],
    default_charges={type_H2A_zwit:+1, type_Cl:-1})

# (1) Coupling to the Reservoir
#        ==================
#        |()  -> H2A+ + OH- |  (1.1)
#        ==================

RE.add_reaction(
    gamma=cH2A_bulk*cOH_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature),
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_H2A_zwit, type_OH],
    product_coefficients=[+1, +1],
    default_charges={type_H2A_zwit:+1, type_OH:-1})
# (1) Coupling to the Reservoir
#        ==================
#        |()  -> Na+ + A- |  (1.1)
#        ==================

RE.add_reaction(
    gamma=cNa_bulk*cA_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature),
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_Na, type_A_zwit],
    product_coefficients=[+1, +1],
    default_charges={type_Na:+1, type_A_zwit:-1})

# (1) Coupling to the Reservoir
#        ==================
#        |()  -> H+ + A- |  (1.1)
#        ==================

RE.add_reaction(
    gamma=cH_bulk*cA_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature),
    reactant_types=[],
    reactant_coefficients=[],
    product_types=[type_H, type_A_zwit],
    product_coefficients=[+1, +1],
    default_charges={type_H:+1, type_A_zwit:-1})


"""

#        ==================
#        |Cl-  -> OH- |  (1.2)
#        ==================
# Reaction coefficient:   KbT lnK_{Cl>OH-} = summ_{\nu_i\mu_i}
#                         K_{Cl>OH-} = C_{OH-}/c_{Cl-}
#                         exp taken on both sides: since opposite stoichemetric coefficients, there is div and exponential \mu terms cancelled out to 1
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=cOH_bulk/cCl_bulk, 
    reactant_types=[type_Cl],
    reactant_coefficients=[+1],
    product_types=[type_OH],
    product_coefficients=[+1],
    default_charges={type_Cl:-1, type_OH:-1})

#        ==================
#        |Na+  -> H+ |  (1.3)
#        ==================
# Reaction coefficient:   KbT lnK_{Na+>H+} = summ_{\nu_i\mu_i}  {where \mu={(\mu{id} + \mu{ex})-(\mu_cref)}}
#                         K_{Na+>H+} = C_{H+}/c_{Na+}
#                         exp taken on both sides: since opposite stoichemetric coefficients, there is div and exponential \mu terms cancelled out to 1 
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=cH_bulk/cNa_bulk,
    reactant_types=[type_Na],
    reactant_coefficients=[+1],
    product_types=[type_H],
    product_coefficients=[+1],
    default_charges={type_Na:+1, type_H:+1})

"""
#        ==================
#        |Cl-  -> A- |  (1.2)
#        ==================
# Reaction coefficient:   KbT lnK_{Cl>OH-} = summ_{\nu_i\mu_i}
#                         K_{Cl>OH-} = C_{OH-}/c_{Cl-}
#                         exp taken on both sides: since opposite stoichemetric coefficients, there is div and exponential \mu terms cancelled out to 1
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=cA_bulk/cCl_bulk,
    reactant_types=[type_Cl],
    reactant_coefficients=[+1],
    product_types=[type_A_zwit],
    product_coefficients=[+1],
    default_charges={type_Cl:-1, type_A_zwit:-1})

#        ==================
#        |OH-  -> A- |  (1.2)
#        ==================
# Reaction coefficient:   KbT lnK_{Cl>OH-} = summ_{\nu_i\mu_i}
#                         K_{Cl>OH-} = C_{OH-}/c_{Cl-}
#                         exp taken on both sides: since opposite stoichemetric coefficients, there is div and exponential \mu terms cancelled out to 1
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=cA_bulk/cOH_bulk,
    reactant_types=[type_OH],
    reactant_coefficients=[+1],
    product_types=[type_A_zwit],
    product_coefficients=[+1],
    default_charges={type_OH:-1, type_A_zwit:-1})



#        ==================
#        |Na+  -> H2A+ |  (1.3)
#        ==================
# Reaction coefficient:   KbT lnK_{Na+>H+} = summ_{\nu_i\mu_i}  {where \mu={(\mu{id} + \mu{ex})-(\mu_cref)}}
#                         K_{Na+>H+} = C_{H+}/c_{Na+}
#                         exp taken on both sides: since opposite stoichemetric coefficients, there is div and exponential \mu terms cancelled out to 1
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=cH2A_bulk/cNa_bulk,
    reactant_types=[type_Na],
    reactant_coefficients=[+1],
    product_types=[type_H2A_zwit],
    product_coefficients=[+1],
    default_charges={type_Na:+1, type_H2A_zwit:+1})

#        ==================
#        |H+  -> H2A+ |  (1.3)
#        ==================
# Reaction coefficient:   KbT lnK_{Na+>H+} = summ_{\nu_i\mu_i}  {where \mu={(\mu{id} + \mu{ex})-(\mu_cref)}}
#                         K_{Na+>H+} = C_{H+}/c_{Na+}
#                         exp taken on both sides: since opposite stoichemetric coefficients, there is div and exponential \mu terms cancelled out to 1
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=cH2A_bulk/cH_bulk,
    reactant_types=[type_H],
    reactant_coefficients=[+1],
    product_types=[type_H2A_zwit],
    product_coefficients=[+1],
    default_charges={type_H:+1, type_H2A_zwit:+1})

"""
#RE.set_non_interacting_type(len(types))


# 2. Reactions within the system
#        ==================
#        |HA  -> A- + H+ |  (2.1)
#        ==================
# Reaction coefficient:   kbT lnK_A = summ_{\nu_i\mu_i}  {where \mu={(\mu{id} + \mu{ex})-(\mu_cref)}}
#                         Kc = K_A(dimensionless)*(c_ref)^sum_{\nu}= K_A*c_ref is fixed: Kcideal_in_mol_per_l
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=Kcideal_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l,
    reactant_types=[type_HA],
    reactant_coefficients=[+1],
    product_types=[type_A,type_H],
    product_coefficients=[+1, +1],
    default_charges={type_HA:0, type_A:-1, type_H:+1})

#          HA -> A- + H+
#          H+ + OH- -> ()

#        ==================
#        |HA +OH- -> A-   |  (2.1)
#        ==================
# Reaction coefficient:   kbT lnK_A' = summ_{\nu_i\mu_i}  
#                          K_A' = (K_A/K_w)
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}} = c_ref^{-1}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

RE.add_reaction(
    gamma=(Kcideal_in_mol_per_l/Kw)*conversion_factor_from_1_per_sigma_3_to_mol_per_l,
    reactant_types=[type_HA, type_OH],
    reactant_coefficients=[+1, +1],
    product_types=[type_A],
    product_coefficients=[+1],
    default_charges={type_HA:0, type_OH:-1, type_A:-1})

#          HA + OH- -> A-
#           () -> Na+ + OH-

#        ==================
#        |HA -> A- + Na+   |  (2.1)
#        ==================
# Reaction coefficient:   kbT lnK_A" = summ_{\nu_i\mu_i}  
#                          K_A" = (K_A*k_NaOH/K_w)
#                         Since KnaoH unlike other K s is in units of (1/sigma3)^2. Make it dimensionless first before conversion
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}} = (c_ref^{summ_\nu_KA} * c_ref^{summ_\nu_KNaOH/ c_re                                                                                 f^{summ_\nu_Kw}})
#                        In our system we input in units of (c^{ref}_inmol/L / fac_from_1/Gamma^3_to_mol/L) ^ {sum_{\nu}}

k_NaOH = cNa_bulk*cOH_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature)
RE.add_reaction(
    gamma=(Kcideal_in_mol_per_l*k_NaOH/Kw)*conversion_factor_from_1_per_sigma_3_to_mol_per_l,
    reactant_types=[type_HA],
    reactant_coefficients=[+1],
    product_types=[type_A, type_Na],
    product_coefficients=[+1, +1],
    default_charges={type_HA:0, type_A:-1, type_Na:+1})
print(k_NaOH)

#===================ZWITTERION REACTIONS==============================================

#        ==================
#        |HA_z  -> A_z- + H+ |  (2.1)
#        ==================
# Reaction coefficient:   kbT lnK_A = summ_{\nu_i\mu_i}  {where \mu={(\mu{id} + \mu{ex})-(\mu_cref)}}
#                         Kc = K_A(dimensionless)*(c_ref)^sum_{\nu}= K_A*c_ref is fixed: Kcideal_in_mol_per_l
#                        Gamma = K of reaction in units of c_ref^{sum_{\nu}}
#                        In our system we input in units of (c^{ref}_inmol/L * fac_from_mol/L_to_1/Gamma^3) ^ {sum_{\nu}}

ka1_rel = ka
ka2_rel = kb
ka1_sim = ka1_rel * ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**1.0)
ka2_sim = ka2_rel * ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**-1.0)

K_in_units_ref_conc_power_sum_nu = ka1_rel * ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**1.0)
RE.add_reaction(
    gamma=K_in_units_ref_conc_power_sum_nu,
    reactant_types=[type_HA_zwit],
    reactant_coefficients=[+1],
    product_types=[type_A_zwit,type_H],
    product_coefficients=[+1, +1],
    default_charges={type_HA_zwit:0, type_A_zwit:-1, type_H:+1})

#        ==================
#        |H2A_z  -> A_z- + 2H+ |  (2.1)
#        ==================

K_in_units_ref_conc_power_sum_nu = (ka1_rel/ka2_rel) * ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**2.0)
RE.add_reaction(
    gamma=K_in_units_ref_conc_power_sum_nu,
    reactant_types=[type_H2A_zwit],
    reactant_coefficients=[+1],
    product_types=[type_A_zwit,type_H],
    product_coefficients=[+1, +2],
    default_charges={type_H2A_zwit:+1, type_A_zwit:-1, type_H:+1})

#        ==================
#        |H2A_z + 2OH- -> A_z- |  (2.1)
#        ==================
Kw_rel = Kw
K_in_units_ref_conc_power_sum_nu  = (ka1_rel/ka2_rel) *(1/(Kw_rel*Kw_rel))* ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**-2.0)

RE.add_reaction(
    gamma=K_in_units_ref_conc_power_sum_nu,
    reactant_types=[type_H2A_zwit, type_OH],
    reactant_coefficients=[+1, +2],
    product_types=[type_A_zwit],
    product_coefficients=[+1],
    default_charges={type_H2A_zwit:+1, type_OH:-1, type_A_zwit:-1})


#        ==================
#        |H2A_z --> A_z- +  2Na+ |  (2.1)
#        ==================

K_NaOH_rel = k_NaOH / ((cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**2.0)# k_NaOH is in dimension of (sigma_3)^2
K_in_units_ref_conc_power_sum_nu  = (ka1_rel/ka2_rel) *(1.0/(Kw_rel*Kw_rel))* (K_NaOH_rel*K_NaOH_rel)* ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**2.0)

RE.add_reaction(
    gamma=K_in_units_ref_conc_power_sum_nu,
    reactant_types=[type_H2A_zwit],
    reactant_coefficients=[+1],
    product_types=[type_A_zwit, type_Na],
    product_coefficients=[+1, +2],
    default_charges={type_H2A_zwit:+1, type_Na:+1, type_A_zwit:-1})


#        ==================
#        |HA_z + H+ --> H2A_z |  (2.1)
#        ==================

K_in_units_ref_conc_power_sum_nu  = ka2_rel * ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**-1.0) 
RE.add_reaction(
    gamma=K_in_units_ref_conc_power_sum_nu,
    reactant_types=[type_HA_zwit, type_H],
    reactant_coefficients=[+1, +1],
    product_types=[type_H2A_zwit],
    product_coefficients=[+1],
    default_charges={type_HA_zwit:0, type_H:+1, type_H2A_zwit:+1})

#        ==================
#        |HA_z --> H2A_z + OH- |  (2.1)
#        ==================

K_in_units_ref_conc_power_sum_nu  = (ka2_rel*Kw_rel) * ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**1.0)         
RE.add_reaction(
    gamma=K_in_units_ref_conc_power_sum_nu,
    reactant_types=[type_HA_zwit],
    reactant_coefficients=[+1],
    product_types=[type_H2A_zwit, type_OH],
    product_coefficients=[+1, +1],
    default_charges={type_HA_zwit:0, type_H2A_zwit:+1, type_OH:-1})

#        ==================
#        |HA_z +Na+ --> H2A_z  |  (2.1)
#        ==================


K_NaOH_rel = k_NaOH / ((cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**2.0)
K_in_units_ref_conc_power_sum_nu  = (ka2_rel*Kw_rel/K_NaOH_rel) * ( (cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l)**-1.0) 
RE.add_reaction(
    gamma=K_in_units_ref_conc_power_sum_nu,
    reactant_types=[type_HA_zwit, type_Na],
    reactant_coefficients=[+1, +1],
    product_types=[type_H2A_zwit],
    product_coefficients=[+1],
    default_charges={type_HA_zwit:0, type_Na:+1, type_H2A_zwit:+1})








#https://espressomd.github.io/doc/espressomd.html#espressomd.reaction_methods.ReactionAlgorithm.set_volume
RE.set_wall_constraints_in_z_direction(slab_start_z =wall_offset ,slab_end_z= box_l_z_tube-wall_offset )

RE.set_volume(volume = box_l_x*box_l_y*box_l_z_tube)



#print(RE.get_status())
print("Excecution Stage@  REACTIONS ARE  DEFINED  ...." )
#RE.set_non_interacting_type(len(types))
def reaction(steps):
    global type_HA, type_A, N0
    RE.reaction(reaction_steps=steps)
    for a in range(int(N0/10.0)):
        MC_swap_A_HA_particles(type_HA, type_A)


def check_particle_crossing():
    part_positions = system.part.all().pos
    part_ids = system.part.all().id
    part_type = system.part.all().type
    for itr1, itr2 in zip(part_positions, part_type):
        if(itr1[2]<wall_offset or itr1[2]>(box_l_z_tube-wall_offset)):
            raise RuntimeError("!!!!!!!!!!!!!!Particle going outside the specified boundary in z-direction !!!!!!!", itr1, itr2)

RE.reaction(reaction_steps=100)
# (Stage 3 checked) visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()




print("100 times reactions done before warmup")

#visualizer.screenshot(f'screenshot_3.png')

char=[]
pos_x=[]
pos_y=[]
pos_z=[]
allparticles = system.part.all()
for p in allparticles:
    if(p.type==type_H2A_zwit):
        char.append('a')
        pos_x.append(p.pos[0])
        pos_y.append(p.pos[1])
        pos_z.append(p.pos[2])
    if(p.type==type_A_zwit):
        char.append('b')
        pos_x.append(p.pos[0])
        pos_y.append(p.pos[1])
        pos_z.append(p.pos[2])
    if(p.type==TYPE_SURFACE):
        char.append('c')
        pos_x.append(p.pos[0])
        pos_y.append(p.pos[1])
        pos_z.append(p.pos[2])
    else:
        char.append('d')
        pos_x.append(p.pos[0])
        pos_y.append(p.pos[1])
        pos_z.append(p.pos[2])

np.savetxt("movie_final.xyz" , np.column_stack([char, pos_x, pos_y, pos_z]), fmt='%s\t', delimiter='\t')
#____________________________Set up electrostatic interactions______________________________

#                   (((((((    IDEAL CASE CHANGE   )))))))))))

p3m = electrostatics.P3M(prefactor=bjerrum*temperature, accuracy=1e-3)
elc = espressomd.electrostatics.ELC(
    actor=p3m, maxPWerror=1e-3, gap_size=elc_gap)
system.actors.add(elc)

#system.actors.add(p3m)
print("P3M parameters:\n")
p3m_params = p3m.get_params()
for key in list(p3m_params.keys()):
    print("{} = {}".format(key, p3m_params[key]))

system.force_cap=0
system.time_step=TIME_STEPS
num_sample_warmup=100
num_samples = 2000
reaction_runs=50#int((c_poly*Volume))
integrator_runs=100

file_positions="Ions_pH_"+str(pH)+"_run_1"
# Warmup (Include electrostatics)
#############################################################
# warmup integration (with capped LJ potential)

# Warmup Integration Loop
i = 0
while (i < num_sample_warmup ):
#    print(i)
    RE.reaction(reaction_steps=reaction_runs)
    system.integrator.run(steps=integrator_runs)
    i += 1


ion_itr=[]
ion_id=[]
ion_type=[]
ion_pos_x=[]
ion_pos_y=[]
ion_pos_z=[]

# Warmup Integration Loop
i = 0
while (i < num_samples ):

    RE.reaction(reaction_steps=reaction_runs)
    system.integrator.run(steps=integrator_runs)
    if(i%5==0):
        slice_tot_pr=system.part.all()
        for prt in slice_tot_pr:
            if(prt.type!=type_constraint_A and prt.type!=type_constraint_B and prt.type!=type_HA and prt.type!=type_A):
                ion_itr.append(i)
                ion_id.append(prt.id)
                ion_type.append(prt.type)
                ion_pos_x.append(prt.pos[0])
                ion_pos_y.append(prt.pos[1])
                ion_pos_z.append(prt.pos[2])


    if(i%200==0):
        
        np.savetxt(file_positions+str(".out"), np.column_stack([ion_itr, ion_id, ion_type, ion_pos_x, ion_pos_y, ion_pos_z]), fmt='%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t', delimiter='\t')

    i += 1


casper[10]  =  caaper_terminate[10][20]+1
#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()







print("Warm up started .\n")
num_samples = 10000000
reaction_runs=50#int((c_poly*Volume))
integrator_runs=100
print('num_samples',num_samples, 'reaction_runs',reaction_runs, 'integrator_runs',integrator_runs,'\n')


# Warmup (Neutral brush )
#############################################################
# warmup integration (with capped LJ potential)
system.time_step=0.0001
warm_n_times = 1000#20
lj_cap = 2
system.force_cap=lj_cap

# Warmup Integration Loop
act_min_dist = system.analysis.min_dist()
i = 0
#start_warmup = time.time()
while (i < warm_n_times ):
#    print(i, "warmup")
#    RE.reaction(reaction_steps=reaction_runs)
#    system.integrator.run(steps=integrator_runs)
    system.integrator.run(1)
    i += 1
    lj_cap = lj_cap + 1
    system.force_cap=lj_cap

#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()

"""
#*****************************************************************
bins=500
CountInBin=np.zeros((bins))
TotalCounts = 0
#ListofBins=[]
#ListofParticles=[]

for integrator_stp in range (100000):
    system.integrator.run(1)
    for p in system.part:
        if(p.type!=type_constraint_A and p.type!=type_constraint_B):
            bin_id = int(p.pos[2]/0.1)
            CountInBin[bin_id] += 1
            TotalCounts +=1
    ListofBins=[]
    ListofParticles=[]

    for b in range(bins):
        ListofBins.append(1.0 + (0.1*b))
        ListofParticles.append(CountInBin[b]/TotalCounts)

np.savetxt("distribution.out", np.column_stack([ListofBins, ListofParticles]), fmt='%.7f\t%.7f\t', delimiter='\t')
#****************************************************************
"""



print("Warmup without ekectrostatics for neutral brush ends successfully!!!!")
RE.reaction(reaction_steps=(5*reaction_runs))

print("Warmup with reaction calls to introduce some ionization and ions successful!!!!")


#____________________________Set up electrostatic interactions______________________________

#                   (((((((    IDEAL CASE CHANGE   )))))))))))

p3m = electrostatics.P3M(prefactor=bjerrum*temperature, accuracy=1e-3)
elc = espressomd.electrostatics.ELC(
    actor=p3m, maxPWerror=1e-3, gap_size=elc_gap)
system.actors.add(elc)

#system.actors.add(p3m)
print("P3M parameters:\n")
p3m_params = p3m.get_params()
for key in list(p3m_params.keys()):
    print("{} = {}".format(key, p3m_params[key]))

#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()

# Warmup (Include electrostatics)
#############################################################
# warmup integration (with capped LJ potential)
system.time_step=0.0001
warm_steps = 10
warm_n_times = 1000#20
lj_cap = 2
system.force_cap=lj_cap

# Warmup Integration Loop
act_min_dist = system.analysis.min_dist()
i = 0
start_warmup = time.time()
while (i < warm_n_times ):
#    print(i, "warmup with electrostatics")
    RE.reaction(reaction_steps=reaction_runs)
    system.integrator.run(steps=integrator_runs)
    check_particle_crossing()
    i += 1
    lj_cap = lj_cap + 1
    system.force_cap=lj_cap

print("Warmup with electrostatics for charged system ends successfully (Force cap on)!!!!")
#end_warmup=time.time()
#elapsed_time_in_minutes = (end_warmup - start_warmup)/60.0 #in seconds
#time_per_cyle=elapsed_time_in_minutes/warm_n_times #in minutes/cycle
#nr_of_cylces_in_5_minutes=int((5.-5)/time_per_cyle) #minus 5 minute to make sure the last checkpoint is written before the time slot on bee ends
#print("nr_of_cylces_in_5_minutes", nr_of_cylces_in_5_minutes)

############################################################
# remove force capping
###########################################################

system.force_cap=0
system.time_step=TIME_STEPS


# Warmup (Include electrostatics)
#############################################################
# warmup integration (with capped LJ potential)
warm_n_times = 200#20

# Warmup Integration Loop
i = 0
while (i < warm_n_times ):
#    print(i, "warmup with electrostatics")
    RE.reaction(reaction_steps=reaction_runs)
    system.integrator.run(steps=integrator_runs)
    i += 1


print("Equiliberation done successfully!!!!")

'''
**********************************************************
                  Umbrella Sampling 
**********************************************************

#z_min=0
z_max = 50
z_min_init=z_min
#bin_size=1.0#1.0#(z_max-z_min)
num_of_bins=(int)((z_max-z_min)/bin_size)

k_harmonic=5.0

#For metafile
file_name_ls=[]
minimum_ls=[]
kspring_ls=[]

#===========================================Umbrella Sampling==========================================

K_har = k_harmonic
r_min=-5# THis sometimes causes error if taken too large like > 50   ...why , also too large like 30 was causing issue with missing peaks too
r_max=5
energy_width_r=0.2
N_points = (int)(((r_max-r_min)/energy_width_r)+1)
tabulated_force=[]
tabulated_energy=[]
for tab in range (N_points):
    r_p = r_min + (tab*energy_width_r)
    energy_rp = 0.5*K_har*((r_p-0)**2)
    force_rp = -K_har*(r_p-0)
    tabulated_energy.append(energy_rp)
    tabulated_force.append(force_rp)

#======================================================================================================

#Usampling_const_shape = espressomd.shapes.Wall(normal=[0, 0, 1], dist=box_l_z_tube/2.0)
#Usampling_constraint = system.constraints.add(
#particle_type=type_Usampling_wall, penetrable=True, only_positive=False, shape=Usampling_const_shape)

#par_position=((box_l_x/2.0),(box_l_y/2.0), (box_l_z/2.0))
#par = system.part.add(pos=par_position , type=type_probe  , q=z_probe, ext_force=[0,0, 0])
#probe_id = par.id
#par1=system.part.add(pos=np.random.random(3) * (box_l_x, box_l_y, 10), type=type_probe_2, q=-z_probe, ext_force=[0,0,0])
'''
#bin_id=13
#num_of_bins-1
bin_copy=bin_id
#for bin_id in  range (num_of_bins):
while (bin_id==bin_copy):
    bin_equib_pt = (z_min + (bin_id*bin_size)) + (bin_size/2.0)#z_min + (bin_size/2.0)

#    system.constraints.remove(Usampling_constraint)

#    Usampling_const_shape = espressomd.shapes.Wall(normal=[0, 0, 1], dist=bin_equib_pt)
#    Usampling_constraint = system.constraints.add(
#    particle_type=type_Usampling_wall, penetrable=True, only_positive=False, shape=Usampling_const_shape)

#    system.non_bonded_inter[type_probe, type_Usampling_wall].tabulated.set_params(
#    min=r_min, max=r_max, energy=tabulated_energy, force=tabulated_force)

#    system.part.select(type=type_probe).remove()
#    system.part.by_id(probe_id).pos=(box_l_x/2.0, box_l_y/2.0, bin_equib_pt-1)

#    system.non_bonded_inter[type_HA, type_probe].lennard_jones.set_params(epsilon=LJ_EPSILON+1, sigma=LJ_SIGMA, cutoff=LJ_SIGMA * 2.5, shift="auto")
#    print("Started energy minimization...\n")
#    remove_overlap(system, STEEPEST_DESCENT_PARAMS)
#    print("MD runs done until energy minimization converges after removing overlaps .")



#    thermostat_seed = np.random.randint(np.random.randint(1000000))
#    system.thermostat.set_langevin(kT=1.0, gamma=1.0, seed=thermostat_seed)
#    system.integrator.set_vv()


    min_range = 10000000000000000
    max_range = -10000000000000000
    accumulate_in_bin = np.empty((10000000), dtype=float)
    tot_trials_in_bin = np.empty((10000000), dtype=float)

    time_ls=[]
    position_ls=[]
    type_ls=[]
    num_samples = 2000
#    print("bin_id",  bin_id)
    sum_charge = 0
    counter_charge = 0
    charge_probe=[]
    for i in range(num_samples):
        RE.reaction(reaction_steps=reaction_runs)
        system.integrator.run(100)

        dist_rr=system.part.by_id(probe_id).pos
        position_ls.append(dist_rr[2])
        type_ls.append(dist_rr[2]-bin_equib_pt)
        time_ls.append(i)

        charge_probe.append(system.part.by_id(probe_id).q)
        sum_charge = sum_charge  + system.part.by_id(probe_id).q
        counter_charge += 1;

        if(i%10==0):
            print( sum_charge/counter_charge )



    
    with open(f"position_info_{bin_id}.dat", "a") as write_to_dist_file:
        np.savetxt(write_to_dist_file , np.column_stack([time_ls,position_ls]), fmt='%.7s\t', delimiter='\t', newline='\n')
    with open(f"rel_position_info_{bin_id}.dat", "a") as write_to_dist_file_1:
        np.savetxt(write_to_dist_file_1 , np.column_stack([time_ls,type_ls]), fmt='%.7s\t', delimiter='\t', newline='\n')
    with open(f"charge_info_{bin_id}.dat", "a") as write_to_dist_file_2:
        np.savetxt(write_to_dist_file_2 , np.column_stack([time_ls,charge_probe]), fmt='%.7s\t', delimiter='\t', newline='\n')



    string_here="position_info_"+str(bin_id)+".dat"
    file_name_ls.append(string_here)
#    "{} {}".format(str(value),mylist2[index])
    minimum_ls.append(bin_equib_pt)
#    print(bin_equib_pt)
    kspring_ls.append(K_har)
    with open("metadata.out", "a") as write_to_meta_file:
        np.savetxt(write_to_meta_file , np.column_stack([file_name_ls,minimum_ls,kspring_ls]), fmt='%.30s\t', delimiter='\t', newline='\n')


    bin_id = bin_id-1

'''
**********************************************************
                  Umbrella Sampling Finishes 
**********************************************************
'''



#visualizer.screenshot(f'screenshot_4.png')
#visualizer = espressomd.visualization.openGLLive(system,particle_coloring='type')
#visualizer.run()
##################################### ( Particle Setup ) ##########################################

################################## ( Thermostat ) ################################################
#thermostat_seed = np.random.randint(np.random.randint(1000000))
#system.thermostat.set_langevin(kT=temperature, gamma=1.0, seed=thermostat_seed)
#system.integrator.set_vv()

#system.integrator.run(steps=2000)
#print("Ran 2000 integration steps")
'''
def chain_size():
    print(len(Num_polymer_ls_seq), Num_polymers_in_brush)
    if(len(Num_polymer_ls_seq)!= Num_polymers_in_brush):
        raise RuntimeError(" Generated list of polydisperse polymer chains is not similar to the specified one ")
#id_start  = i*Num_polymer_ls_seq[i]
#id_end    = ((i+1)*(Num_polymer_ls_seq[i]))-1
    id_prev = -1

    mean_end_to_end_distance=0
    counter_end_to_end_distance = 0
    for i in range (len(Num_polymer_ls_seq)):
        
        id_start = id_prev+1
        id_end = id_prev + int(Num_polymer_ls_seq[i])
        id_prev = id_end
        print(f"For chain {i+1} with {Num_polymer_ls_seq[i]} monomers: {id_start} ;   {id_end}")
        position_first_monomer = system.part.by_id(track_brush_particle_id[id_start]).pos
        position_end_monomer = system.part.by_id(track_brush_particle_id[id_end]).pos

        
        end_to_end_dist_per_chain = np.sqrt(((position_end_monomer[0]-position_first_monomer[0])**2) + ((position_end_monomer[1]-position_first_monomer[1])**2) + ((position_end_monomer[2]-position_first_monomer[2])**2))    
        mean_end_to_end_distance =  mean_end_to_end_distance + end_to_end_dist_per_chain
        counter_end_to_end_distance = counter_end_to_end_distance +1

    mean_rg_polymer_brush = 0
    counter_rg_polymer_brush = 0

    id_prev = -1
    for i in range (len(Num_polymer_ls_seq)):
        mean_radius_of_gyration = 0
        id_start = id_prev+1
        id_end = id_prev + int(Num_polymer_ls_seq[i])
        id_prev = id_end

        for j in range (id_start, (id_end+1)):

            for k in range ((j+1), (id_end+1)):

                id_pr1  = j
                id_pr2  = k
                pos_r1 = system.part.by_id(track_brush_particle_id[id_pr1]).pos
                pos_r2 = system.part.by_id(track_brush_particle_id[id_pr2]).pos

                radius_of_gyration = ((pos_r2[0]-pos_r1[0])**2) + ((pos_r2[1]-pos_r1[1])**2) + ((pos_r2[2]-pos_r1[2])**2)
                mean_radius_of_gyration = mean_radius_of_gyration  + radius_of_gyration

        mean_radius_of_gyration = (mean_radius_of_gyration/(Num_polymer_ls_seq[i]**2))#Rg-of_a chain = 1/2N2 SIGMA(|ri-rj|^2)
        mean_rg_polymer_brush = mean_rg_polymer_brush + mean_radius_of_gyration
        counter_rg_polymer_brush = counter_rg_polymer_brush + 1
        

    return ((mean_end_to_end_distance/counter_end_to_end_distance), (mean_rg_polymer_brush/counter_rg_polymer_brush))

Re_Rg = chain_size( )

def calc_error(observable):
    N_itr = len(observable)
    mean = 0
    error = 0
    for i in observable:
        mean = mean + i
    for i in observable:
        error = error + ((i-mean)**2)

    if(N_itr==1):
        return_mean = 0
        return_error = 0
    else:
        return_mean = (mean/N_itr)
        return_error = (error/(N_itr*(N_itr-1)))**0.5
    return (return_mean, return_error)


polymer_ionization=[]
zwitterion_uptake_total=[]
zwitterion_uptake_total_into_brush=[]
chain_end_to_end_distance=[]
chain_radius_of_gyration=[]
#zwitterion_uptake_frac=[]
#zwitterion_ionization=[]
#zwitterion_effective_charge=[]
#total_H_system=[]
#total_OH_system=[]
#total_Na_system=[]
#total_Cl_system=[]
#total_A_system=[]
#total_A_zwit_system=[]
#total_HA_zwit_system=[]
#total_H2A_zwit_system=[]

eta_H=[]
eta_OH=[]
eta_Na=[]
eta_Cl=[]
eta_A_zwit=[]
eta_HA_zwit=[]
eta_H2A_zwit=[]
eta_cation=[]
eta_anion=[]

eta_H_brush=[]
eta_OH_brush=[]
eta_Na_brush=[]
eta_Cl_brush=[]
eta_A_zwit_brush=[]
eta_HA_zwit_brush=[]
eta_H2A_zwit_brush=[]
eta_cation_brush=[]
eta_anion_brush=[]



N_H= []
N_OH= []
N_Na= []
N_Cl= []
N_A=[]
N_HA=[]
N_HA_zwit=[]
N_H2A_zwit=[]
N_A_zwit=[]

N_H_brush= []
N_OH_brush= []
N_Na_brush= []
N_Cl_brush= []
N_A_brush=[]
N_HA_brush=[]
N_HA_zwit_brush=[]
N_H2A_zwit_brush=[]
N_A_zwit_brush=[]





cH_bulk_ls =[]
cOH_bulk_ls =[]
cNa_bulk_ls =[]
cCl_bulk_ls =[]
cHA_bulk_ls=[]
cH2A_bulk_ls=[]
cA_bulk_ls=[]

cH_bulk_ls.append(cH_bulk)
cOH_bulk_ls.append(cOH_bulk)
cNa_bulk_ls.append(cNa_bulk)
cCl_bulk_ls.append(cCl_bulk)
cHA_bulk_ls.append(cHA_bulk)
cH2A_bulk_ls.append(cH2A_bulk)
cA_bulk_ls.append(cA_bulk)


num_Hs= []
num_OHs= []

ion_itr=[]
ion_id=[]
ion_type=[]
ion_pos_x=[]
ion_pos_y=[]
ion_pos_z=[]

monomer_itr=[]
monomer_id=[]
monomer_type=[]
monomer_pos_x=[]
monomer_pos_y=[]
monomer_pos_z=[]


if(os.path.exists("checkpoint.pgz")):
    #try to load safed statistical data
    with gzip.GzipFile("checkpoint.pgz", 'rb') as fcheck:
        print("loading checkpoint")
        data = pickle.load(fcheck)
        num_Hs=data[0]
        num_OHs=data[1]


filename="Nions_sys_pH_"+str(pH)+"_run_1"
filename_inside_brush="Nions_within_brush_pH_"+str(pH)+"_run_1"
filename2="Cions_bulk_pH_"+str(pH)+"_run_1"
filename3="Mean_part_coeff_"+str(pH)+"_run_1"
filename4="Mean_qts_"+str(pH)+"_run_1"
file_positions="Ions_pH_"+str(pH)+"_run_1"
file_positions_2="brush_pH_"+str(pH)+"_run_1"

#file1 = open(filename+str(".dat"), "a")  # append mode
#file2 = open(filename2+str(".dat"), "a")  # append mode
#file3 = open(filename3+str(".dat"), "a")

#For a I_res i.e pH and c_A- 
for i in range(num_samples):

    RE.reaction(reaction_steps=reaction_runs)
    system.integrator.run(integrator_runs)


    Re_Rg = chain_size()
    chain_end_to_end_distance.append(Re_Rg[0])
    chain_radius_of_gyration.append(Re_Rg[1])

    num_A=system.number_of_particles(type=type_A)
    num_OH=system.number_of_particles(type=type_OH)
    num_H=system.number_of_particles(type=type_H)
    num_Cl=system.number_of_particles(type=type_Cl)
    num_Na=system.number_of_particles(type=type_Na)
    num_HA=system.number_of_particles(type=type_HA)
    num_H2A_zwit=system.number_of_particles(type=type_H2A_zwit)
    num_A_zwit=system.number_of_particles(type=type_A_zwit)
    num_HA_zwit=system.number_of_particles(type=type_HA_zwit)#    visualizer.update()

    N_A.append(num_A)
    N_OH.append(num_OH)
    N_H.append(num_H)
    N_Cl.append(num_Cl)
    N_Na.append(num_Na)
    N_HA.append(num_HA)
    N_H2A_zwit.append(num_H2A_zwit)
    N_A_zwit.append(num_A_zwit)
    N_HA_zwit.append(num_HA_zwit)

    num_in_brush = np.empty((len(types)), dtype=float)
    for i1 in types:
        num_in_brush[i1] = 0
        for p1 in system.part.select(type=i1):
            if(p1.pos[2] <= (Re_Rg[0]+LJ_SIGMA)):
                num_in_brush[i1] = num_in_brush[i1] +1

    N_A_brush.append(num_in_brush[type_A])
    N_OH_brush.append(num_in_brush[type_OH])
    N_H_brush.append(num_in_brush[type_H])
    N_Cl_brush.append(num_in_brush[type_Cl])
    N_Na_brush.append(num_in_brush[type_Na])
    N_HA_brush.append(num_in_brush[type_HA])
    N_H2A_zwit_brush.append(num_in_brush[type_H2A_zwit])
    N_A_zwit_brush.append(num_in_brush[type_A_zwit])
    N_HA_zwit_brush.append(num_in_brush[type_HA_zwit])

#    print(num_A, num_H, num_HA, num_OH, num_Na, num_Cl, num_HA_zwit, num_H2A_zwit, num_A_zwit)
#    print(num_A_brush, num_H_brush, num_HA_brush, num_OH_brush, num_Na_brush, num_Cl_brush, num_HA_zwit_brush, num_H2A_zwit_brush, num_A_zwit_brush)


    
    polymer_ionization.append(num_A/(num_A+num_HA))
    zwitterion_uptake_total.append(num_A_zwit+num_HA_zwit+num_H2A_zwit)
    zwitterion_uptake_total_into_brush.append(num_in_brush[type_A_zwit]+num_in_brush[type_HA_zwit]+num_in_brush[type_H2A_zwit])
#    zwitterion_uptake_frac.append((num_A_zwit+num_HA_zwit+num_H2A_zwit)/ round((cA_bulk+cHA_bulk+cH2A_bulk)*Volume))
#    if((num_A_zwit+num_H2A_zwit+num_HA_zwit)!=0):
#        zwitterion_ionization.append((num_A_zwit-num_H2A_zwit)/(num_A_zwit+num_H2A_zwit+num_HA_zwit))
#    zwitterion_effective_charge.append(num_H2A_zwit - num_A_zwit)
#    total_H_system.append(num_H)
#    total_OH_system.append(num_OH)
#    total_Na_system.append(num_Na)
#    total_Cl_system.append(num_Cl)
#    total_A_system.append(num_A)
#    total_A_zwit_system.append(num_A_zwit)
#    total_HA_zwit_system.append(num_HA_zwit)
#    total_H2A_zwit_system.append(num_H2A_zwit)
    eta_H.append((num_H/Volume) / cH_bulk)
    eta_OH.append((num_OH/Volume) / cOH_bulk)
    eta_Na.append((num_Na/Volume) / cNa_bulk)
    eta_Cl.append((num_Cl/Volume) / cCl_bulk)
    eta_A_zwit.append((num_A_zwit/Volume) / cA_bulk)
    eta_HA_zwit.append((num_HA_zwit/Volume) / cHA_bulk)
    eta_H2A_zwit.append((num_H2A_zwit/Volume) / cH2A_bulk)

    num_cation =  num_H + num_Na+ num_H2A_zwit
    num_anion = num_OH + num_Cl +num_A_zwit
    eta_cation.append((num_cation/Volume) / (cH2A_bulk+ cNa_bulk+cH_bulk))
    eta_anion.append((num_anion/Volume) / (cA_bulk+ cCl_bulk+cOH_bulk))

    eta_H_brush.append((num_in_brush[type_H]/Volume) / cH_bulk)
    eta_OH_brush.append((num_in_brush[type_OH]/Volume) / cOH_bulk)
    eta_Na_brush.append((num_in_brush[type_Na]/Volume) / cNa_bulk)
    eta_Cl_brush.append((num_in_brush[type_Cl]/Volume) / cCl_bulk)
    eta_A_zwit_brush.append((num_in_brush[type_A_zwit]/Volume) / cA_bulk)
    eta_HA_zwit_brush.append((num_in_brush[type_HA_zwit]/Volume) / cHA_bulk)
    eta_H2A_zwit_brush.append((num_in_brush[type_H2A_zwit]/Volume) / cH2A_bulk)
    
    num_cation =  num_in_brush[type_H] + num_in_brush[type_Na]+ num_in_brush[type_H2A_zwit]
    num_anion = num_in_brush[type_OH] + num_in_brush[type_Cl] + num_in_brush[type_A_zwit]
    eta_cation_brush.append((num_cation/Volume) / (cH2A_bulk+ cNa_bulk+cH_bulk))
    eta_anion_brush.append((num_anion/Volume) / (cA_bulk+ cCl_bulk+cOH_bulk))
    
# particle_id , x, y, z    
    if(i%25==0):
        slice_tot_pr=system.part.all()
        for prt in slice_tot_pr: 
            if(prt.type!=type_constraint_A and prt.type!=type_constraint_B and prt.type!=type_HA and prt.type!=type_A):
                ion_itr.append(i)
                ion_id.append(prt.id)
                ion_type.append(prt.type)
                ion_pos_x.append(prt.pos[0])
                ion_pos_y.append(prt.pos[1])
                ion_pos_z.append(prt.pos[2])
       
        for id in range(len(track_brush_particle_id)): 
#        id_start  = Np*Num_beads_each_polymer
#        id_end    = ((Np+1)*(Num_beads_each_polymer))-1
            position_monomer = system.part.by_id(track_brush_particle_id[id]).pos
            id_monomer = system.part.by_id(track_brush_particle_id[id]).id
            type_monomer = system.part.by_id(track_brush_particle_id[id]).type
            monomer_itr.append(i)
            monomer_id.append(id_monomer)
            monomer_type.append(type_monomer)
            monomer_pos_x.append(position_monomer[0])
            monomer_pos_y.append(position_monomer[1])
            monomer_pos_z.append(position_monomer[2])
#        position_end_monomer = system.part.by_id(track_brush_particle_id[id_end]).pos
 
    N_steps=len(N_A)
#    if(i%nr_of_cylces_in_5_minutes==0):
    if(i%500==0):
        
        np.savetxt(file_positions+str(".out"), np.column_stack([ion_itr, ion_id, ion_type, ion_pos_x, ion_pos_y, ion_pos_z]), fmt='%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t', delimiter='\t')

        np.savetxt(file_positions_2+str(".out"), np.column_stack([monomer_itr, monomer_id, monomer_type, monomer_pos_x, monomer_pos_y, monomer_pos_z]), fmt='%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t', delimiter='\t')

        np.savetxt(filename+str(".out"), np.column_stack([N_A, N_H, N_HA, N_OH, N_Na, N_Cl, N_HA_zwit, N_H2A_zwit, N_A_zwit]), fmt='%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t', delimiter='\t', header="N_A, N_H, N_HA, N_OH, N_Na, N_Cl, N_HA_zwit, N_H2A_zwit, N_A_zwit")

        np.savetxt(filename_inside_brush+str(".out"), np.column_stack([N_A_brush, N_H_brush, N_HA_brush, N_OH_brush, N_Na_brush, N_Cl_brush, N_HA_zwit_brush, N_H2A_zwit_brush, N_A_zwit_brush]), fmt='%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t', delimiter='\t', header="N_A, N_H, N_HA, N_OH, N_Na, N_Cl, N_HA_zwit, N_H2A_zwit, N_A_zwit")
#        np.savetxt(filename+str(".out"), np.column_stack([N_A, N_H, N_HA, N_OH, N_Na, N_Cl]), fmt='%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t', delimiter='\t', header="N_A, N_H, N_HA, N_OH, N_Na, N_Cl")
        np.savetxt(filename2+str(".out"), np.column_stack([cH_bulk_ls, cOH_bulk_ls, cNa_bulk_ls, cCl_bulk_ls, cHA_bulk_ls, cH2A_bulk_ls, cA_bulk_ls]), fmt='%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t', delimiter='\t')

#        np.savetxt(filename2+str(".out"), np.column_stack([cH_bulk_ls, cOH_bulk_ls, cNa_bulk_ls, cCl_bulk_ls, cHA_bulk_ls, cH2A_bulk_ls, cA_bulk_ls]), fmt='%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t%.7f\t', delimiter='\t')


        mean_polymer_ionization, error_polymer_ionization = calc_error(polymer_ionization)
        mean_zwitterion_uptake_total, error_zwitterion_uptake_total = calc_error(zwitterion_uptake_total)
        mean_zwitterion_uptake_total_into_brush, error_zwitterion_uptake_total_into_brush = calc_error(zwitterion_uptake_total_into_brush)
        mean_chain_end_to_end_distance, error_chain_end_to_end_distance = calc_error(chain_end_to_end_distance)
        mean_chain_radius_of_gyration, error_chain_radius_of_gyration = calc_error(chain_radius_of_gyration)
#        mean_zwitterion_uptake_frac, error_zwitterion_uptake_frac = calc_error(zwitterion_uptake_frac)
#        mean_zwitterion_ionization, error_zwitterion_ionization = calc_error(zwitterion_ionization)
#        mean_zwitterion_effective_charge, error_zwitterion_effective_charge = calc_error(zwitterion_effective_charge)

#        mean_total_H_system, error_total_H_system = calc_error(total_H_system)
#        mean_total_OH_system, error_total_OH_system = calc_error(total_OH_system)
#        mean_total_Na_system, error_total_Na_system = calc_error(total_Na_system)
#        mean_total_Cl_system, error_total_Cl_system = calc_error(total_Cl_system)
#        mean_total_A_system, error_total_A_system = calc_error(total_A_system)
#        mean_total_A_zwit_system, error_total_A_zwit_system = calc_error(total_A_zwit_system)
#        mean_total_HA_zwit_system, error_total_HA_zwit_system = calc_error(total_HA_zwit_system)
#        mean_total_H2A_zwit_system, error_total_H2A_zwit_system = calc_error(total_H2A_zwit_system)
  






        mean_eta_H, error_eta_H = calc_error(eta_H)
        mean_eta_OH, error_eta_OH = calc_error(eta_OH)
        mean_eta_Na, error_eta_Na = calc_error(eta_Na)
        mean_eta_Cl, error_eta_Cl = calc_error(eta_Cl)
        mean_eta_A_zwit, error_eta_A_zwit = calc_error(eta_A_zwit)
        mean_eta_HA_zwit, error_eta_HA_zwit = calc_error(eta_HA_zwit)
        mean_eta_H2A_zwit, error_eta_H2A_zwit = calc_error(eta_H2A_zwit)
        mean_eta_cation, error_eta_cation = calc_error(eta_cation)
        mean_eta_anion, error_eta_anion = calc_error(eta_anion)

        mean_eta_H_brush, error_eta_H_brush = calc_error(eta_H_brush)
        mean_eta_OH_brush, error_eta_OH_brush = calc_error(eta_OH_brush)
        mean_eta_Na_brush, error_eta_Na_brush = calc_error(eta_Na_brush)
        mean_eta_Cl_brush, error_eta_Cl_brush = calc_error(eta_Cl_brush)
        mean_eta_A_zwit_brush, error_eta_A_zwit_brush = calc_error(eta_A_zwit_brush)
        mean_eta_HA_zwit_brush, error_eta_HA_zwit_brush = calc_error(eta_HA_zwit_brush)
        mean_eta_H2A_zwit_brush, error_eta_H2A_zwit_brush = calc_error(eta_H2A_zwit_brush)
        mean_eta_cation_brush, error_eta_cation_brush = calc_error(eta_cation_brush)
        mean_eta_anion_brush, error_eta_anion_brush = calc_error(eta_anion_brush)


#        OUTPUT_LIST_1 = [mean_total_H_system, mean_total_OH_system,   mean_total_Na_system , mean_total_Cl_system , mean_total_A_system]

#        OUTPUT_LIST_1 = [mean_total_H_system, error_total_H_system , mean_total_OH_system, error_total_OH_system , mean_total_Na_system, error_total_Na_system , mean_total_Cl_system, error_total_Cl_system , mean_total_A_system, error_total_A_system , mean_total_A_zwit_system, error_total_A_zwit_system , mean_total_HA_zwit_system, error_total_HA_zwit_system, mean_total_H2A_zwit_system, error_total_H2A_zwit_system ]

        OUTPUT_LIST_1 = [mean_eta_H, error_eta_H,
        mean_eta_OH, error_eta_OH,
        mean_eta_Na, error_eta_Na,
        mean_eta_Cl, error_eta_Cl,
        mean_eta_A_zwit, error_eta_A_zwit,
        mean_eta_HA_zwit, error_eta_HA_zwit,
        mean_eta_H2A_zwit, error_eta_H2A_zwit,
        mean_eta_cation, error_eta_cation,
        mean_eta_anion, error_eta_anion,

        mean_eta_H_brush, error_eta_H_brush ,
        mean_eta_OH_brush, error_eta_OH_brush ,
        mean_eta_Na_brush, error_eta_Na_brush ,
        mean_eta_Cl_brush, error_eta_Cl_brush ,
        mean_eta_A_zwit_brush, error_eta_A_zwit_brush,
        mean_eta_HA_zwit_brush, error_eta_HA_zwit_brush,
        mean_eta_H2A_zwit_brush, error_eta_H2A_zwit_brush ,
        mean_eta_cation_brush, error_eta_cation_brush ,
        mean_eta_anion_brush, error_eta_anion_brush ]

        np.savetxt(filename3+str(".out"), [OUTPUT_LIST_1],  header="mean_eta_H, error_eta_H,mean_eta_OH, error_eta_OH, mean_eta_Na, error_eta_Na, mean_eta_Cl, error_eta_Cl, mean_eta_A_zwit, error_eta_A_zwit, mean_eta_HA_zwit, error_eta_HA_zwit, mean_eta_H2A_zwit, error_eta_H2A_zwit, mean_eta_cation, error_eta_cation, mean_eta_anion, error_eta_anion, mean_eta_H_brush, error_eta_H_brush , mean_eta_OH_brush, error_eta_OH_brush , mean_eta_Na_brush, error_eta_Na_brush , mean_eta_Cl_brush, error_eta_Cl_brush , mean_eta_A_zwit_brush, error_eta_A_zwit_brush, mean_eta_HA_zwit_brush, error_eta_HA_zwit_brush, mean_eta_H2A_zwit_brush, error_eta_H2A_zwit_brush , mean_eta_cation_brush, error_eta_cation_brush , mean_eta_anion_brush, error_eta_anion_brush")
              


        OUTPUT_LIST = [mean_polymer_ionization, error_polymer_ionization ,
        mean_zwitterion_uptake_total, error_zwitterion_uptake_total,
        mean_zwitterion_uptake_total_into_brush, error_zwitterion_uptake_total_into_brush,
        mean_chain_end_to_end_distance, error_chain_end_to_end_distance,
        mean_chain_radius_of_gyration, error_chain_radius_of_gyration]


#        OUTPUT_LIST = [mean_polymer_ionization ,mean_eta_H ,mean_eta_OH  ,mean_eta_Na ,mean_eta_Cl  ]
        np.savetxt(filename4+str(".out"), [OUTPUT_LIST],  header="mean_polymer_ionization, error_polymer_ionization,mean_zwitterion_uptake_total, error_zwitterion_uptake_total, mean_zwitterion_uptake_total_into_brush, error_zwitterion_uptake_total_into_brush, mean_chain_end-to-end-distance, error_chain_end-to-end-distance, mean_chain_radius_of_gyration, error_chain_radius_of_gyration")
              
'''


"""
        alpha.append(num_A/(num_A+num_HA))
        eta_H.append((num_H/Volume))
        eta_OH.append((num_OH/Volume))
        eta_Na.append(num_Na/Volume)
        eta_Cl.append(num_Cl/Volume)

#        pH_sys = np.log((csys_H*(conversion_factor_from_1_per_sigma_3_to_mol_per_l/cref_in_mol_per_l)))/np.log(10)

#        file3 = open(filename3+str(".dat"), "a")
        print(num_A, num_H, num_HA, num_OH, num_Na, num_Cl, num_HA_zwit, num_H2A_zwit, num_A_zwit)
        output_vars = np.array([num_A, num_H, num_HA, num_OH, num_Na, num_Cl, num_HA_zwit, num_H2A_zwit, num_A_zwit])
        np.savetxt(filename3+str(".dat"), output_vars, fmt='%.7f\t', delimiter='\t')

        print(i,f"  N_H: {num_H}",
              f"  N_OH: {num_OH}",
              f"  N_Na: {num_Na}",
              f"  N_Cl: {num_Cl}",
              f"  N_HA: {num_HA}",
              f" N_A: {num_A}",
              f"  N_HA_z: {num_HA_zwit}",
              f"  N_H2A_z: {num_H2A_zwit}",
              f"  N_A_z: {num_A_zwit}",
              f"  alpha: {num_A/(num_A+num_HA)}",
              f"  alpha_z: {(num_A_zwit-num_H2A_zwit)/(num_A_zwit+num_H2A_zwit+num_HA_zwit)}")
        
"""
#        with gzip.GzipFile("checkpoint.pgz", 'wb') as fcheck:
#            print("writing checkpoint")
#            data=[num_Hs, num_OHs]
#            pickle.dump(data, fcheck)


#        out_put=[(num_A/(num_A+num_HA)), ((num_H/Volume)), ((num_OH/Volume)), (num_Na/Volume), (num_Cl/Volume)]
#        np.savetxt(file1, [out_put] )




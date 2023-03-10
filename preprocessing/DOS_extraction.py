import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import tqdm
from pymatgen.ext.matproj import MPRester
from pymatgen.electronic_structure.core import Orbital, Spin
from sklearn.preprocessing import OneHotEncoder
from scipy import interpolate

from utils import *
mpr = MPRester('Your API')

class DOSExtractor:
    def __init__(self):
        print('Extraing DOS from MP ...')
        
        data = mpr.query({'nelements': {"$gt":0}}, properties=["task_id","elements","formation_energy_per_atom"])
        
        elem_lst = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn',
                     'Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
                     'In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
                     'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm', 
                     'Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg']
        enc = OneHotEncoder()
        enc.fit(np.reshape(elem_lst,[-1,1]))
        
        self.enc = enc
        self.data = data
        
        
    def extraction(self):
    
        docs = []
        
        print('preprocessing...')
        for datum in tqdm.tqdm(self.data):
        
            try:
                doc = {}
                
                mpid = datum['task_id']
                elements = datum['elements']
                nelement = len(elements)
                dos = mpr.get_dos_by_material_id(mpid)
                energies = dos.energies
                efermi = dos.efermi

                # embedding composition
                ohe = np.sum(self.enc.transform(np.reshape(elements,[-1,1])).toarray(),axis=0)
                
                # DOS summation
                total_dos = totalize(dos)
                
                # interpolate DOS
                dos_lst = []
                for do in total_dos:
                    x, y, imin, imax = interpolate_DOS(energies,do,-20,10,1500)
                    dos_lst.append([i/nelement for i in y])
                padding = np.zeros(1500-len(ohe))
                dos_lst.append(np.concatenate((ohe,padding),axis=None))
    
                doc['id'] = mpid
                doc['elements'] = elements
                doc['energies'] = energies
                doc['efermi'] = efermi
                doc['dos'] = np.array(dos_lst)
                
                docs.append(doc)
             
            except AttributeError:
                pass 
                
        savepickle(docs,'MP_relax_feature.pkl')

def interpolate_DOS(original_x, original_y, emin, emax, nedos):
    
    indmax = np.argmax(original_x > emax)
    indmin = np.argmin(original_x < emin)
    
    if original_x[0] > emin:
        original_x = np.insert(original_x,0,emin)
        original_y = np.insert(original_y,0,0)
    if original_x[-1] < emax:
        original_x = np.insert(original_x,-1,emax)
        original_y = np.insert(original_y,-1,0)
    f1 = interpolate.interp1d(original_x,original_y)
    
    x = np.linspace(emin, emax, nedos)
    y = f1(x)
    
    return x, y , indmin, indmax

def totalize(dos):
    total_dos = []
    for site, atom_dos in dos.pdos.items():
        sitewise = []
        for orb, pdos in atom_dos.items():
            pdos = pdos[Spin.up]
            sitewise.append(pdos)
        total_dos.append(sitewise)
    total_dos = np.sum(np.array(total_dos),axis=0)
    
    return total_dos

extractor = DOSExtractor()
extractor.extraction()

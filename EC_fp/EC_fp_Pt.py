#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:24:03 2019

@author: cchang373
"""

import rdkit
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from imolecule import generate
from bitarray import bitarray
import numpy as np

def get_hash(lst):
    return hash(tuple(lst))
            
def heavy_count(mol,idxs):
    """Count surrounding heavy atom number"""
    count = 0
    for num, bonds in enumerate(mol.GetBonds()):
        if mol.GetBondWithIdx(num).GetBeginAtomIdx() == idxs:
            if mol.GetAtomWithIdx(mol.GetBondWithIdx(num).GetEndAtomIdx()).GetSymbol() != 'H':
                count += 1
        elif mol.GetBondWithIdx(num).GetEndAtomIdx() == idxs:
            if mol.GetAtomWithIdx(mol.GetBondWithIdx(num).GetBeginAtomIdx()).GetSymbol() != 'H':
                count += 1
    return count

def H_count(mol,idxs):
    """Count surrounding hydrogen atom number"""
    mol_H=Chem.AddHs(mol)
    hcount=0
    for num, bonds in enumerate(mol_H.GetBonds()):
        if mol_H.GetBondWithIdx(num).GetBeginAtomIdx() == idxs:
            if mol_H.GetAtomWithIdx(mol_H.GetBondWithIdx(num).GetEndAtomIdx()).GetSymbol() == 'H':
                hcount += 1
        elif mol_H.GetBondWithIdx(num).GetEndAtomIdx() == idxs:
            if mol_H.GetAtomWithIdx(mol_H.GetBondWithIdx(num).GetBeginAtomIdx()).GetSymbol() == 'H':
                hcount += 1
    return hcount

def valence(mol,idxs):
    """
    :rtype: int
    """
    m_valency={'O':2,'C':4,'H':1,'Pt':np.nan,'Rh':np.nan,'Pd':np.nan,'Au':np.nan,'Ag':np.nan}
    #m_valency={'O':2,'C':4,'H':1,'Pt':1,'Rh':1,'Pd':1,'Au':1,'Ag':1}
    atom=mol.GetAtomWithIdx(idxs)
    valency=m_valency[atom.GetSymbol()]
    return valency

def charge(mol,idxs):
    AllChem.ComputeGasteigerCharges(mol)
    charge=float(mol.GetAtomWithIdx(idxs).GetProp('_GasteigerCharge'))
    return charge

def negativity(mol,idxs):
    m_negativity={'O':3.44,'C':2.55,'H':2.2,'Pt':np.nan,'Rh':np.nan,'Pd':np.nan,'Au':np.nan,'Ag':np.nan}
    #m_negativity={'O':3.44,'C':2.55,'H':2.2,'Pt':2.2,'Rh':2.2,'Pd':2.2,'Au':2.2,'Ag':2.2}
    atom=mol.GetAtomWithIdx(idxs)
    electro_negativity=m_negativity[atom.GetSymbol()]
    return electro_negativity

def mass(mol,idxs):
    m_mass = {'O':15.999,'C':12.0107,'H':1.00794,'Pt':np.nan,'Rh':np.nan,'Pd':np.nan,'Au':np.nan,'Ag':np.nan}
    #m_mass={'O':15.999,'C':12.0107,'H':1.00794,'Pt':4.0026,'Rh':4.0026,'Pd':4.0026,'Au':4.0026,'Ag':4.0026}
    atom=mol.GetAtomWithIdx(idxs)
    a_mass=m_mass[atom.GetSymbol()]
    return a_mass

def atomic_number(mol,idxs):
    m_number = {'O':8,'C':6,'H':1,'Pt':np.nan,'Rh':np.nan,'Pd':np.nan,'Au':np.nan,'Ag':np.nan}
    #m_number={'O':8,'C':6,'H':1,'Pt':2,'Rh':2,'Pd':2,'Au':2,'Ag':2}
    atom=mol.GetAtomWithIdx(idxs)
    number=m_number[atom.GetSymbol()]
    return number

def invariants(mol):
    """
    Generate initial atom identifiers using atomic invariants
    :param mol: molecule object parsed from SMILES string
    :rtype: dict
    dict={hash_bit:((atom_idxs,radius),)}
    """
    atoms_dict={}
    
    for idxs,atom in enumerate(mol.GetAtoms()):
        components=[]
        components.append(atomic_number(mol,idxs))
        components.append(heavy_count(mol,idxs))
        components.append(H_count(mol,idxs))
        components.append(valence(mol,idxs))
        #components.append(charge(mol,idxs))
        components.append(negativity(mol,idxs))
        components.append(mass(mol,idxs))
        
        atoms_dict[idxs]=get_hash(components)
    return atoms_dict

def bond(mol,idxs):
    o_bond=[]
    single=rdkit.Chem.rdchem.BondType.SINGLE
    double=rdkit.Chem.rdchem.BondType.DOUBLE
    triple=rdkit.Chem.rdchem.BondType.TRIPLE
    for num, bonds in enumerate(mol.GetBonds()):
        if mol.GetBondWithIdx(num).GetBeginAtomIdx() == idxs:
            index=mol.GetBondWithIdx(num).GetEndAtomIdx()
            if mol.GetBonds()[num].GetBondType() == single:
                order=1
            elif mol.GetBonds()[num].GetBondType() == double:
                order=2
            elif mol.GetBonds()[num].GetBondType() == triple:
                order=3
            o_bond.append([order,index,num])
        elif mol.GetBondWithIdx(num).GetEndAtomIdx() == idxs:
            index=mol.GetBondWithIdx(num).GetBeginAtomIdx()
            if mol.GetBonds()[num].GetBondType() == single:
                order=1
            elif mol.GetBonds()[num].GetBondType() == double:
                order=2
            elif mol.GetBonds()[num].GetBondType() == triple:
                order=3
            o_bond.append([order,index,num])#bond order, another atom index, bond number
    return o_bond

    
def ecfp(mol,radius):
    """
    Generate ecfp fingerprint based on atomic number, heavy atoms and hydrogen atoms count, valency, charge, electronegativity and mass
    :param mol_dict: dict
    :param radius: int
    :rtype: dict
    """
    #mol=Chem.AddHs(mol)
    bitInfo={}
    atoms_dict=invariants(mol)
    
    for idxs,i in atoms_dict.items():
        bitInfo[i]=bitInfo.get(i,())+((idxs,0),)
    
    neighborhoods=[]
    atom_neighborhoods=[len(mol.GetBonds())*bitarray('0') for a in mol.GetAtoms()]
    dead_atoms=len(mol.GetAtoms())*bitarray('0')
    
    for r in range(1,radius+1):
        round_ids={} #new bit ID this iteration
        round_atom_neighborhoods=copy.deepcopy(atom_neighborhoods) #bond to include under this r
        neighborhoods_this_round=[] #(round_atom_neighborhoods,round_ids,idxs)
        
        for idxs,a in enumerate(mol.GetAtoms()):
            if dead_atoms[idxs]:
                continue
            nbsr=[] #list to hash this iteration
            o_bond=bond(mol,idxs)
            for b in o_bond:
                round_atom_neighborhoods[idxs][b[2]] = True
                round_atom_neighborhoods[idxs] |= atom_neighborhoods[b[1]]
                nbsr.append((b[0],atoms_dict[b[1]]))
            nbsr=sorted(nbsr)
            nbsr=[item for sublist in nbsr for item in sublist]
            nbsr.insert(0,atoms_dict[idxs])
            nbsr.insert(0,r)
            
            round_ids[idxs]=get_hash(nbsr)
            neighborhoods_this_round.append((round_atom_neighborhoods[idxs],round_ids[idxs],idxs))
        for lst in neighborhoods_this_round:
            if lst[0] not in neighborhoods:
                bitInfo[lst[1]] = bitInfo.get(lst[1],())+((lst[2],r),)
                neighborhoods.append(lst[0])
            else:
                dead_atoms[lst[2]]=True
        atoms_dict=round_ids
        atom_neighborhoods=copy.deepcopy(round_atom_neighborhoods)
    return bitInfo
    
        
#mol_Pt = Chem.MolFromSmiles('[Rh]')
#mol_Rh = Chem.MolFromSmiles('[Rh]')
#print(ecfp(mol_Pt,0))
#print(ecfp(mol_Rh,1))
#print(ecfp(mol_Rh,2)) 

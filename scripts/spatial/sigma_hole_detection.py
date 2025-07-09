#! /usr/bin/env python3

import copy
import numpy as np

def obtain_coordinates(path1, path2):
    with open(f'{path1}', 'r') as file:
        input = file.readlines()
    coordinates = []
    elem = [] 
    for line in input[2:]: 
        parts = line.split() 
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3]) 
        elem.append(parts[0])
        coordinates.append([x, y, z])
    coordinates_array_1 = np.array(coordinates)
    elem_array_1 = np.array(elem)

    with open(f'{path2}', 'r') as file:
        input = file.readlines()
    coordinates = []
    elem = [] 
    for line in input[2:]: 
        parts = line.split() 
        x, y, z =  float(parts[1]), float(parts[2]), float(parts[3]) 
        elem.append(parts[0])
        coordinates.append([x, y, z])
    coordinates_array_2 = np.array(coordinates)
    elem_array_2 = np.array(elem)
    return coordinates_array_1, coordinates_array_2, elem_array_1, elem_array_2

def obtain_coonectivity(coord_1, bond_threshold=2.5):
    "Find bonds between atoms in coord_1 based on distance threshold "
    bond = []
    for i, c1 in enumerate(coord_1):
        for j, c2 in enumerate(coord_1):
            if i < j:
                distance = np.linalg.norm(c1 - c2)
                if distance < bond_threshold:  # Add bond if within threshold
                    bond.append([i, j])
    return bond
    

def calculate_distance(coords1, coords2):
    min_distance = float('inf')
    for atom1 in coords1:
        for atom2 in coords2:
            distance = np.linalg.norm(atom1 - atom2)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def calculate_min_dist_to_atom(coords1, coords2):
    min_distance = float('inf')
    min_index = -1
    for i, atom2 in enumerate(coords2):
        distance = np.linalg.norm(coords1 - atom2)
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_distance, min_index

def detect_sigma_relevant_atoms(elem_1, bonds_1, coord_1, coord_2, halogen='Cl'):
    # Set distance cutoff based on element type
    dist_cutoffs = {'Cl': 4.0, 'Br': 4.5, 'I': 5.0, 'S': 4.0, 'P': 4.0, 'As': 4.0, 'Se': 4.0}
    dist_cutoff = dist_cutoffs.get(halogen, 4.0)
    results = []
    for i, el in enumerate(elem_1):
        if el == halogen:
            neighbors = []
            for bond in bonds_1:
                if i in bond:
                    neighbor = bond[0] if bond[1] == i else bond[1]
                    neighbors.append(neighbor)
            for j, atom2 in enumerate(coord_2):
                dist = np.linalg.norm(coord_1[i] - atom2)
                if dist < dist_cutoff:
                    # For each neighbor, measure angle, but only keep the max angle for this contact
                    max_angle = None
                    for iNeigh in neighbors:
                        vector1 = coord_1[i] - coord_1[iNeigh]
                        vector2 = coord_1[i] - atom2
                        dot_product = np.dot(vector1, vector2)
                        magnitude1 = np.linalg.norm(vector1)
                        magnitude2 = np.linalg.norm(vector2)
                        if magnitude1 == 0 or magnitude2 == 0:
                            continue
                        angle = np.degrees(np.arccos(dot_product / (magnitude1 * magnitude2)))
                        if (max_angle is None) or (angle > max_angle):
                            max_angle = angle
                    if max_angle is not None:
                        results.append((i, j, round(max_angle, 3), halogen))
    return results

def find_sigma_bond(path1, path2):
    coord_1, coord_2, elem_1, elem_2 = obtain_coordinates(path1, path2)
    bonds_1 = obtain_coonectivity(coord_1)
    bonds_2 = obtain_coonectivity(coord_2)
    possible_sigma_bond = []
    for element in ['Cl', 'Br', 'I', 'S', 'P', 'As', 'Se']:
        if element in elem_1:
            results = detect_sigma_relevant_atoms(elem_1, bonds_1, coord_1, coord_2, halogen=element)
            for res in results:
                possible_sigma_bond.append([1, res[0], 2, res[1], res[2], element])
        if element in elem_2:
            results = detect_sigma_relevant_atoms(elem_2, bonds_2, coord_2, coord_1, halogen=element)
            for res in results:
                possible_sigma_bond.append([2, res[0], 1, res[1], res[2], element])
    
    sigma_bond = []
    for s in possible_sigma_bond:
        # s = [mol_idx, halogen_idx, other_mol_idx, iNCI, angle, element]
        # Determine which element array to use for iNCI
        if s[0] == 1:
            contact_elem = elem_2[s[3]]
        else:
            contact_elem = elem_1[s[3]]
        # Set angle threshold based on contact atom type
        if contact_elem == 'C':
            angle_thresh = 150
        elif contact_elem in ['O', 'N']:
            angle_thresh = 170
        else:
            angle_thresh = 150  # Default/fallback
        if s[4] > angle_thresh:
            if s[0] == 1:
                sigma_bond.append([s[1], s[3]])
            elif s[2] == 1:
                sigma_bond.append([s[3], s[1]])
    return sigma_bond


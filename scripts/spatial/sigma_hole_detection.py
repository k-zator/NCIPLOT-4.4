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
    " find bonds between atoms in coord_1 based on distance threshold "
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
    for i, atom2 in enumerate(coords2):
        distance = np.linalg.norm(coords1 - atom2)
        if distance < min_distance:
            min_distance = distance
    return min_distance, i

def detect_sigma_relevant_atoms(elem_1, bonds_1, coord_1, coord_2):
    for i, el in enumerate(elem_1):
        if el == 'Cl':
            iCl = i
            # identify what is Cl's neighbour - there should be only one bond
            neigh = [x for x in bonds_1 if iCl in x][0]
            # remove link to bonds1 list so neigh.remove(iCl) doesn't change bonds1
            neigh = copy.deepcopy(neigh)
            neigh.remove(iCl)
            iNeigh = neigh[0]
            # now identify third atom in the angle
            dist, iNCI = calculate_min_dist_to_atom(coord_1[iCl], coord_2)
            if dist < 4: # i.e. when there's a sigma-hole in principle
                # calculate the angle between the two vectors
                vector1 = coord_1[iCl] - coord_1[iNeigh]
                vector2 = coord_1[iCl] - coord_2[iNCI]

                dot_product = np.dot(vector1, vector2)
                magnitude1 = np.linalg.norm(vector1) # normalise
                magnitude2 = np.linalg.norm(vector2)
                angle = np.arccos(dot_product / (magnitude1 * magnitude2))
                angle = np.degrees(angle)
    return iCl, iNCI, round(angle,3)


def find_sigma_bond(path1, path2):
    # Load the coordinates and elements from the XYZ files
    coord_1, coord_2, elem_1, elem_2 = obtain_coordinates(path1, path2)
    bonds_1 = obtain_coonectivity(coord_1)
    bonds_2 = obtain_coonectivity(coord_2)
    sigma_bond = []
    # Search for the Cl atom in the first molecule
    if "Cl" in elem_1:
        iCl, iNCI, angle = detect_sigma_relevant_atoms(elem_1, bonds_1, coord_1, coord_2)
        sigma_bond.append([1, iCl, 2, iNCI, angle])
    elif "Cl" in elem_2:
        iCl, iNCI, angle = detect_sigma_relevant_atoms(elem_2, bonds_2, coord_2, coord_1)
        sigma_bond.append([2, iCl, 1, iNCI, angle])
    return sigma_bond
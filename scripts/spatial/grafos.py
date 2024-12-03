import networkx as nx

def build_adsorption_graph(num_atoms, plane_threshold=4, covalent_radius=0.7, adsorption_distance=0.5):
    # Crear grafo inicial
    graph = nx.Graph()
    
    # Crear el primer átomo en el primer plano
    graph.add_node("A1", layer=0, position=(0, 0, 0))
    
    # Parámetros de control
    current_atom = 1
    current_plane = 0

    # Expandir el grafo de acuerdo al número de átomos
    while current_atom < num_atoms:
        current_atom += 1
        atom_label = f"A{current_atom}"
        
        # Decidir el nuevo sitio de adsorción
        # Buscar un nodo de referencia en el grafo
        reference_node = list(graph.nodes)[-1]
        ref_position = graph.nodes[reference_node]["position"]
        ref_layer = graph.nodes[reference_node]["layer"]
        
        # Calcular si se debe ubicar en el mismo plano o en un plano superior
        if ref_layer < plane_threshold:
##            # aca cambiariamos la ubicacion de donde se puso el uevo atomo, el plan aleatorio aca
            new_position = (ref_position[0], ref_position[1], ref_position[2] + adsorption_distance)
            new_layer = ref_layer + 1
        else:
            # Colocar en el mismo plano
            new_position = (ref_position[0] + adsorption_distance, ref_position[1], ref_position[2])
            new_layer = ref_layer

        # Agregar el nuevo átomo al grafo
        graph.add_node(atom_label, layer=new_layer, position=new_position)
        graph.add_edge(reference_node, atom_label)
        
    return graph

# Configuración y ejecución
adsorption_graph = build_adsorption_graph(num_atoms=10)
print(adsorption_graph)

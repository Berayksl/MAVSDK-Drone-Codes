import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def buchi_automaton_creator(txt_file): #this function reads the output of
    G = nx.DiGraph()
    f = open(txt_file, "r")
    f.readline() #skip the first line

    state_labels = {} # a dictionary that keeps the labels for all states
    accepting_states = []

    for line in f: #iterate over the lines to add the nodes into the graph
        line_parts = line.strip().split()
        if '/*' in line_parts:
            node = line_parts[0]

            #check if a given node is the initial or accepting state:
            if 'init' in node:
                initial_state = node
            elif 'accept' in node:
                accepting_states.append(node)

            if node == 'accept_all':
                G.add_edge(node, node, label='(1)')

            label = line_parts[line_parts.index('/*') + 1]
            
            if not G.has_node(node):
                G.add_node(node)
            G.nodes[node]['label'] = label
            check_if = f.readline().split()[0]

            if check_if == 'if': #check if we are in an 'if' statement block
                statement = f.readline().strip().split(' -> goto ')
                while '::' in statement[0]:
                    edge_label = statement[0][3:]
                    second_node = statement[1]
                    G.add_edge(node, second_node, label=edge_label)
                    statement = f.readline().strip().split(' -> goto ')


    #draw the graph:
    np.random.seed(1)
    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G, 'label')
    if txt_file == 'Spec-2.txt':
        nx.draw(G,pos,with_labels=True, labels = node_labels, font_weight='bold',arrows = True, node_size=800, node_color='lightblue', font_color='black', font_size=9,connectionstyle="arc3,rad=0.3")
    else:
        nx.draw(G,pos,with_labels=True, labels = node_labels, font_weight='bold',arrows = True, node_size=800, node_color='lightblue', font_color='black', font_size=9)
    
    #designate the initial and accpeting states with specified colors:
    nx.draw_networkx_nodes(G, pos, nodelist=[initial_state],node_color='yellow', node_size=800)
    nx.draw_networkx_nodes(G, pos, nodelist=accepting_states,node_color='green', node_size=800)

    edge_labels = nx.get_edge_attributes(G, 'label')
   
    #Place the label texts on the arrows (i had to place them manually instead of using draw_networkx_edge_labels() function because the self loop labels would overlap with the nodes):
    i = 0
    for edge, label in edge_labels.items():
        if edge[0] == edge[1]:
            x1, y1 = pos[edge[0]]
            label_x = x1
            label_y = y1 + 0.1
        else:
            if txt_file == 'Spec-2.txt':
                if i == 0:
                    label_pos = 0.4
                    x1, y1 = pos[edge[0]]
                    x2, y2 = pos[edge[1]]
                    label_x =  x1 * label_pos + x2 * (1.0 - label_pos)
                    label_y = y1 * label_pos + y2 * (1.0 - label_pos)
                    
                else:
                    label_pos = 0.3
                    x1, y1 = pos[edge[0]]
                    x2, y2 = pos[edge[1]]
                    label_x =  x1 * label_pos + x2 * (1.0 - label_pos) - 0.17
                    label_y = y1 * label_pos + y2 * (1.0 - label_pos) 
            else:
                label_pos = 0.5
                x1, y1 = pos[edge[0]]
                x2, y2 = pos[edge[1]]
                label_x =  x1 * label_pos + x2 * (1.0 - label_pos)
                label_y = y1 * label_pos + y2 * (1.0 - label_pos)
            i += 1
        plt.text(label_x, label_y, label, fontsize=10, color='black',fontweight='bold')

    plt.show()

    return G, initial_state
#automaton_creator('Spec_Negation.txt')

''' 
#NEW VERSION FOR TIME PRODUCT MDP

def logic_statement_parser(transition_system,buchi,trans_neighbor,current_node, time_horizon):
#takes the current state and returns the list of valid edges which is the list of valid PA transitions
  valid_edges = []
  trans_state_label = transition_system.nodes[trans_neighbor]['label']
  trans_state = current_node[0]
  buchi_state = current_node[1]
  t = current_node[2] 
  for buchi_neighbor in list(buchi.neighbors(buchi_state)):
      logic_statement = buchi.get_edge_data(buchi_state, buchi_neighbor)['label']
      logic_statement = logic_statement.replace('(','')
      logic_statement = logic_statement.replace(')','')

      if '&&' in logic_statement:
          temp = logic_statement.split('&&')
      else:
          temp = logic_statement.split('||')

      logic_statement_elements = [element.replace('!', "") for element in temp] #remove '!'
      logic_statement_elements = [element.replace(' ', "") for element in logic_statement_elements] #remove spaces

      logic_statement = logic_statement.replace('!', "not ") #replace ! with 'not' for eval() function
      logic_statement = logic_statement.replace('&&', "and") #replace && with 'and' for eval() function
      logic_statement = logic_statement.replace('||', "or") #replace || with 'or' for eval() function
      
      variables = {}
      for a in logic_statement_elements:
          if a == '1': #accept all observations
              variables[a] = True
          else:
              variables[a] = trans_state_label == a

      if eval(logic_statement, variables): #check if transition system observation satisfies the edge label for transition in buchi automaton
          if t < time_horizon-1:
            valid_edges.append(((trans_state, buchi_state, t), (trans_neighbor, buchi_neighbor, t+1), logic_statement))
  return valid_edges
'''

def logic_statement_parser(transition_system,buchi,trans_state, trans_neighbor, buchi_state):
#takes the current state and returns the list of valid edges which is the list of vali PA transitions
  valid_edges = []
  trans_state_label = transition_system.nodes[trans_neighbor]['label']
  for buchi_neighbor in list(buchi.neighbors(buchi_state)):
      logic_statement = buchi.get_edge_data(buchi_state, buchi_neighbor)['label']
      logic_statement = logic_statement.replace('(','')
      logic_statement = logic_statement.replace(')','')


      if '&&' in logic_statement:
          temp = logic_statement.split('&&')
      else:
          temp = logic_statement.split('||')

      logic_statement_elements = [element.replace('!', "") for element in temp] #remove '!'
      logic_statement_elements = [element.replace(' ', "") for element in logic_statement_elements] #remove spaces

      logic_statement = logic_statement.replace('!', "not ") #replace ! with 'not' for eval() function
      logic_statement = logic_statement.replace('&&', "and") #replace && with 'and' for eval() function
      logic_statement = logic_statement.replace('||', "or") #replace || with 'or' for eval() function

      variables = {}
      for a in logic_statement_elements:
          if a == '1': #accept all observations
              variables[a] = True
          else:
              variables[a] = trans_state_label == a

      if eval(logic_statement, variables): #check if transition system observation satisfies the edge label for transition in buchi automaton
          valid_edges.append(((trans_state, buchi_state), (trans_neighbor, buchi_neighbor), logic_statement))
  return valid_edges


def create_product_automaton(transition_system, buchi, initial_buchi_state):
    product_automaton = nx.DiGraph()

    for state1 in transition_system.nodes():
      for state2 in buchi.nodes():
          product_state = (state1, state2)
          product_automaton.add_node(product_state)

    #remove the initial nodes whose observation satisfies any of the FSA/Buchi edges
    initial_nodes_to_prune = []
    for node in product_automaton.nodes():
       if 'init' in node[1]:
        trans_state = node[0]
        buchi_state = node[1]
        valid_edges = logic_statement_parser(transition_system, buchi,trans_state,trans_state, buchi_state)
        l = [edge[1][1] for edge in valid_edges]
        if any(element != initial_buchi_state for element in l):
           initial_nodes_to_prune.append(node)
    product_automaton.remove_nodes_from(initial_nodes_to_prune)


    for node in product_automaton.nodes():
        trans_state = node[0]
        buchi_state = node[1]
        for trans_neighbor in list(transition_system.neighbors(trans_state)):
          trans_state_label = transition_system.nodes[trans_neighbor]['label']
          valid_edges = logic_statement_parser(transition_system, buchi, trans_state, trans_neighbor, buchi_state)
          try:
              c = 0 #flag that checks if any edges added to the graph
              for edge in valid_edges:
                  if trans_state_label in edge[2]:
                      product_automaton.add_edge(edge[0], edge[1])
                      c = 1
              if c == 0:
                  product_automaton.add_edge(valid_edges[0][0], valid_edges[0][1])

          except:
              continue
    #prune the automaton by eliminating dead end nodes:

    # nodes_to_prune = []
    # for node in product_automaton.nodes():
    #     if len(list(product_automaton.neighbors(node))) == 0:
    #         nodes_to_prune.append(node)

    # product_automaton.remove_nodes_from(nodes_to_prune)


    #shortest_path = nx.shortest_path(product_automaton, source=((0,0),'T0_init'), target=((7,7),'accept_all'), method='dijkstra')
    #print(shortest_path)

    return product_automaton


def create_time_product_MDP(transition_system, buchi, initial_buchi_state, time_horizon):
    time_product_MDP = nx.DiGraph()

    for state1 in transition_system.nodes():
      for state2 in buchi.nodes():
          for t in range(time_horizon):
            product_state = (state1, state2, t)
            time_product_MDP.add_node(product_state)

    #remove the initial nodes whose observation satisfies any of the FSA/Buchi edges
    initial_nodes_to_prune = []
    for node in time_product_MDP.nodes():
       if 'init' in node[1]:
            trans_state = node[0]
            buchi_state = node[1]
            valid_edges = logic_statement_parser(transition_system, buchi,trans_state,node, time_horizon)
            l = [edge[1][1] for edge in valid_edges]
            if any(element != initial_buchi_state for element in l):
                initial_nodes_to_prune.append(node)

    time_product_MDP.remove_nodes_from(initial_nodes_to_prune)

    for node in time_product_MDP.nodes():
        trans_state = node[0]
        buchi_state = node[1]
        for trans_neighbor in list(transition_system.neighbors(trans_state)):
            trans_state_label = transition_system.nodes[trans_neighbor]['label']
            valid_edges = logic_statement_parser(transition_system, buchi, trans_neighbor,node, time_horizon)
            try:
                c = 0 #flag that checks if any edges added to the graph
                for edge in valid_edges:
                    if trans_state_label in edge[2]:
                        time_product_MDP.add_edge(edge[0], edge[1])
                        c = 1
                if c == 0:
                    time_product_MDP.add_edge(valid_edges[0][0], valid_edges[0][1])

            except:
                continue
          
    #prune the automaton by eliminating dead end nodes:

    nodes_to_prune = []
    for node in time_product_MDP.nodes():
        if len(list(time_product_MDP.neighbors(node))) == 0:
            nodes_to_prune.append(node)

    time_product_MDP.remove_nodes_from(nodes_to_prune)


    #shortest_path = nx.shortest_path(product_automaton, source=((0,0),'T0_init'), target=((7,7),'accept_all'), method='dijkstra')
    #print(shortest_path)

    return time_product_MDP
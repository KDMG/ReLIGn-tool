import os
import pathlib
import re
import subprocess
from timeit import default_timer as timer
import pandas as pd
from numpy.f2py.auxfuncs import throw_error
from pandas.core.common import flatten
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
# from pm4py.objects.log.adapters.pandas import csv_import_adapter as csv_importer #pm4py-1.5.0.1
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as petrinet_visualizer
from pm4py.visualization.petri_net.variants import token_decoration_frequency
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.obj import Trace
from pm4py.objects.log.obj import Event
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils as utils
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from ProcessRepairing.scripts.database import query
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignment
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
from progress.bar import IncrementalBar
import random
import argparse
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

""" Function to split the file containing subs. It returns a list
INPUT: -pathubfile: path to the file (e.g. *_new_patterns_filtered.subs)
RETURN: -a: list of sub files
"""


def split_subgraph(pathsubfile):
    var_lettura = open(pathsubfile, "r").readlines()

    a = []
    for x in var_lettura:

        if x != "\n":
            c = x.strip("\n")
            b = c.split(" ")
            for y in b:
                if y == '':
                    b.remove('')
            a.append(b)
    return a


"""
INPUT: -sub_ocmatrix_file: path to the occurrence matrix for the subs (e.g., *_table2_on_file.csv)
       -subname: subgraph
RETURN: list of graphs in which the input subgraph occurs
"""


def list_graph_occurence(sub_ocmatrix_file, subname):
    # df = csv_importer.import_dataframe_from_path(sub_ocmatrix_file, sep=";")  #pm4py-1.5.0.1
    df = pd.read_csv(sub_ocmatrix_file, sep=';')
    graphs = []
    for x in range(len(df)):
        if (df.loc[x]["Sub" + subname] == 1):
            grafo = df.loc[x]['grafo']
            n = grafo[5:]
            graphs.append("graph" + n)
    return graphs


""" 
INPUT: -lista: list of strings representing integers
RETURN: -max: max number in the list
"""


def massimo_lista(lista):
    max = int(lista[0])
    pos = 1
    while pos < len(lista):
        if int(lista[pos]) > max:
            max = int(lista[pos])
        pos = pos + 1
    return max


""" 
INPUT: -lista: list of strings representing integers
RETURN: -min: min number in the list
"""


def minimo_lista(lista):
    min = int(lista[0])
    pos = 1
    while pos < len(lista):
        if int(lista[pos]) < min:
            min = int(lista[pos])
        pos = pos + 1
    return str(min)


"""
INPUT: -places: set of places in a net
RETURN: -new_place: the available name for a new place
"""


def places_name_available(places, transitions):
    place_name = []
    place_number = []
    trans_name = []

    for place in places:
        place_name.append(place.name)
    for trans in transitions:
        trans_name.append(trans.name)
    for x in place_name:
        place_number.append(int(x.split("n")[1]))
    for y in trans_name:
        if y[:1] == "n":
            place_number.append(int(y.split("n")[1]))

    max = massimo_lista(place_number)
    new_place = str(max + 1)
    return new_place


"""
INPUT: -transations: set of transition of the net
RETURN: -new_transation: an available name for a new transition
"""


def transition_hidden_available(transitions):
    trans_name = []
    trans_number = []

    for trans in transitions:
        trans_name.append(trans.name)

    for x in trans_name:
        if x[:1] == "h":
            trans_number.append(int(x.split("h")[1]))

    if (trans_number != []):
        max = massimo_lista(trans_number)
        new_trans = str(max + 1)
        return new_trans
    else:
        return str(1)


"""
INPUT: -transations: set of transitions in the net
RETURN: -new_transation: an available name for a new transition
"""


def transition_name_available(transitions):
    trans_name = []
    trans_number = []

    for trans in transitions:
        trans_name.append(trans.name)

    for x in trans_name:
        if x[:1] == "s":
            trans_number.append(int(x.split("s")[1]))

    if (trans_number != []):
        max = massimo_lista(trans_number)
        new_trans = str(max + 1)
        return new_trans
    else:
        return str(1)


"""
INPUT: -path_file: list of patterns with corresponding subs (e.g., bpmdemo2_new_patterns_filtered.subs)
RETURN: list of patterns (as a list of lists of subs)
"""


# "*_new_patterns_filtered.subs"
def create_patterns_list(path_file):
    patterns = []
    sub = []
    a = split_subgraph(path_file)
    for y in a:
        if y != ['S'] and y[0] != 'd':
            sub.append(y[2][4:])
        elif y == ['S']:
            if sub != []:
                patterns.append(sub)
                sub = []
    patterns.append(sub)
    return patterns


"""
INPUT: -pattern_file: list of patterns with corresponding subs (e.g., bpmdemo2_new_patterns_filtered.subs)
       -pattern_number: index of the pattern in the list
RETURN: list of sub for the input pattern
"""


# "*_new_patterns_filtered.subs"
def list_sub_pattern(pattern_file, pattern_number):
    pattern_list = create_patterns_list(pattern_file)
    list_sub = pattern_list[pattern_number - 1]
    return list_sub


"""
INPUT: -n: index of a sub
       -sub_file: path to a file containing the list of all subgraphs (e.g., nomedataset.subs) 
RETURN: the instance graph of the input sub
"""


# "*.subs"
def sub_graph(sub_file):
    subgraph = open(sub_file, "r").readlines()
    return subgraph


"""
The function writes the file sub_sgiso_input.txt that can be used as a first argument for the sgiso tool
INPUT: -subgrap: sub risultato di sub_graph()
       -pattern: the path "../patterns_file/" 
"""


def write_subfile(subgrap, pattern, graph_name):
    file = open(os.path.join(pattern, graph_name + ".g"), "w")

    for x in subgrap:
        file.write(x)
    file.close()


"""  The function prints the input string to a file
INPUT: -output: string to write in the file
       -pattern: the folder "../patterns_file/"
       -sub: a string representing the id of the sub 
       -mod: opening mode for the file
"""
def write_outputfile(output, pattern, sub, mod, name = ''):
    lig_name = os.path.splitext(os.path.basename(sub))[0] + name
    print(output, flush=True)
    with open(os.path.join(sub, "output_" + lig_name +".txt"), mod) as file:
        file.write(output + "\n")



"""
The function writes files graphn.g that can be used as input for the tool gm
INPUT: -subgrap: the instance graph of the input sub (i.e. the output of sub_graph())
       -n: graph number
       -pattern: the folder "../patterns_file/"
"""
def write_graphfile(subgrap, n, pattern):
    subcopy = []
    for x in subgrap:
        subcopy.append(x)
    i = 1
    dict = {}

    for x in range(len(subcopy)):
        if subcopy[x] == 'Found':
            break
        elif subcopy[x] == 'v':
            dict[subcopy[x + 1]] = i
            subcopy[x + 1] = i
            i = i + 1
        elif subcopy[x] == 'd' or subcopy[x] == 'e':
            subcopy[x + 1] = dict[subcopy[x + 1]]
            subcopy[x + 2] = dict[subcopy[x + 2]]

    file = open(os.path.join(pattern, "graph" + n + ".g"), "w")

    for x in range(len(subcopy)):
        if subcopy[x] == 'Found':
            break
        elif subcopy[x] == 'v' or subcopy[x] == 'd' or subcopy[x] == 'e':
            file.write('\n' + subcopy[x])
        else:
            file.write(' ' + str(subcopy[x]))
    file.close()


"""
INPUT: -sub_number: the number of a sub (it will be used to get the IG from the sub through sub_graph)
       -graph_number: the number of the graph (IG of the trace) used as input to sgiso
       -pattern: folder "../patterns_file/"
RETURN: the output of the sgiso tool
"""


def find_instances(graph, pattern):
    subgraph = sub_graph(os.path.join(pattern, "subelements.txt"))
    if 'S\n' in subgraph:
        subgraph.remove('S\n')

    write_subfile(subgraph, pattern, 'graphsub')
    write_subfile(graph, pattern, 'complete_graph')

    out = subprocess.Popen([os.path.join('subdue_files', 'sgiso'),
                            os.path.join(pattern, 'graphsub.g'),
                            os.path.join(pattern, 'complete_graph.g')],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    sub = stdout.decode("utf-8")
    sub2 = sub.split()

    return sub2


def create_subelements_file(name_database, pattern):
    testo = []
    n = query.query_count_row(name_database)

    for x in range(n):

        riga1 = query.query_with_fetchone(x, name_database)

        if type(riga1) == tuple:
            if riga1[0] == None:
                riga1 = None
        if riga1 == None:
            riga2 = query.query_subcontent(x, name_database)
            if riga2 == None:
                continue
            riga = riga2[0].split(" ")

            if riga[1] != '1':
                a = []
                for x in riga:
                    try:
                        val = int(x)
                    except ValueError:
                        continue
                    if x not in a:
                        a.append(x)
                i = 1
                for y in a:
                    for w in range(len(riga)):
                        try:
                            val = int(riga[w])
                        except ValueError:
                            continue
                        if riga[w] == y:
                            riga[w] = i
                    i = i + 1

            # print(riga)

            testo.append(riga)
        else:
            # print(riga1[0])
            testo.append(riga1[0])

    file = open(os.path.join(pattern, "subelements.txt"), "w")

    for y in testo:
        file.write("\nS\n")
        if type(y) == list:
            for k in y:
                file.write(str(k))
                file.write(" ")

        else:
            file.write(str(y))

    file.close()


"""
INPUT: -graph: a sub (i.e. the output of find_instances())
RETURN: start_sub: list of start nodes in the sub, end_sub: list of end nodes in the sub.
"""


def startend_node(graph):
    sub_label = []
    start_edge = []
    end_edge = []
    node = []
    for x in range(len(graph)):
        if graph[x] == "instances.":
            break
        elif graph[x] == "v":
            node.append(graph[x + 1])
            sub_label.append(graph[x + 2])
        elif graph[x] == "d" or graph[x] == "e":
            start_edge.append(graph[x + 1])
            end_edge.append(graph[x + 2])

    start_sub = []
    end_sub = []
    for y in node:
        if start_edge == [] and end_edge == []:
            start_sub.append(y)
            end_sub.append(y)
        elif y not in start_edge:
            end_sub.append(y)
        elif y not in end_edge:
            start_sub.append(y)

    return start_sub, end_sub, sub_label


"""
INPUT: -graph: a sub (i.e. the output of find_instances())
RETURN: start_sub: list of starting nodes of the sub, end_sub: list of ending nodes in the sub
"""


def startend_graph(graph):
    start_edge = []
    end_edge = []
    node = []
    for x in graph:
        if x[0] == "v":
            node.append(x[1])
        elif x[0] == "d" or x[0] == "e":
            start_edge.append(x[1])
            end_edge.append(x[2])

    start_sub = []
    end_sub = []
    for y in node:
        if y not in start_edge:
            end_sub.append(y)
        elif y not in end_edge:
            start_sub.append(y)

    return start_sub, end_sub


"""
INPUT: -graph_number: the number of a graph
       -pattern: the folder "../patterns_file/"
RETURN: id for the alignment
"""


def get_id_mapping(graph_number, pattern):
    file = open(os.path.join(pattern, "traceIdMapping.txt"), "r")

    for y in file:
        y1 = []
        i = 0
        for y3 in y:
            if y3 == ";":
                i = i + 1
                break
            else:
                y1.append(y3)
            i = i + 1
        c = "".join(y1)
        if int(c) == graph_number:
            y4 = y[i:]
            break

    file.close()
    return y4


""" The function creates the dict with numTrace and traceId by querying the db
RETURN: -dict_traceid: a dict with pairs 'numTrace':'idTrace'
"""


def create_dict_trace(name_database):
    dict_traceid = {}

    traceid = query.query_with_fetchall(name_database)
    for x in traceid:
        dict_traceid['graph' + x[0]] = x[1]
    return dict_traceid


""" The function returns from the log the trace object corresponding to the input graph
INPUT: -log: an event log
       -dict_trace: a dictionary of pairs idTrace and numTrace
       -graph: the number of the trace
RETURN: -trace: an object of type Trace containing the target trace
"""


def search_trace(log, dict_trace, graph):
    trace = Trace()
    for t in log:
        if t.attributes['concept:name'] == dict_trace[graph]:
            trace = t
    return trace


""" The functions returns the type of move in an aligment
INPUT: -move: a tuple of the alignment with labels on trasitions
RETURN: -"M": move on model
        -"L": move on log
        -"L/M": synchronous move
"""


def def_move(move):
    if move[0] == ">>" and move[1] != ">>":
        return "M"
    elif move[0] != ">>" and move[1] == ">>":
        return "L"
    else:
        return "L/M"


""" The functions takes a graph in input and returns the corresponding alignment
INPUT: -pattern: the folder with the files
       -dict_trace: a dictionarity with pairs idTrace:numTrace
       -graph: the number of a trace
RETURN: -text: the alignment
"""


def search_alignment(pattern, dict_trace, graph, dataset):
    if not os.path.isfile(os.path.join(pattern, "alignment.csv")):
        cmd = ['java', "-jar", os.path.join('BIGfiles', 'ComputePrecision.jar'), pattern + dataset + '.xes', pattern + dataset + '_petriNet.pnml']
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Error in JAR execution:")

    lines = open(os.path.join(pattern, "alignment.csv"), "r").readlines()

    i = 0
    for i, line in enumerate(lines):
        if line.startswith("Index"):
            break
    if i != 0:
        with open(os.path.join(pattern + "alignment.csv"), "w") as file:
            file.writelines(lines[i:])
    # df = csv_importer.import_dataframe_from_path(pattern + "alignment.csv", sep=",") #pm4py-1.5.0.1
    df = pd.read_csv(os.path.join(pattern, "alignment.csv"), sep=",")

    for j in range(len(df)):

        if df.loc[j]['Match'] == "NaN":
            break
        else:
            trace_string = df.loc[j]['Case IDs']
            #if(type(trace_string)=='str'):
            list_trace = trace_string.split('|')

        if dict_trace[graph] in list_trace:
            alignment = df.loc[j]['Match']
            break
            
    text = alignment.split("|")

    return text


""" The function checks that the input sub occurs in the graphs in the list
INPUT: -graph_list: a list of graph names
       -subnumber: the number of a sub
       -pattern: the folder "../patterns_file/"
RETURN: -graph_list: the list containing only graphs with the input sub
"""


def check_graphlist(input_data, graph_list, sub, pattern):
    list = []
    for x in graph_list:
        list.append(x)
    bar = IncrementalBar('Correctness check graph_list', max=len(graph_list))
    for x in list:
        sub = find_instances(graph_list[x], pattern)
        if sub[1] == '0':
            graph_list.pop(x)
        else:
            if input_data['fast_mode'] == True:
                graph_list = {x: graph_list[x]}
                break
            bar.next()
    bar.finish()
    return graph_list


""" The function gets the Raw Fitness Cost, from file alignment.csv, of the trace corresponding to the input graph
INPUT: -pattern: folder including the files
       -dict_trace: a dictionary with pairs idTrace:numTrace
       -graph: the graph name
RETURN: -float(cost): Raw Fitness Cost
"""


def search_fitness_cost(pattern, dict_trace, graph):
    lines = open(os.path.join(pattern, "alignment.csv"), "r").readlines()

    if lines[0][:5] != "Index":
        open(os.path.join(pattern, "alignment.csv"), "w").writelines(lines)

    # df = csv_importer.import_dataframe_from_path(pattern + "alignment.csv", sep=",") #pm4py1.5.0.1
    df = pd.read_csv(os.path.join(pattern, "alignment.csv"), sep=",")

    for j in range(len(df)):
        if df.loc[j]['Match'] == "NaN":
            break
        else:
            trace_string = df.loc[j]['Case IDs']
            list_trace = trace_string.split('|')

        if dict_trace[graph] in list_trace:
            cost = df.loc[j]['Raw Fitness Cost']
            break

    return float(cost)


""" The function gets the Raw Fitness Cost, from file alignment.csv, of the trace corresponding to the input graph
INPUT: -pattern: folder including the files
       -dict_trace: a dictionary with pairs idTrace:numTrace
       -graph_list: a list of graphs in which the target sub occurs
RETURN: -first_trace: a graph with the smallest Raw Fitness Cost
        -mincost: Raw Fitness Cost
"""


def select_graph(pattern, dict_trace, graphlist):
    mincost = search_fitness_cost(pattern, dict_trace, graphlist[0])
    mintracelist = []
    for x in graphlist:
        cost = search_fitness_cost(pattern, dict_trace, x)
        if cost < mincost:
            mincost = cost
            mintracelist = []
            mintracelist.append(x)
        elif cost == mincost:
            mintracelist.append(x)

    first_trace = random.choice(mintracelist)
    # print("Raw Fitness Cost: ", mincost, "Graph list: ", mintracelist)
    return first_trace, mincost


""" The function searches in the input alignment the position of the start transition of the graph, counting synchronous/log moves. 
    For each start node, it takes all transitions before the identified point, only considering moves on model and synchronous moves. Then it applies a
    token-based replay, obtaining the reached_marking.
INPUT: -pattern: the folder containing the files
       -dataset: the dataset name
       -trace: the trace name
       -start: a list of start nodes in the graph
RETURN: -reached_marking: a dictionary with pairs 'start':'marking'
"""


def dirk_marking_start(dataset, start, text, trace, pattern, namesub):
    net, initial_marking, final_marking = pnml_importer.apply(os.path.join(pattern, dataset + '_petriNet.pnml'))
    new_trace = Trace(attributes=trace.attributes)
    im = str(initial_marking).strip('[]\'').split(':')
    i_marking = im[0]
    m = minimo_lista(start)

    k = 0
    reached_marking = []
    del_event = []
    g = 0
    for w in text:
        if int(m) == 1:
            break
        elif k == int(m):
            break

        if w[:3] == "[L]":
            k = k + 1
            g = g + 1
            if k != int(m):
                for d in del_event:
                    new_trace.append(d)
                del_event = []
        elif w[:5] == "[L/M]":
            k = k + 1
            if k != int(m):

                for d in del_event:
                    new_trace.append(d)
                del_event = []
                new_trace.append(trace[g])
                g = g + 1

        elif w[:8] == "[M-REAL]" or w[:8] == "[M-INVI]":
            event = Event()
            event['org:resource'] = 'NONE'
            event['lifecycle:transition'] = 'complete'
            event['concept:name'] = w[8:]
            del_event.append(event)

    if k > 1:

        new_log = EventLog()
        new_log.append(new_trace)

        replayed_traces = token_replay.apply(new_log, net, initial_marking, final_marking, parameters={"try_to_reach_final_marking_through_hidden":False})
        # print("Token-based Replay: ", replayed_traces)
        write_outputfile("Token-based Replay:  " + str(replayed_traces), pattern, namesub, "a")
        #count=0
        #for u in replayed_traces[0]['activated_transitions']:
        #    count = count + 1

        #if(count>g):
        #    print(trace[g-1]['concept:name'])
        #    for tr in net.transitions:
        #        if(tr.label==trace[g-1]['concept:name']):
        #            for pl in net.places:
        #                for arc in pl.out_arcs:
        #                    if arc.target.name == tr.name:
        #                        reached_marking.append(pl.name)

        #    print('Correct Reached Marking ---- >',reached_marking)
        #else:
        for v in replayed_traces[0]['reached_marking']:
            reached_marking.append(v.name)
    else:
        reached = i_marking
        reached_marking.append(reached.split(":")[0])

    return reached_marking


""" The function searches in the input alignment the position of the end transition of the graph, counting synchronous/log moves. 
    For each end node, it takes all transitions before the identified point, only considering moves on model and synchronous moves. Then it applies a
    token-based replay, obtaining the reached_marking.
INPUT: -pattern: the folder containing the files
       -dataset: the dataset name
       -trace: the trace name
       -end: a list of start nodes in the graph
RETURN: -reached_marking: a dictionary with pairs 'end':'marking'

"""


def dirk_marking_end(dataset, end, text, trace, pattern, sub):
    net, initial_marking, final_marking = pnml_importer.apply(os.path.join(pattern, dataset + '_petriNet.pnml'))
    reached_marking = []
    m = str(massimo_lista(end))

    k = 0
    new_trace = Trace(attributes=trace.attributes)
    g = 0
    for w in text:

        if w[:3] == "[L]":

            k = k + 1
            if k > int(m):
                break
            g = g + 1

        elif w[:5] == "[L/M]":
            k = k + 1

            if k > int(m):
                break

            new_trace.append(trace[g])
            g = g + 1

        elif w[:8] == "[M-REAL]" or w[:8] == "[M-INVI]":
            event = Event()
            event['org:resource'] = 'NONE'
            event['lifecycle:transition'] = 'complete'
            event['concept:name'] = w[8:]
            new_trace.append(event)

    if k > 1:

        new_log = EventLog()
        new_log.append(new_trace)
        replayed_traces = token_replay.apply(new_log, net, initial_marking, final_marking, parameters={"try_to_reach_final_marking_through_hidden":False})
        # print("Token-based Replay: ", replayed_traces)
        write_outputfile("Token-based Replay:  " + str(replayed_traces), pattern, sub, "a")

        #count = 0
        #for u in replayed_traces[0]['activated_transitions']:
        #    count = count + 1

        #if (count > g):
        #    print(trace[g - 1]['concept:name'])
        #    for tr in net.transitions:
        #        if (tr.label == trace[g - 1]['concept:name']):
        #            for pl in net.places:
        #                for arc in pl.out_arcs:
        #                    if arc.target.name == tr.name:
        #                        reached_marking.append(pl.name)

        #    print('Correct Reached Marking ---- >', reached_marking)
        #else:
        for v in replayed_traces[0]['reached_marking']:
            reached_marking.append(v.name)

    return reached_marking


""" The functions simplifies a sub, by removing the part that cannot be followed. 
By checking the alignment the sub is cut until a move on model/log is found
INPUT: -start: a list with the start nodes of the sub
       -text: the alignment
       -subgraph: the output of find_instances()
RETURN: a subgraph
"""


def start_pre_process_repairing(start, text, subgraph):
    m = minimo_lista(start)
    k = 1
    del_event = []

    for w in text:

        if k >= int(m):
            if w[:3] == "[L]":
                break
            elif w[:5] == "[L/M]":
                del_event.append(w[5:])
            elif w[:8] == "[M-REAL]" or w[:8] == "[M-INVI]":
                break

        elif w[:3] == "[L]":
            k = k + 1
        elif w[:5] == "[L/M]":
            k = k + 1
        elif w[:8] == "[M-REAL]" or w[:8] == "[M-INVI]":
            continue

    number = []
    for y in del_event:
        j = 0
        for p in subgraph:
            if y == p:
                p1 = j - 2
                p2 = j + 1
                number.append(subgraph[j - 1])
                del subgraph[p1:p2]
                break
            j = j + 1

    for n in number:
        j = 0
        for q in range(len(subgraph)):
            if subgraph[j] == 'Found':
                break
            elif subgraph[j] == 'd' and (subgraph[j + 1] == n or subgraph[j + 2] == n):
                del subgraph[j:j + 4]
            else:
                j = j + 1

    return subgraph


"""
 The functions simplifies a sub, by removing the part that cannot be followed. 
By checking the alignment the sub is cut from the end until a move on model/log is found
INPUT: -end: a list with the end nodes of the sub
       -text: the alignment
       -subgraph: the output of find_instances()
RETURN: a subgraph

NOTE: 07/21 bug fix: if the final node is [L/M] and before there is a [M], the end node was not correctly deleted.
"""


def end_pre_process_repairing(end, text, subgraph):
    m = massimo_lista(end)
    k = 1
    x = 0
    q = 0 #mi serve per controllare che quando k==m siamo sul nodo finale corretto.
    del_event = []

    for w in text:
        if k == m:
            while k == m:
                if text[x][:3] == "[L]":
                    q = q +1
                    break
                elif text[x][:5] == "[L/M]":
                    del_event.append(text[x][5:])
                    x = x - 1
                    q = q +1
                elif text[x][:8] == "[M-REAL]" or text[x][:8] == "[M-INVI]":
                    if q == 0:
                        x = x + 1
                        continue
                    break
            break
        elif w[:3] == "[L]":
            k = k + 1
        elif w[:5] == "[L/M]":
            k = k + 1
        elif w[:8] == "[M-REAL]" or w[:8] == "[M-INVI]":
            x = x + 1
            continue

        x = x + 1

    number = []
    subgraph.reverse()
    for y in del_event:
        j = 0
        for p in subgraph:
            if y == p:
                p1 = j
                p2 = j + 3
                number.append(subgraph[j + 1])
                del subgraph[p1:p2]
                break
            j = j + 1

    subgraph.reverse()
    for n in number:
        j = 0
        for q in range(len(subgraph)):
            if subgraph[j] == 'Found':
                break
            elif subgraph[j] == 'd' and (subgraph[j + 1] == n or subgraph[j + 2] == n):
                del subgraph[j:j + 4]
            else:
                j = j + 1

    return subgraph


""" The function create the Petri net of the input subgraph and returns the start/end Transition objects in two dictionaries
INPUT: -subgraph: the output of find_instances()
       -net: the net model
       -start: list of the sub's start nodes
       -end: list of the sub's start nodes
RETURN: -start_result, end_result: dictionaries 'number_node_start':'corresponding_object_Transitions'
"""


def create_sub_petrinet(subgraph, net, start, end, pattern, sub, added_components):
    transitions = net.transitions
    places = net.places
    place_number = []
    arc = []
    place_dict = {}
    trans_dict = {}

    for x in range(len(subgraph)):
        if subgraph[x] == "Found":
            break
        elif subgraph[x] == 'd' or subgraph[x] == 'e':
            arc.append((subgraph[x + 1], subgraph[x + 2]))
            if subgraph[x + 2] not in place_number:
                n = places_name_available(places, transitions)
                place = PetriNet.Place("n" + n)
                net.places.add(place)
                added_components.append(place)
                place_dict[subgraph[x + 2]] = place
                place_number.append(subgraph[x + 2])
            else:
                n = places_name_available(places, transitions)
                place = PetriNet.Place("n" + n)
                net.places.add(place)
                added_components.append(place)
                place_dict[str(subgraph[x + 1]) + str(subgraph[x + 2])] = place
        elif subgraph[x] == 'v':
            n = transition_name_available(transitions)
            trans = PetriNet.Transition("s" + n, subgraph[x + 2])
            net.transitions.add(trans)
            added_components.append(trans)
            trans_dict[subgraph[x + 1]] = trans

    for y in arc:
        ap = str(y[0]) + str(y[1])
        if y[1] in place_number:
            utils.add_arc_from_to(trans_dict[y[0]], place_dict[y[1]], net)
            write_outputfile(
                "Added:  " + str(trans_dict[y[0]].label) + " " + str(trans_dict[y[0]].name) + " --> " + str(
                    place_dict[y[1]]), pattern, sub, "a")
            added_arc = [arc for arc in net.arcs if arc.source == trans_dict[y[0]] and arc.target == place_dict[y[1]]]
            added_components.extend(added_arc)
            # print("Added: ", trans_dict[y[0]].label, trans_dict[y[0]].name, "-->", place_dict[y[1]])
            utils.add_arc_from_to(place_dict[y[1]], trans_dict[y[1]], net)
            write_outputfile("Added:  " + str(place_dict[y[1]]) + " --> " + str(trans_dict[y[1]].label) + " " + str(
                trans_dict[y[1]].name), pattern, sub, "a")
            added_arc = [arc for arc in net.arcs if arc.source == place_dict[y[1]] and arc.target == trans_dict[y[1]]]
            added_components.extend(added_arc)
            # print("Added: ",place_dict[y[1]], "-->", trans_dict[y[1]].label, trans_dict[y[1]].name)
            place_number.remove(y[1])
        elif ap in place_dict:
            utils.add_arc_from_to(trans_dict[y[0]], place_dict[ap], net)
            write_outputfile(
                "Added:  " + str(trans_dict[y[0]].label) + " " + str(trans_dict[y[0]].name) + " --> " + str(
                    place_dict[ap]), pattern, sub, "a")
            added_arc = [arc for arc in net.arcs if arc.source == trans_dict[y[0]] and arc.target == place_dict[ap]]
            added_components.extend(added_arc)
            # print("Added: ", trans_dict[y[0]].label, trans_dict[y[0]].name, "-->", place_dict[ap])
            utils.add_arc_from_to(place_dict[ap], trans_dict[y[1]], net)
            write_outputfile("Added:  " + str(place_dict[ap]) + " --> " + str(trans_dict[y[1]].label) + " " + str(
                trans_dict[y[1]].name), pattern, sub, "a")
            added_arc = [arc for arc in net.arcs if arc.source == place_dict[ap] and arc.target == trans_dict[y[1]]]
            added_components.extend(added_arc)
            # print("Added: ",place_dict[ap], "-->", trans_dict[y[1]].label, trans_dict[y[1]].name)

    start_result = {}
    end_result = {}
    start_result['start'] = trans_dict[minimo_lista(start)].name
    end_result['end'] = trans_dict[str(massimo_lista(end))].name
    for s in start:
        start_result[s] = trans_dict[s]
    for e in end:
        end_result[e] = trans_dict[e]

    return start_result, end_result


""" The function repairs the subgraph with the net
INPUT: -subgraph: the output of find_instances()
       -net: the net model
       -start: list of the sub's start nodes
       -end: list of the sub's start nodes
       -start_marking: the output of dirk_marking_start()
       -end_marking: the output of  dirk_marking_end()
"""


def repairing(subgraph, net, initial_marking, final_marking, start, end, start_marking, end_marking, pattern, sub):
    added_components = []
    start_trans, end_trans = create_sub_petrinet(subgraph, net, start, end, pattern, sub, added_components)

    places = net.places
    transitions = net.transitions

    if len(start) > 1:
        n = transition_hidden_available(transitions)
        t = PetriNet.Transition("h" + n, None)
        net.transitions.add(t)
        added_components.append(t)
        for v in start_marking:
            for place in net.places:
                if place.name == v:
                    utils.add_arc_from_to(place, t, net)
                    write_outputfile("Added: " + str(place) + " -- > " + str(t) + " " + str(t.name), pattern, sub, "a")
                    added_arc = [arc for arc in net.arcs if arc.source==place and arc.target==t]
                    added_components.extend(added_arc)
                    # print("Added: ", place, " -- > ", t, t.name)
        for x in start:
            n = places_name_available(places, transitions)
            place = PetriNet.Place("n" + n)
            net.places.add(place)
            added_components.append(place)
            utils.add_arc_from_to(t, place, net)
            write_outputfile("Added:  " + str(t) + " " + str(t.name) + " -- > " + str(place), pattern, sub, "a")
            added_arc = [arc for arc in net.arcs if arc.source == t and arc.target == place]
            added_components.extend(added_arc)
            # print("Added: ", t, t.name, " -- > ", place)
            utils.add_arc_from_to(place, start_trans[x], net)
            write_outputfile("Added: " + str(place) + " -- > " + str(start_trans[x]) + " " + str(start_trans[x].name),
                             pattern, sub, "a")
            added_arc = [arc for arc in net.arcs if arc.source == place and arc.target == start_trans[x]]
            added_components.extend(added_arc)
            # print("Added: ", place, " -- > ", start_trans[x], start_trans[x].name)
    else:
        for v in start_marking:
            for place in net.places:
                if place.name == v:
                    utils.add_arc_from_to(place, start_trans[start[0]], net)
                    write_outputfile("Added: " + str(place) + " -- > " + str(start_trans[start[0]]) + " " + str(
                        start_trans[start[0]].name), pattern, sub, "a")
                    added_arc = [arc for arc in net.arcs if arc.source == place and arc.target == start_trans[start[0]]]
                    added_components.extend(added_arc)
                    # print("Added: ", place, " -- > ", start_trans[start[0]], start_trans[start[0]].name)

    if len(end) > 1:
        n = transition_hidden_available(transitions)
        t = PetriNet.Transition("h" + n, None)
        net.transitions.add(t)
        added_components.append(t)
        for y in end_marking:
            for place in net.places:
                if place.name == y:
                    utils.add_arc_from_to(t, place, net)
                    write_outputfile("Added: " + str(t) + " " + str(t.name) + " --> " + str(place), pattern, sub, "a")
                    added_arc = [arc for arc in net.arcs if arc.source == place and arc.target == t]
                    added_components.extend(added_arc)
                    # print("Added: ", t, t.name, " --> ", place)
        for x in end:
            n = places_name_available(places, transitions)
            place = PetriNet.Place("n" + n)
            net.places.add(place)
            added_components.append(place)
            utils.add_arc_from_to(place, t, net)
            write_outputfile("Added: " + str(place) + " -- > " + str(t), pattern, sub, "a")
            added_arc = [arc for arc in net.arcs if arc.source == place and arc.target == t]
            added_components.extend(added_arc)
            # print("Added: ", place, " -- > ", t)
            utils.add_arc_from_to(end_trans[x], place, net)
            write_outputfile("Added: " + str(end_trans[x]) + " " + str(end_trans[x].name) + " -- > " + str(place),
                             pattern, sub, "a")
            added_arc = [arc for arc in net.arcs if arc.source == end_trans[x] and arc.target == place]
            added_components.extend(added_arc)
            # print("Added: ", end_trans[x], end_trans[x].name, " -- > ", place)
    else:
        for v in end_marking:
            for place in net.places:
                if place.name == v:
                    utils.add_arc_from_to(end_trans[end[0]], place, net)
                    write_outputfile(
                        "Added: " + str(end_trans[end[0]]) + " " + str(end_trans[end[0]].name) + " -- > " + str(place),
                        pattern, sub, "a")
                    added_arc = [arc for arc in net.arcs if arc.source == end_trans[end[0]] and arc.target == place]
                    added_components.extend(added_arc)
                    # print("Added: ", end_trans[end[0]], end_trans[end[0]].name, " -- > ", place)

    start_end_trans = [start_trans['start'], end_trans['end']]
    return start_end_trans, net, added_components


""" The function repairs the model adding a hidden transition between the last nodes of the subs and the model and the end node
INPUT: -net: the net model
       -arcs: set of arcs in the net
       -places: set of places in the net
       -transitions: set of transitions in the net
"""


""" The function finds the position of the alignment in the sub
INPUT: -al: an alignment
       -start: the name of the start transition in the sub
       -end: the name of the end transition in the sub
RETURN: -pos_start: the index of the alignment list corresponding to the start node
        -pos_end: the index of the alignment list corresponding to the start node
"""


def pos_node_alignment(al, start, end):
    i = 0
    pos_start = 0
    pos_end = 0

    for a in al[0]['alignment']:
        if a[0][1] == end and a[0][1] == start:
            pos_end = i
            pos_start = i
            break
        elif a[0][1] == end:
            pos_end = i
            break
        elif a[0][1] == start:
            pos_start = i
        i = i + 1

    return pos_start, pos_end


""" The function removes an arc from a Petri net
INPUT: -net: Petri net
       -arc: Arc of the Petri net
RETURNS: -net: Petri net
"""


def remove_arc(net, arc):
    net.arcs.remove(arc)
    arc.source.out_arcs.remove(arc)
    arc.target.in_arcs.remove(arc)

    return net


""" The function checks that the repairment is necessary, in case the outgoing arcs of the two hidden transition (h1,h2) 
are directed to the same places
INPUT: -trans: a transition 
       -tr: a transition
       -places: list of places of the net
RETURN: -True: if repairment with the hidden transition is necessary
        -False: if repairment is not necessary
"""


def check_rep_ltrans(c_trans, c_tr, c_places):
    h1_add=[]
    h2_add=[]
    result = True
    for c_place in c_places:
        for arc in c_place.out_arcs:
            if arc.target.name == c_trans.name:
                h1_add.append(c_place.name)
        for at in c_place.in_arcs:
            if at.source.name == c_tr.name:
                h2_add.append(c_place.name)
    if (len(h1_add) == len(h2_add)):
        i = 0
        for p in h1_add:
            if p in h2_add:
                i = i + 1
        if i == len(h1_add):
            result = False

    return result



""" The function repairs the model a second time in order to make the input trace fit, 
fixing the move on log step by step
INPUT: -trace: the target trace
       -start: name of the start transition
       -end: name of the end transition
       -net: the net model
       -initial_marking: initial marking of the model
       -final_marking: ending marking of the model
       -sub: list of transition labels for the sub
       -pattern: path for 'patterns_file'
       -nsub: number of the sub 
RETURN: -'U': if the sub is perfectly fitting with no need to add arcs
        -'UA': some arcs have been added to make the sub fitting
        -'UNA': the sub was not followed in the alignment
        -'UNG': the case was not managed
"""


""" The function repairs the model a second time connecting the first transition of the sub to the places enabling
the transition of the first move on log in the alignment. This is done by adding an arc going from the places in which the firing of the transition (of the last move on log in the alignment) put a token  
    to the next transition in the log.
INPUT: -trace: the target trace
       -start: name of the start transition
       -end: name of the end transition
        -net: the net model
      -initial_marking: initial marking of the model
       -final_marking: ending marking of the model
       -sub: list of the labels for the transitions in the sub
RETURN: -'U': if the sub is perfectly fitting with no need to add arcs
        -'UA': some arcs have been added to make the sub fitting
        -'UNA': the sub was not followed in the alignment
"""


"""Computes Precision, Fitness, Generalization and Simplicity w.r.t. an Event Log composed by graphs in which the sub occurs
INPUT: -graph_list: list of graphs in which the sub occurs
       -log: Event Log
       -dict_trace: dictionary with pairs idTrace:numTrace
       -net: the net model
       -initial_marking: initial marking of the model
       -final_marking: final marking of the model 
"""


def valutazione_rete(graph_list, log, dict_trace, net, initial_marking, final_marking, pattern, sub):
    new_eventlog = EventLog()
    for graph in graph_list:
        traccia = search_trace(log, dict_trace, graph)
        new_eventlog.append(traccia)

    # xes_exporter.apply(new_eventlog,'testlog.xes')

    fitness = replay_evaluator.apply(new_eventlog, net, initial_marking, final_marking,
                                     variant=replay_evaluator.Variants.ALIGNMENT_BASED)
    write_outputfile("Fitness:  " + str(fitness), pattern, sub, "a")
    # print("Fitness: ", fitness)
    precision = precision_evaluator.apply(new_eventlog, net, initial_marking, final_marking,
                                          variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    write_outputfile("Precision:  " + str(precision), pattern, sub, "a")
    # print("Precision: ", precision)
    generalization = generalization_evaluator.apply(new_eventlog, net, initial_marking, final_marking)
    write_outputfile("Generalization:  " + str(generalization), pattern, sub, "a")
    # print("Generalization: ", generalization)
    simplicity = simplicity_evaluator.apply(net)
    write_outputfile("Simplicity:  " + str(simplicity), pattern, sub, "a")
    # print("Simplicity: ", simplicity)


"""Computes Precision, Fitness, Generalization and Simplicity w.r.t. a complete Event Log
INPUT: -log: Event Log
       -net: the net model
       -initial_marking: initial marking of the model
       -final_marking: final marking of the model 
"""

def valutazione_rete_logcompleto(log, net, initial_marking, final_marking, pattern, sub):
    fitness = replay_evaluator.apply(log, net, initial_marking, final_marking,
                                     variant=replay_evaluator.Variants.ALIGNMENT_BASED)

    write_outputfile("Fitness:  " + str(fitness), pattern, sub, "a", '_evaluation')
    #print("Fitness: ", fitness)
    precision = precision_evaluator.apply(log, net, initial_marking, final_marking,
                                          variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)

    write_outputfile("Precision:  " + str(precision), pattern, sub, "a", '_evaluation')
    #print("Precision: ", precision)
    generalization = generalization_evaluator.apply(log, net, initial_marking, final_marking)

    write_outputfile("Generalization:  " + str(generalization), pattern, sub, "a", '_evaluation')
    #print("Generalization: ", generalization)
    simplicity = simplicity_evaluator.apply(net)

    write_outputfile("Simplicity:  " + str(simplicity), pattern, sub, "a", '_evaluation')
    #print("Simplicity: ", simplicity)


""" The function shows a Petri net
INPUT: -log: Event Log
       -net: the model net
       -initial_marking: initial marking of the model
       -final_marking: final marking of the model
"""


def visualizza_rete(log, net, im, fm):
    parameters = {petrinet_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "jpg", "debug": True}
    gvz = token_decoration_frequency.apply(net, im, fm, parameters=parameters)
    petrinet_visualizer.view(gvz)


""" The function shows a Petri net with performances
INPUT: -log: Event Log
       -net: the model net
       -initial_marking: initial marking of the model
       -final_marking: final marking of the model
"""

from pm4py.visualization.petri_net import visualizer as petrinet_visualizer
from pm4py.objects.petri_net.obj import PetriNet

from pm4py.visualization.petri_net import visualizer as petrinet_visualizer

def visualizza_rete_performance(log, net, im, fm, added_components):
    parameters = {
        petrinet_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "jpg",
        "debug": True
    }

    gvz = petrinet_visualizer.apply(net, im, fm, parameters=parameters)

    # Dividi added_components in label e archi (tuple)
    node_labels = {x.name if x.name is not None else x.label for x in added_components if isinstance(x, (PetriNet.Place, PetriNet.Transition))}
    arc_labels = {x for x in added_components if isinstance(x, PetriNet.Arc)}

    id_mapping = {}
    id_trans = {}

    new_body = []
    for line in gvz.body:
        for label in node_labels:
            if f'label={label} ' in line or f'label="{label}"' in line:
                id_mapping[label] = line.split()[0]
        for t in set.union(net.places, net.transitions):
            label = t.name
            if f'label={label} ' in line or f'label="{label}"' in line:
                id_trans[label] = line.split()[0]

    for line in gvz.body:
        for label in node_labels:
            if f'label={label} ' in line or f'label="{label}"' in line:
                if 'fillcolor=white' in line:
                    line = line.replace('fontcolor=black', 'fontcolor=red')
                    line = line.replace(']', ' color=red]')
                if 'fillcolor=black' in line:
                    line = line.replace('fillcolor=black', 'fillcolor=red')
                else:
                    line = line.replace('shape', 'fontcolor=red shape')
                line = line.replace(']', ' color=red]')
                line = line.replace('border=1', 'border=2')


        for a in arc_labels:
            src_label = a.source.name
            dst_label = a.target.name
            if src_label in id_trans and dst_label in id_trans:
                if f'{id_trans[src_label]} -> {id_trans[dst_label]}' in line:
                    line = line.replace('[', '[color=red penwidth=3 ')
        new_body.append(line)

    for t in net.transitions:
        if t.label is not None:
            new_body = [line.replace(t.name+' ', '"'+str(t.label)+' "') for line in new_body]
    gvz.body = new_body
    petrinet_visualizer.view(gvz)
    return gvz




""" The function exports an event log with the traces in which the sub occurs
INPUT: -graph_list: a list of graphs in which the sub occurs
       -log:  Event Log
       -dict_trace: dictionary with pairs idTrace:numTrace
       -sub: the number of the sub
"""


def export_eventlog_test(graph_list, log, dict_trace, sub):
    new_eventlog = EventLog()
    for gra in graph_list:
        tra = search_trace(log, dict_trace, gra)
        new_eventlog.append(tra)

    xes_exporter.apply(new_eventlog, os.path.join('..', 'testlog_'+ sub + '.xes'))


""" The functions performs the alignment of each trace in which the sub occurs
INPUT: -graph_list: a list of graphs in which the sub occurs
       -log:  Event Log
       -dict_trace: dictionary with pairs idTrace:numTrace
       -net: the net model
       -initial_marking: initial marking of the model
       -final_marking: final marking of the model 
"""


def all_alignment(graph_list, log, dict_trace, net, initial_marking, final_marking):
    print("Alignment of all traces in which the following sub occurs: ")
    for graph in graph_list:
        traccia = search_trace(log, dict_trace, graph)
        new_eventlog = EventLog()
        new_eventlog.append(traccia)

        align = alignment.apply_log(new_eventlog, net, initial_marking, final_marking)
        print("New Alignment " + traccia.attributes['concept:name'] + ": ", align)


""" The function cut the input graph to write it in the graph+n+.g file that can be passed to the tool gm
INPUT: -pattern: the folder containing files
       -graph: the graph name
       -subnumber: the number of the sub
RETURN: -n_sub: the graph part containing the sub to write on the file
"""


def graph_sub(pattern, graph, sub_number):
    sub = graph.split('\n')
    sub = [s for s in sub if s != '']
    sub = [s.split(' ') for s in sub]
    sub = list(flatten(sub))
    sub = [s for s in sub if s != '']


    # esegue sgiso e ritorna la sub con i nodi rispetto al grafo
    subgraph = find_instances(graph, pattern)

    # ritorna i nodi di inizio e fine sub
    start, end, sub_label = startend_node(subgraph)

    n_sub = []

    for x in range(len(sub)):
        if sub[x] == 'v':
            if int(sub[x + 1]) >= int(minimo_lista(start)) and int(sub[x + 1]) <= massimo_lista(end):
                n_sub.append(sub[x])
                n_sub.append(sub[x + 1])
                n_sub.append(sub[x + 2])
        elif sub[x] == 'd' or sub[x] == 'e':
            if int(minimo_lista(start)) <= int(sub[x + 1]) <= massimo_lista(end) and int(
                    minimo_lista(start)) <= int(sub[x + 2]) <= massimo_lista(end):
                n_sub.append(sub[x])
                n_sub.append(sub[x + 1])
                n_sub.append(sub[x + 2])
                n_sub.append(sub[x + 3])
    return n_sub


""" The function runs the gm tool to calculate the matching cost between the two input graphs
INPUT: -graph1: name of the first graph
       -graph1: name of the second graph
        - sub_number: the number of the sub
RETURN: -float(sub2[3]): Matching Cost
"""


def graph_matching(pattern, graph1, graph2, sub_number):
    secondgraph = graph_sub(pattern, graph2, sub_number)
    write_graphfile(secondgraph, "2", pattern)

    if graph1 == 'sub':
        out = subprocess.Popen([os.path.join('subdue_files','gm'),
                                os.path.join(pattern, 'graphsub.g'),
                                os.path.join(pattern, 'graph2.g')],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        sub = stdout.decode("utf-8")
        sub2 = sub.split()
    else:
        firstgraph = graph_sub(pattern, graph1, sub_number)
        write_graphfile(firstgraph, "1", pattern)
        out = subprocess.Popen([os.path.join(pattern, 'gm'),
                                os.path.join(pattern, 'graph1.g'),
                                os.path.join(pattern, 'graph2.g')],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        sub = stdout.decode("utf-8")
        sub2 = sub.split()

    return float(sub2[3])


""" The functions return the graph with the smallest matching cost
INPUT: -pattern: the folder containing files
       -graph: the graph name
       -graph_list: list of graphs in which the sub occurs
       -sub_number: the sub number
RETURN: -mingraph: list of graphs with the smallest cost
        -mincost: the smallest cost
"""


def graph_choice(pattern, graph, graph_list, sub_number):
    mincost = graph_matching(pattern, graph, list(graph_list.values())[0], sub_number)
    mingraph = []
    for x in graph_list:
        cost = graph_matching(pattern, graph, graph_list[x], sub_number)
        if cost < mincost:
            mincost = cost
            mingraph = []
            mingraph.append(x)
        elif cost == mincost:
            mingraph.append(x)

    return mingraph, mincost


""" The function creates the dictionaty with the ranking of the graphs matching the sub, ordered by increasing cost
INPUT: -pattern: the folder containing the files
       -graph: the graph name
       -graph_list: list of graphs in which the sub occurs
       -sub_number: the sub number
RETURN: -dict: dictionary with the ranking of the graphs based on the matching cost
"""


def create_dict_graph(pattern, graph, graph_list, sub_number):
    bar = IncrementalBar('Create dict_graph: ', max=len(graph_list))
    dict = {}
    i = 1
    list = []
    list = graph_list.copy()
    while len(list) != 0:
        min_graph, cost = graph_choice(pattern, graph, list, sub_number)
        for y in min_graph:
            bar.next()
            list.pop(y)
            dict[i] = (y, cost)
            i = i + 1
    bar.finish()

    return dict


""" The functions returns the graph with the smallest matching cost
INPUT: -graph: the graph name
       -graph_dict: dictionary including the output of create_dict_graph
       -log: Event Log
       -dict_trace: dictionary with pairs 'numTrace':'idTrace'
       -start_name: name of the start transition
       -end_name: name of the end transition
       -net: the net model
       -initial_marking: initial marking of the model
       -final_marking: final marking finale of the model
       -sub: list of the labels of the transitions in the sub
"""



def process_g_file(pattern, dataset):
    graphs = []
    with open(os.path.join(pattern, dataset+'.g'), 'r') as f:
        graphs = [g for g in f.read().split('XP')]
    if '' in graphs:
        graphs.remove('')
    """
    graphs = [g.split('\n') for g in graphs]
    new_graphs = []
    for g in graphs:
        gr = []
        for s in g:
            if s.strip():  # ignora stringhe vuote
                gr.extend(s.split())
        new_graphs.append(gr)
    return new_graphs
    """
    return graphs

def check_plaintext(path):
    PATTERN = re.compile(
        r"""^Fitness:\s*{\s*            # riga Fitness
            'percFitTraces':\s*-?\d+(\.\d+)?,\s*
            'averageFitness':\s*-?\d+(\.\d+)?,\s*
            'percentage_of_fitting_traces':\s*-?\d+(\.\d+)?,\s*
            'average_trace_fitness':\s*-?\d+(\.\d+)?,\s*
            'log_fitness':\s*-?\d+(\.\d+)?\s*}\s*\n
            Precision:\s*-?\d+(\.\d+)?\s*\n
            Generalization:\s*-?\d+(\.\d+)?\s*\n
            Simplicity:\s*-?\d+(\.\d+)?\s*$""",
        re.VERBOSE | re.MULTILINE,
    )
    text = pathlib.Path(path).read_text().strip()
    return bool(PATTERN.match(text))



def main(input_data, pattern, dataset, numsub, namesub):
    debug = False

    print('Start repairing..')
    # Event log
    log = xes_importer.apply(os.path.join(pattern, dataset + '.xes'))
    # Model
    net, initial_marking, final_marking = pnml_importer.apply(os.path.join(pattern, dataset + '_petriNet.pnml'))
    
    # net, initial_marking, final_marking = pnml_importer.apply(pattern + '/reti_Fahland/repaired_'+str(x)+'.pnml')

    # given the pattern number, return the list of subs
    # lista = list_sub_pattern(pattern + dataset + "_new_patterns_filtered.subs", 2)
    # print("Pattern: ", lista)
    # write_outputfile("Pattern:  " + str(lista), pattern, sub, "w")

    sub = numsub  # lista[0] if we take the sub from the list of patterns

    graphs_indexes = ['graph' + str(n + 1) for n in range(0, len(log))]
    trace_names = [t.attributes['concept:name'] for t in log]

    dict_trace = dict(zip(graphs_indexes, trace_names))

    graphs = process_g_file(pattern, dataset)
    graphs = dict(zip(graphs_indexes, graphs))

    new_graph_list = check_graphlist(input_data, graphs, sub, pattern)
    write_outputfile("Fast mode: " + str(input_data['fast_mode']), pattern, namesub, "a")
    write_outputfile("Number of graphs in which the sub occurs: " + str(len(new_graph_list)), pattern, namesub, "a")
    dict_graph = create_dict_graph(pattern, "sub", new_graph_list, sub)
    graph = dict_graph[1][0]
    # print("Graph Selected: ", graph, " Matching Cost: ", dict_graph[1][1])
    write_outputfile("Graph Selected:  " + str(graph) + "  Matching Cost:  " + str(dict_graph[1][1]), pattern, namesub, "a")

    write_outputfile("\nEvaluation initial net:", pattern, namesub, "a")
    #print("\nValutazione rete sub_" + str(x) + ":")
    # evaluation of the log composed by only traces in which the sub occurs
    # evaluation on the complete log
    if os.path.exists(os.path.join(pattern, 'output__evaluation.txt')):
        if not check_plaintext(os.path.join(pattern, 'output__evaluation.txt')):
                valutazione_rete_logcompleto(log, net, initial_marking, final_marking, 'original', pattern)

    # visualization of the net
    # visualizza_rete_performance(log, net, initial_marking, final_marking, [])

    # executes sgiso and returns the sub with the nodes w.r.t the graph
    subgraph = find_instances(new_graph_list[graph], pattern)
    # print("Subgraph: ", subgraph)
    write_outputfile("Subgraph:  " + str(subgraph), pattern, namesub, "a")

    # create the event log with the traces in which the sub occurs
    export_eventlog_test(new_graph_list, log, dict_trace, sub)

    # returns the start and end nodes
    start, end, sub_label = startend_node(subgraph)
    # print("Sub iniziale: ", sub_label)
    write_outputfile("Initial sub:  " + str(sub_label), pattern, namesub, "a")

    # Trace
    trace = search_trace(log, dict_trace, graph)
    # print('Trace: ', trace.attributes['concept:name'])
    write_outputfile('Trace:  ' + str(trace.attributes['concept:name']), pattern, namesub, "a")

    # Alignment
    text = search_alignment(pattern, dict_trace, graph, dataset)
    # print('Alignment: ' + text)
    write_outputfile('Alignment: ' + str(text), pattern, namesub, "a")

    # Pre-filtering of the sub already present in the model
    new_subgrap = start_pre_process_repairing(start, text, subgraph)
    new_subgraph = end_pre_process_repairing(end, text, new_subgrap)
    # print("Subgraph semplificata: ", new_subgraph)
    write_outputfile("Simplified subgraph:  " + str(new_subgraph), pattern, namesub, "a")

    # returns the start and end nodes
    start, end, sub_label = startend_node(new_subgraph)
    # print("Sub semplificata: ", sub_label)
    write_outputfile("Simplified sub:  " + str(sub_label), pattern, namesub, "a")

    if sub_label == []:
        raise IndexError

    # print("Start: ", minimo_lista(start))
    write_outputfile("Start:  " + str(minimo_lista(start)), pattern, namesub, "a")
    # returns the places where to attach the start nodes
    reached_marking_start = dirk_marking_start(dataset, start, text, trace, pattern, namesub)
    # print("Reached Marking: ", reached_marking_start)
    write_outputfile("Reached Marking:  " + str(reached_marking_start), pattern, namesub, "a")

    # print("End: ", massimo_lista(end))
    write_outputfile("End:  " + str(massimo_lista(end)), pattern, namesub, "a")
    # returns the place where to attach the end nodes
    reached_marking_end = dirk_marking_end(dataset, end, text, trace, pattern, namesub)
    # print("Reached Marking: ", reached_marking_end)
    write_outputfile("Reached Marking:  " + str(reached_marking_end), pattern, namesub, "a")

    if not new_subgrap:
        raise ValueError
    # tempo1 = timer()
    # repair the model with the subgraph
    start_end_name, net_repaired, added_components = repairing(new_subgraph, net, initial_marking, final_marking, start, end,
                                             reached_marking_start, reached_marking_end, pattern, namesub)

    gvz = visualizza_rete_performance(log, net, initial_marking, final_marking, added_components)
    valutazione_rete_logcompleto(log, net_repaired, initial_marking, final_marking, pattern, namesub)

    rete = [net_repaired, initial_marking, final_marking]

    pnml_exporter.apply(rete[0], rete[1], os.path.join(namesub, "repaired_petriNet.pnml"), final_marking=rete[2])
    gvz.render(filename= os.path.join(namesub, "repaired_petriNet"), format='jpg', cleanup=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Model Repair Supported by Frequent Anomalous Local Instance Graphs")
    #parser.add_argument("path", type=str,
      #                  help="Path della directory contenente: *_table2_on_file.csv | *_new_patterns_filtered.subs | rules_log.txt")
    parser.add_argument("datasetname", type=str, help="Name of the dataset to analyse")
    parser.add_argument("numsub", type=str, help="Number of the sub with which the model is to be repaired")
    args = parser.parse_args()
    main("../patterns_file/", args.datasetname, args.numsub, args.namesub) #BPI2017Denied, testBank2000NoRandomNoise

    #main("../patterns_file/", "fineExp", "59")

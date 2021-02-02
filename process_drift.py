from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.traces.log import case_statistics
import statsmodels.api as sm 
from graphviz import Source
import matplotlib.pyplot as plt
import ruptures as rpt
import pandas as pd
import pm4py
import numpy as np
import re
import datetime
import os

# CONSTANTS
# Input and output files
INPUT_XES_FILE = 'data/italian_help_desk.xes'
OUTPUT_GRAPH_FILE = 'merged_dfg'

# Change point detection
PENALTY = 4
MODEL = "rbf"

# Period filtering
CHOSEN_PERIOD1 = 4
CHOSEN_PERIOD2 = 6

# Name of the variable in .xes, which contains the activity labels for the graph
ACTIVITY_NAMES = 'concept:name'

# ---------------------------------------------------------------------------

#Function to convert the DOT output source code to a dataframe of the graph
def dot_to_df(gviz):

    node_name = []
    node_l = []
    node_occ = []

    edge_from = []
    edge_to = []
    edge_label = []

    for x in gviz.source.split('\n'):
        #print(x)
        node = re.search('\t([-]{0,1}[0-9]+) \[label',x)
        if node is not None:
            node = node.groups()[0]

        node_label = re.search('\[label="([a-zA-Z0-9 ]*)',x)
        if node_label is not None:
            node_label = node_label.groups()[0]

        node_occurrence = re.search('\[label="[a-zA-Z0-9 ]*\(([0-9]*)\)',x)
        if node_occurrence is not None:
            node_occurrence = node_occurrence.groups()[0]

        edge_f = re.search('\t([-]{0,1}[0-9]+) ->',x)
        if edge_f is not None:
            edge_f = edge_f.groups()[0]

        edge_t = re.search('-> ([-]{0,1}[0-9]+) \[label',x)
        if edge_t is not None:
            edge_t =edge_t.groups()[0]

        edge_l = re.search('-> [-]{0,1}[0-9]+ \[label=([0-9]+)',x)
        if edge_l is not None:
            edge_l = edge_l.groups()[0]

        # Append the edge or node to the appropriate lists
        if (node is not None) and (node_label is not None) and (node_occurrence is not None):
            node_name.append(node)
            node_l.append(node_label)
            node_occ.append(node_occurrence)

        if (edge_f is not None) and (edge_t is not None) and (edge_l is not None):
            edge_from.append(edge_f)
            edge_to.append(edge_t)
            edge_label.append(edge_l)

    # Create the dataframe from the lists

    d = {"node": node_name, "node_label": node_l, "node_occurrence": node_occ}
    nodes = pd.DataFrame(data=d)

    d = {"edge_from": edge_from, "edge_to": edge_to, "edge_label": edge_label}
    edges = pd.DataFrame(data=d)
    
    return nodes, edges


def import_log_file(file_name):
    log = xes_importer.apply('italian_help_desk.xes')
    attributes_list = pm4py.get_attributes(log)
    print(attributes_list)
    variants = pm4py.get_variants(log)
    return log


def get_change_points(log):
    attr_datetime = pm4py.get_attribute_values(log, 'time:timestamp')
    start_date = min(attr_datetime).date()
    end_date = max(attr_datetime).date()
    delta = datetime.timedelta(days=1)
    print("Start date: ", start_date, "\nEnd date: ", end_date)

    event_counts = {}
    i = start_date
    while i <= end_date:
        event_counts[i.strftime('%Y-%m-%d')] = 0
        #print(i)
        i += delta

    #print(event_counts)

    for t in attr_datetime:
        event_counts[t.date().strftime('%Y-%m-%d')] += 1

    dates = np.array(list(event_counts.values()))

    # detection
    algo = rpt.Pelt(model=MODEL).fit(dates)
    detect_result = algo.predict(pen=PENALTY)

    # display
    rpt.display(dates, detect_result, detect_result)
    plt.savefig('change_points.png')
    plt.show()
    print('Change point plot is saved as "change_points.png"')

    return event_counts, detect_result


def save_full_dfg(log):
    dfg = dfg_discovery.apply(log)

    gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
    dfg_visualization.view(gviz)
    parameters = {dfg_visualization.Variants.PERFORMANCE.value.Parameters.FORMAT: "svg"}
    gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY, parameters=parameters)
    dfg_visualization.save(gviz, "dfg_full.svg")
    print('Full DFG saves as "dfg_full.svg"')
    return gviz


def filter_for_periods(detect_result, event_counts):
    start_element1 = 0 if CHOSEN_PERIOD1 == 1 else detect_result[CHOSEN_PERIOD1-2]
    end_element1 = detect_result[CHOSEN_PERIOD1-1]

    start_element2 = 0 if CHOSEN_PERIOD2 == 1 else detect_result[CHOSEN_PERIOD2-2]
    end_element2 = detect_result[CHOSEN_PERIOD2-1]

    days = list(event_counts.keys())
    #print(days[start_element1])
    start_day1 = days[start_element1]
    end_day1 = days[end_element1-1]
    days_count1 = end_element1 - start_element1

    start_day2 = days[start_element2]
    end_day2 = days[end_element2-1]
    days_count2 = end_element2 - start_element2

    # Traces that are FULLY CONTAINED in the given timeframe
    period_1_log = timestamp_filter.filter_traces_contained(log, start_day1+" 00:00:00", end_day1+" 23:59:59")
    period_2_log = timestamp_filter.filter_traces_contained(log, start_day2+" 00:00:00", end_day2+" 23:59:59")

    # Traces that INTERSECT with the given timeframe
    # period_1_log = timestamp_filter.filter_traces_intersecting(log, start_day+" 00:00:00", end_day+" 23:59:59")

    dfg1 = dfg_discovery.apply(period_1_log)
    dfg2 = dfg_discovery.apply(period_2_log)

    gviz1 = dfg_visualization.apply(dfg1, log=period_1_log, variant=dfg_visualization.Variants.FREQUENCY)
    dfg_visualization.view(gviz1)

    # Saving the DFG
    parameters = {dfg_visualization.Variants.PERFORMANCE.value.Parameters.FORMAT: "svg"}
    gviz1 = dfg_visualization.apply(dfg1, log=period_1_log, variant=dfg_visualization.Variants.FREQUENCY, parameters=parameters)
    dfg_visualization.save(gviz1, "dfg1.svg")

    nodes_period1, edges_period1 = dot_to_df(gviz1)

    gviz2 = dfg_visualization.apply(dfg2, log=period_2_log, variant=dfg_visualization.Variants.FREQUENCY)
    dfg_visualization.view(gviz2)

    # Saving the DFG
    parameters = {dfg_visualization.Variants.PERFORMANCE.value.Parameters.FORMAT: "svg"}
    gviz2 = dfg_visualization.apply(dfg2, log=period_2_log, variant=dfg_visualization.Variants.FREQUENCY, parameters=parameters)
    dfg_visualization.save(gviz2, "dfg2.svg")


    return days_count1, days_count2, period_1_log, period_2_log, gviz1, gviz2


def get_statistics(period_1_log, period_2_log):
    variants_count1 = case_statistics.get_variant_statistics(period_1_log)
    variants_count1 = sorted(variants_count1, key=lambda x: x['count'], reverse=True)

    variants_count2 = case_statistics.get_variant_statistics(period_2_log)
    variants_count2 = sorted(variants_count2, key=lambda x: x['count'], reverse=True)

    trace_count1 = 0
    trace_count2 = 0

    for i in variants_count1:
        trace_count1 += i["count"]

    for i in variants_count2:
        trace_count2 += i["count"]


def slope_from_dateseries(time_series):
    # Calculating the trendline for the given time series
    # with linear regression
    day_order = []
    for i in range(0, len(time_series.index)):
        day_order.append(i)
    x = day_order
    y = time_series['count'].tolist() 
    #plt.scatter(x, y) 

    # adding the constant term 
    x = sm.add_constant(x) 

    result = sm.OLS(y, x).fit() 
    return result.params[1]

def get_slope(start_node, end_node, log):
    # Here we filter for a transition withing a given (filtered) log file
    # and we call the slope_from_dateseries on it in order to get the slope
    dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    unique_cases = dataframe['case:concept:name'].unique()
    nunique_cases = dataframe['case:concept:name'].nunique()
    time_series = []
    
    for n in range(nunique_cases):
        case_df = dataframe[dataframe['case:concept:name'] == unique_cases[n]]
        case_df.sort_values(by=['time:timestamp'])
        for index, row in case_df.iterrows():
            #print(row)
            if min(case_df.index) + len(case_df.index) - 1 == index:
                break
            elif row[ACTIVITY_NAMES] == start_node and case_df.loc[index+1, ACTIVITY_NAMES] == end_node:
                time_series.append(case_df.loc[index+1, 'time:timestamp'])
        del case_df
    
    day_min = min(time_series).date()
    day_max = max(time_series).date()
    
    df = pd.DataFrame()
    
    date_series = []
    for x in time_series:
        date_series.append(x.date())
    
    daterange = pd.date_range(start=day_min, end=day_max)
    
    for single_date in daterange:
        #print(single_date)
        data = [{'day': single_date.strftime("%Y-%m-%d"), 'count': date_series.count(single_date.date())}]
        df = df.append(data, ignore_index=True)
    
    slope = slope_from_dateseries(df)
    
    return slope

def add_slope_to_period_df(edges, log):
    # We loop through each of the edges in our dataframe,
    # and call the get_slope function 
    edges["slope"] = 0
    print("Calculating slope for each transition....")
    for index, row in edges.iterrows():
        try:
            from_node = nodes_full.loc[nodes_full["node"] == row["edge_from"],"node_label"].iloc[0].strip()
            to_node = nodes_full.loc[nodes_full["node"] == row["edge_to"],"node_label"].iloc[0].strip()
            print("Transition: ", from_node," -> ", to_node)
            edges.loc[index, "slope"] = get_slope(from_node, to_node, log)
        except:
            pass
    print("Done.")
    return edges


def merge_graphs(edges_period1, edges_period2, days_count1, days_count2, edges_full):
    edges_period1["daily_avg"] = edges_period1["edge_label"].astype(int) / days_count1
    edges_period2["daily_avg"] = edges_period2["edge_label"].astype(int) / days_count2
    edges_merged = edges_full.copy()
    edges_merged["edge_label"] = 0
    edges_merged["daily_avg"] = 0
    edges_merged["daily_diff"] = 0
    edges_merged["slope"] = 0
    edges_merged["slope_diff"] = 0

    for index, row in edges_merged.iterrows():
        for i, r in edges_period2.iterrows():
            if r["edge_from"] == row["edge_from"] and r["edge_to"] == row["edge_to"]:
                edges_merged.loc[index, "edge_label"] = r["edge_label"]
                edges_merged.loc[index, "daily_avg"] = r["daily_avg"]
                edges_merged.loc[index, "slope"] = r["slope"]
                
        for i, r in edges_period1.iterrows():
            if (r["edge_from"] == row["edge_from"] and r["edge_to"] == row["edge_to"]) and r["daily_avg"] != 0:
                edges_merged.loc[index, "daily_diff"] = edges_merged.loc[index, "daily_avg"] / r["daily_avg"] - 1
            
            if (r["edge_from"] == row["edge_from"] and r["edge_to"] == row["edge_to"]) and r["slope"] != 0:
                edges_merged.loc[index, "slope_diff"] = edges_merged.loc[index, "slope"] - r["slope"]

    # Removing edges which do not exist in the first period and in the second either
    edges_merged = edges_merged[(edges_merged[['daily_avg','daily_diff', 'slope', 'slope_diff']] != 0).any(axis=1)]

    return edges_merged


def df_to_dot(nodes, edges, filename):
    s = """
digraph {
	graph [bgcolor=transparent]
	node [shape=box]
"""
    for index, row in nodes.iterrows():
        s += '	' + row["node"] +' [label="' + row["node_label"] + '" fillcolor="#FFFFFF" style=filled]\n'
    
    for index, row in edges.iterrows():
        s += '	' + row["edge_from"] +' -> ' + row["edge_to"] + ' [label="' \
                + str("{:.3f}".format(row["daily_avg"])) + '/day\\n(' + str("{:.2f}".format(row["daily_diff"]*100)) \
                + '%)\\nSC: ' + str("{:.4f}".format(row["slope_diff"])) + '" penwidth=1.0 color="'
        if row["daily_diff"] < 0:
            s+= 'red'
        elif row["daily_diff"] > 0:
            s+= 'green'
        else:
            s+= 'black'
        s += '"]\n'
    s += """
	overlap=false
	fontsize=10
}
"""
    gviz = Source(s, filename=filename, format="png")
    return gviz


#---------------------------------------------------------------------------------
# print(os.getcwd())
log = import_log_file(INPUT_XES_FILE)
event_counts, detect_result = get_change_points(log)

gviz = save_full_dfg(log)

nodes_full, edges_full = dot_to_df(gviz)

days_count1, days_count2, period_1_log, period_2_log, gviz1, gviz2 = filter_for_periods(detect_result, event_counts)

nodes_period1, edges_period1 = dot_to_df(gviz1)
nodes_period2, edges_period2 = dot_to_df(gviz2)

get_statistics(period_1_log, period_2_log)

add_slope_to_period_df(edges_period1, period_1_log)
add_slope_to_period_df(edges_period2, period_2_log)
print('Slopes are added to the period dataframes')

edges_merged = merge_graphs(edges_period1, edges_period2, days_count1, days_count2, edges_full)

gviz_merge = df_to_dot(nodes_full, edges_merged, OUTPUT_GRAPH_FILE)

print(gviz_merge)
dfg_visualization.view(gviz_merge)
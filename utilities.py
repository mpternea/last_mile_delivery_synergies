# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:04:39 2017

@author: to_va
Contains general functions  that have not been specified as methods for any class.

"""

import numpy as np
import random
import gurobipy as grb
from itertools import groupby
import openpyxl

def getGrid (Lx, Ly, unit_distance,dstops, n, p_res, rhor_y, rvert_x, r3x, r3y):
    # 1) Generate a grid Lx * Ly (Number of squares, not nodes)
    
    x_range = [unit_distance*i for i in range (0, Lx+1)]
    y_range = [unit_distance*i for i in range (0, Ly+1)]
    max_len_x = max(x_range)
    max_len_y = max(y_range)
    num_nodes_x = int(Lx+1)
    NumNodesY = int(Ly+1)
    num_nodes = num_nodes_x * NumNodesY
    nodes = [i for i in range (1, num_nodes+1)]
             
    coords = []
    for y in y_range:
        for x in x_range:
            coords.append((x,y)) 
    coords = dict(zip(nodes, coords))                
    X = {node: coords[node][0] for node in nodes}
    Y = {node: coords[node][1] for node in nodes}
    ### Depot
    Xdepot = x_range[int(len(x_range)/2)]
    Ydepot = y_range[int(len(y_range)/2)]     
    for i in nodes:
        if X[i] == Xdepot and Y[i] == Ydepot:
            depot = i
    
    # 2) Divide into equal regions (e.g. 3*3, 4*4, similarly to the paper)
    # Find the coordinates of the nodes in the borders, in each dimension.
    border_x = list(np.linspace(0, max_len_x, num = n+1))
    BorderY = list(np.linspace(0, max_len_y, num = n+1))
    # Now define areas by (minx,maxx, miny, maxy)
    numAreas = n*n
    Areas = ['A' + str(i) for i in range (1, numAreas +1)]
    x_bounds_all = []
    y_bounds_all = []
    for (iy,y) in enumerate(BorderY):
        if iy == len(BorderY)-1:
            continue
        y_bounds_all.append((y, BorderY[iy+1]))
    for (ix,x) in enumerate(border_x):
        if ix == len (border_x)-1:
            continue
        x_bounds_all.append((x, border_x[ix+1]))
    
    indarea = 0
    x_bounds_area = {}
    y_bounds_area = {}
    for y_bounds in y_bounds_all:
        for x_bounds in x_bounds_all:
            area = Areas[indarea]
            x_bounds_area[area] = x_bounds
            y_bounds_area[area] = y_bounds
            indarea = indarea + 1
            
    # 3) Allocate nodes to areas
    area_node = {}
    areas_node = {node: [] for node in nodes}
    nodes_area = {area: [] for area in Areas}
    for node in nodes:
        XNode = coords[node][0]
        YNode = coords[node][1]
        for area in Areas:
            x1_low = x_bounds_area[area][0]
            x1_up = x_bounds_area[area][1]
            if XNode >= x1_low and XNode<=x1_up:
                y1_low = y_bounds_area[area][0]
                y1_up = y_bounds_area[area][1]
                if YNode >=y1_low and YNode <= y1_up:
                    areas_node[node].append(area)
    for node in areas_node:
        if len (areas_node[node]) ==1:
            area = areas_node[node][0]
            area_node[node] = area
            nodes_area[area].append(node)
        else:
            area = random.choice ( areas_node[node])
            area_node[node] = area
            nodes_area[area].append(node)
            
    # 4) Define Areas as business ("B") or residential ("R").
    AreasRes = random.sample (Areas, int(p_res*numAreas))
    AreasCom = [a for a in Areas if a not in AreasRes]
    
    # 5) Define nodes as business or residential
    nodes_res = []
    nodes_bus = []
    for node in nodes:
        area = area_node[node]
        if area in AreasRes:
            nodes_res.append(node)
        else:
            nodes_bus.append(node)
            
    # 6) Define rectangular bus routes
    nodes_route= {}
    ind = 1
    # Horizontal route
    for y in rhor_y:
        y1_init = int(y * max_len_y) # Might not correspond exaqctly to node
        # Find nearest coordinate
        y_diff  = {y: abs(y-y1_init) for y in y_range}
        y1 = min (y_diff, key = y_diff.get) # the y that is closest
        route_name = 'R'+str(ind)
        nodes_route[route_name] = [node for node in nodes if coords[node][1] == y1]
        ind = ind + 1
    # Vertical Route
    for x in rvert_x:  
        x2_init = int(x* max_len_x)
        x_diff  = {x: abs(x-x2_init) for x in x_range}
        x2 = min (x_diff, key = x_diff.get) # the y that is closest
        route_name = 'R'+str(ind)
        nodes_route[route_name] = [node for node in nodes if coords[node][0] == x2]
        ind = ind + 1
    # Third route
    # Find break points
    
    x3_init = int(r3x*max_len_y)
    x3_dif  = {x: abs(x-x3_init) for x in x_range}
    x3 = min (x3_dif, key = x3_dif.get) # the y that is closest
    
    y31_init = int(r3y[0]*max_len_y)
    DictY31Diff  = {y: abs(y-y31_init) for y in y_range}
    y31 = min (DictY31Diff, key = DictY31Diff.get) # the y that is closest
    
    y32_init = int(r3y[1]*max_len_y)
    DictY32Diff  = {y: abs(y-y32_init) for y in y_range}
    y32 = min (DictY32Diff, key = DictY32Diff.get) # the y that is closest
    
    Nodes1 = [node for node in nodes if Y[node] == y31 and X[node] <= x3]
    Nodes2 = [node for node in nodes if X[node] == x3 and Y[node] >= y31 and Y[node] <= y32]
    Nodes3 = [node for node in nodes if Y[node] == y32 and X[node]>= x3]
    
    route_name = 'R'+str(ind)
    nodes_route[route_name] = Nodes1[0:len(Nodes1)-2] + Nodes2[0:len(Nodes2)-2] + Nodes3 # Includes all internal nodes, not only stops
    routes = list(nodes_route.keys())
    stops_route = {route: getStopsFromNode (nodes_route[route],unit_distance, dstops) for route in routes}
    stops = set()
    for route in routes:
        for stop in stops_route[route]:
            stops.add(stop)
    return (nodes,nodes_res, nodes_bus, routes, stops, X, Y, stops_route,
            AreasRes, AreasCom, Areas, area_node,nodes_area, depot)
    
    
def getWindows (start, end, twlen, bus_start, bus_end, peak_periods):
    # Calculate stime windows, business, peak, according to data
    if (end-start)/twlen !=0:
        numtw = int((end-start)/twlen) +1 # The last time window will be smaller
    else:
        numtw = int((end-start)/twlen)
    
    
    twindows = [i for i in range (1, int(numtw+1))]
    tw_start = {}
    tw_end=  {}
    for (ind,t) in enumerate(twindows):
        tw_start[t] = start + ind*twlen
        tw_end[t] = tw_start[t] + twlen
    if tw_end[twindows[-1]] > end: # We have reached the final time windows
            tw_end[twindows[-1]] = end
    
    tw_start_end = {tw: (tw_start[tw], tw_end[tw]) for tw in twindows}
    
       
    twindows_bus = [i for i in twindows if tw_start[i] >=bus_start and  tw_end[i] <= bus_end]
    twindows_peak = []
    for period in peak_periods:
        start = period[0]
        end = period[1]
        twindows_peak.extend ([tw for tw in twindows if tw_start[tw]>= start and tw_end[tw]<= end])
    return (twindows, twindows_bus, twindows_peak, tw_start_end)

    
    
def getStopsFromNode (nodes_sequence, dnodes, dstops):
    # Creates a sequence of actual stops from a sequence of nodes.
    ratio = dstops / dnodes
    numgroups = int(len(nodes_sequence) / ratio)
    if len(nodes_sequence) % ratio == 0:
        lastnode_ind_coeff = numgroups
    else:
        lastnode_ind_coeff = numgroups + 1
    stop_inds = [int(i*ratio) for i in range (0,lastnode_ind_coeff)]
    stop_seq = [nodes_sequence[i] for i in stop_inds]
    return (stop_seq)
    
    
#def getSum (N,SUM):
#    List01 = [random.uniform(0,1) for i in range (0, N)]
#    sumlist01 = sum(List01)
#    ListN = [round((SUM*i)/sumlist01,0) for i in List01]
#    print ('riri', sum(ListN))
#    return (ListN)

def getSum (n_keys,sum_keys):
    ssum = 0
    sum_so_far = {key: 0 for key in n_keys}
    while ssum < sum_keys:
        key_to_increase = random.choice (n_keys)
        sum_so_far[key_to_increase] =  sum_so_far[key_to_increase] + 1
        ssum = ssum + 1

    return (sum_so_far)


    
def getResults (X, df_peak_tw_m, df_off_tw_m, routes, twindows, dem_nodes):
    # _m: Produced from the model
    routes_used = set()
    stops_used = set()
    stop_node_tw_route = {node:{tw:[] for tw in twindows} for node in dem_nodes} # {NODE:TW:(route,stop)}
    stops_node = {node:set() for node in dem_nodes}
    routes_node  = {node:set() for node in dem_nodes}  
    df_peak_tw = {}
    df_off_tw = {}

#    DictFreqPerAnyPeakTW = {}
#    DictFreqPerAnyOffPeakTW = {}
    # Build a function that interprets the solution
    
    for (i,j,k,t) in X:
        if X[i,j,k,t].x >0:
            stops_used.add(j)
            routes_used.add(i)
            stop_node_tw_route[k][t].append((i,j))
            stops_node[k].add(j)
            routes_node[k].add(i)
    for r in routes:
        df_peak_tw[r] = df_peak_tw_m[r].x
        df_off_tw[r] = df_off_tw_m[r].x
#        DictFreqPerAnyPeakTW[r] = b_freq_any_peak_tw[r] + df_peak_tw[r]
#        DictFreqPerAnyOffPeakTW[r] = b_freq_any_off_tw[r] + df_off_tw[r]
    return (stops_used, routes_used, stop_node_tw_route, stops_node, routes_node,
            df_peak_tw, df_off_tw)
    
def getModelBus (routes, dem_nodes, nodes_range, nodes_bus_range, stops_route, pot_nodes, 
                 pot_stops, dem_node, cap_basic, twindows, twindows_bus, 
                 twindows_peak, twindows_off, vehicle_per_route, cap_bus, fleet_bus,
                 routes_type, links_route, prec_stops_route_link, len_route_link,
                 b_freq_any_peak_tw, b_freq_any_off_tw,time_per_route, twlen, bus_types,
                 demand, freq_tw, dist, c_route, c_stop, c_tm, c_dwell, c_local, c_late, c_freq,
                 fmax, cost_route_round, b_freq_tw, cost_fixed_bus):
    m = grb.Model()
    X= {}
    print ('Now adding X')
    for i in routes:
        for j in stops_route[i]:
            for k in pot_nodes[j]:
                for t in twindows:
                    X[i,j,k,t] = m.addVar (vtype = grb.GRB.INTEGER, name = 'X_%s_%s_%s_%s' %(i,j,k,t))
                    
    B = {}
    print ('Now adding B')
    for i in routes:
        for j in stops_route[i]:
            B[i,j] = m.addVar (vtype = grb.GRB.BINARY, name = 'X_%s_%s' %(i,j))
    
    R = {}
    print ('Now adding R')
    for i in routes:
        R[i] = m.addVar (vtype = grb.GRB.BINARY, name = 'R_%s' %i)
    
    Y = {}
    print ('Now adding Y')
    for i in routes:
        for j in stops_route[i]:
            for k in pot_nodes[j]:
                for t in twindows:
                    Y[i,j,k,t] = m.addVar (vtype = grb.GRB.BINARY, name = 'Y_%s_%s_%s_%s' %(i,j,k,t))
                    
    print ('Now adding late demand')               
    DEM_LATE = {}
    for k in dem_nodes:
        for t in twindows:
            DEM_LATE[k,t] = m.addVar (vtype = grb.GRB.INTEGER, lb = 0, name = 'DEM_LATE_%s_%s' %(k,t))
    
    df_peak_tw = {} # Increase in frequency for peak hours
    df_off_tw = {} # Increase in frequency for off-peak hours
    for i in routes:
        df_peak_tw[i] = m.addVar (vtype = grb.GRB.CONTINUOUS, lb = 0.0, name = 'DF_PEAK_%s' %i)
        df_off_tw[i] = m.addVar (vtype = grb.GRB.CONTINUOUS, lb = 0.0, name = 'DF_OFFPEAK_%s' % i)
    m.update
    
    print ('Now making linear expressions')
    num_veh_route = {}
    for route in routes:
        num_veh_route[route] = 2 * time_per_route[route] * (b_freq_any_peak_tw[route] + df_peak_tw[route])/ twlen
    print ('Now making objectives')
    ## Objective function components
    # 1) Using a route
    SUM1 = grb.quicksum(c_route[i]*R[i] for i in routes)
    # 2) Using a stop 
    SUM2 = grb.quicksum(c_stop[i,j]*B[i,j] for i in routes for j in stops_route[i])
    # 3) Link travel
    tot_load_per_route_time = {}
    for i in routes:
        for t in twindows:
            summ = 0
            for j in stops_route[i]:
                for k in pot_nodes[j]:
                    summ = summ + X[i,j,k,t]
                    tot_load_per_route_time[i,t] = summ
    SUM3 = 0
    tot_load_link_time = {}
    for i in routes:
        for t in twindows:
            for link in links_route[i]:
                sum2 = grb.quicksum(X[i,j,k,t] for j in prec_stops_route_link[i][link] for k in pot_nodes[j])
                tot_load_link_time[i,link,t] = tot_load_per_route_time[i,t] - sum2
                SUM3 = SUM3 + tot_load_link_time[i,link,t] * len_route_link[i,link] * c_tm
                    
    # 4) Bus dwelling stop
    SUM4 = grb.quicksum(X[i,j,k,t] for i in routes for j in stops_route[i] for k in pot_nodes[j] for t in twindows) * c_dwell
    
    # 5) Local Delivery Cost
    SUM5 = c_local * grb.quicksum(Y[i,j,k,t]*dist[j,k] for i in routes for j in stops_route[i] for k in pot_nodes[j] for t in twindows)
    
    # 6) Penalty for unserved demand (for residential areas all time and for commercial areas outside of business hours
    SUM6 = grb.quicksum (DEM_LATE[kt] for kt in DEM_LATE) * c_late
    
    # 7) Frequency Cost (include it here or constrain it )
#    SUM7 = c_freq*(grb.quicksum (freq_tw[i,tw] + df_peak_tw[i] for i in routes for tw in twindows_peak) \
#        + grb.quicksum (freq_tw[i,tw] + df_off_tw[i] for i in routes for tw in twindows_off))
    # 8) Bus operating cost
    SUM8 = grb.LinExpr()
    for i in routes:
        for tw in twindows_peak:
            SUM8 = SUM8 + cost_route_round[i] * (b_freq_tw[i,tw] + df_peak_tw[i])
        for tw in twindows_off:
            SUM8 = SUM8 + cost_route_round[i] * (b_freq_tw[i,tw] + df_off_tw[i])
            
    # 9) Bus fixed cost
    SUM9 = grb.quicksum(num_veh_route[i] * cost_fixed_bus[vehicle_per_route[i]] for i in routes)
    obj = SUM1 + SUM2 + SUM3 + SUM4 + SUM5 + SUM6 + SUM8 + SUM9
    m.setObjective(obj)
    ############### CONSTRAINTS ##################
    #1 All demands must be served
    # 1a: By the end of the day
    print ('Now adding C_DEM_DAY')
    C_DEM_DAY = {}
    for k in nodes_range:
        C_DEM_DAY[k] = m.addConstr (grb.quicksum(X[i,j,k,t] for i in routes \
                    for j in set(stops_route[i]) & set(pot_stops[k])\
                                      for t in twindows) == dem_node[k])
    
    # 1b: Within their time window
    print ('Now adding C_DEM_TW')
    C_DEM_TW = {}
    for k in nodes_bus_range : # Not all nodes, only these that can be served by bus
        for t in twindows_bus:
            C_DEM_TW[k,t] = m.addConstr (grb.quicksum(X[i,j,k,t] for i in routes \
            for j in set(stops_route[i]) & set(pot_stops[k])) >= demand[k,t])
    
    # Calculate unserved demand
    print ('Now adding C_DEM_LATE')
    DemDiff = {} # Initialize the number of demand units that are unserved 
    # at every time unit. Demdiff[k,t] can be either positive or negative.
    for k in nodes_range:
        for t in twindows:
            DemDiff[k,t] = sum(demand[k,ti] for ti in range (1,t+1)) \
                                - grb.quicksum(X[i,j,k,ti] for ti in range (1,t+1) \
                               for i in routes for j in set(stops_route[i]) & set(pot_stops[k]))
    C_DEM_LATE= {}
    for k in nodes_range:
        for t in twindows:
            if k in nodes_range and t in twindows_bus:
                continue
            C_DEM_LATE[k,t] = m.addConstr (DEM_LATE[k,t] >= DemDiff[k,t]) # Define lower bound = 0 in the beginning.
            
    ## 2 Route Capacity
    print ('Now adding C_CAP')
    C_CAP = {}
    for i in routes:
        for t in twindows:
            if t in twindows_peak:
                DCAP = df_peak_tw[i] * cap_bus[vehicle_per_route[i]]
            else:
                DCAP = df_off_tw[i] * cap_bus[vehicle_per_route[i]]
            C_CAP[i,t] = m.addConstr (grb.quicksum(X[i,j,k,t]  for j in stops_route[i] for k in pot_nodes[j])\
                                        <= cap_basic[i,t] + DCAP)
    
    ## 3 Stop Choice
    print ('Now adding C_STOP')
    C_STOP = {}
    for i in routes:
        for j in stops_route[i]:
            for k in pot_nodes[j]:
                for t in twindows:
                   C_STOP[i,j,k,t] = m.addConstr(X[i,j,k,t] <= demand[k,t] * B[i,j])
    
    # 4 Stop-->Route Choice
    print ('Now adding C_ROUTE')
    C_ROUTE = {}
    for i in routes:
        for j in stops_route[i]:
            C_ROUTE[i,j] = m.addConstr(B[i,j] <= R[i])
    
    #
    # 5 Local Delivery
    print ('Now adding C_LOCAL')
    C_LOCAL= {}
    for i in routes:
        for j in stops_route[i]:
            for k in pot_nodes[j]:
                for t in twindows:
                    C_LOCAL[i,j,k,t] = m.addConstr(X[i,j,k,t] <= demand[k,t]*Y[i,j,k,t])
    
    # 6 Fleet size
    print ('Now adding C_FLEETSIZE')
    C_FLEET = {}
    for bt in bus_types:
        C_FLEET[bt] = m.addConstr (grb.quicksum(num_veh_route[route] for route in routes_type[bt]) <= fleet_bus[bt])
    
    # 7 Maximum frequency
    print ('Now adding C_MAX_FREQ')
    C_MAX_FREQ_PEAK = {}
    C_MAX_FREQ_OFFPEAK = {}
    for i in routes:
        C_MAX_FREQ_PEAK[i] = m.addConstr ((b_freq_any_peak_tw[i] + df_peak_tw[i]) / twlen <= fmax)
        C_MAX_FREQ_OFFPEAK[i] = m.addConstr ((b_freq_any_off_tw[i] + df_off_tw[i]) / twlen <= fmax)

    m.update()
    m.params.TimeLimit = 60
    m.params.MIPGap = 0.01
    #m.params.presolve = 0
    m.optimize()
    print ('Running time:', m.runtime)
    rvector = (m, X, df_peak_tw, df_off_tw, SUM1, SUM2, SUM3, SUM4, SUM5, SUM6, SUM8, SUM9)
    if m.status == 2:
        return (rvector)
    else:
        rvector = None
    return (rvector)

    
# From data to VRPTW
def transformInput (nodes, DX, DY, twindows,demand,twlen,tw_start_end,
                       depot, nodes_bus, twindows_bus, start,numvehicles, num_reloads_per_vehicle,
                       horizon):
    offset = start
    TupList = [] # {(X, Y, TW, DEMAND)}
    truck_nodes = []
    #DictTupsPerNode = {node: [] for node in nodes}
    x_tr_node = {}
    y_tr_node = {}
    node_tw_tr_node = {}
    dem_tr_node = {}
    tw_tr_node = {}
    tw_start_tr_node = {}
    tw_end_tr_node = {}
    truck_nodes_or_node = {node : [] for node in nodes}
    for node in nodes:
        X = DX[node]
        Y = DY[node]
        for tw in twindows:
            dem = demand[node,tw]
            tup  = (X, Y, tw, dem)
            if dem ==0:
                continue
            TupList.append(tup)
            trnode = len(TupList)
            x_tr_node[trnode] = X
            y_tr_node[trnode] = Y
            dem_tr_node[trnode] = dem
            tw_tr_node[trnode] = tw
            tw_start_tr_node[trnode] = tw_start_end[tw][0] - offset # At least temporarily
            tw_end_tr_node[trnode] = tw_start_end[tw][1] - offset
            node_tw_tr_node[trnode] = (node,tw)
            truck_nodes.append(trnode)
            truck_nodes_or_node[node].append(trnode)
    for tw in twindows:
        node_tw_tr_node[0] = (depot,tw) # Depot It contains the last tw 
    #        DictTupsPerNode[node].append(tup) 
    # Extend time windows for non-penalized early delivery from 0 to time
    for node in nodes:
        tnodes = truck_nodes_or_node[node] 
        for tnode in tnodes:
            tw = tw_tr_node[tnode]
            if tw in twindows_bus and node in nodes_bus:
                continue
            tw_start_tr_node[tnode] = 0

    # Create reload nodes
    first_reload_id = len([depot] + truck_nodes)
    num_reload_nodes  = numvehicles * num_reloads_per_vehicle
    nodes_reload = [i for i in range (first_reload_id, first_reload_id + num_reload_nodes)]
    locations_reload = [(DX[depot], DY[depot]) for i in nodes_reload]
    nodes_all = [depot] + truck_nodes + nodes_reload
    
    
    locations =  [(DX[depot], DY[depot])] + [(x_tr_node[tnode],y_tr_node[tnode]) for tnode in truck_nodes] + locations_reload ### Add depot when needed
    start_times = [0]+ [tw_start_tr_node[tnode]*3600 for tnode in truck_nodes] + [0 for i in nodes_reload]
    end_times = [0+twlen*3600] + [tw_end_tr_node[tnode]*3600 for tnode in truck_nodes] + [horizon for i in nodes_reload]
    demands = [0] + [dem_tr_node[tnode] for tnode in truck_nodes] + [0 for i in nodes_reload]
    return (locations, start_times, end_times, demands, node_tw_tr_node, 
            nodes_all, nodes_reload, first_reload_id)
    

        
# From VRPTW to results
def getTruckSolution (truck_res, node_tw_tr_node, depot, only_one_tw_per_node, nodes_reload):
    ## Uses the solution of OR tools for truck routing and interprets the results
    truck_solution = []
    for route in truck_res:
        splitted = route.split('->')
        stop_seq = [depot] # Always starts at the depot
        for stopstr in splitted[1:]: # The first one is always the depot, we do not care
            split2 = stopstr.split('Load')
            nodetruckname = int(split2[0])
            if nodetruckname in nodes_reload: ### Added this condition on July 16
                 stop_seq.append(depot)
            else:
#                if only_one_tw_per_node == 0:
                stop_seq.append(node_tw_tr_node[nodetruckname][0])
#                else:
#                    stop_seq.append(nodetruckname)
        truck_solution.append(stop_seq)
    truck_routes = []
    for route in truck_solution:
        if route == [depot, depot] or route == [depot,0]:
            continue
        truck_routes.append(route)
    for route in truck_routes:
        for (index,i) in enumerate(route):
            if i == 0:
                route[index] = depot

    routenames = ['TRUCK_ROUTE_' + str(i) for i in range (1, len(truck_routes)+1)]
    truck_route_route_name = dict(zip(routenames, truck_routes))
    route_no_doubles_route_name = {}
    reload_route = {}
    for route_name in routenames:
        route = truck_route_route_name[route_name]
        route_no_cont_doubles = [x[0] for x in groupby(route)]
        route_no_doubles_route_name[route_name] = route_no_cont_doubles
        internal_route = route_no_cont_doubles[1:len(route_no_cont_doubles)-1]
        if depot in internal_route:
            reload = 1
        else:
            reload = 0
        reload_route[route_name] = reload
    return (truck_routes, route_no_doubles_route_name, reload_route)
    
def getVmtTruck (truck_routes,dist,factor_meters_to_miles):
    VMTTruckPerRoute = {}
    for (indr,route) in enumerate(truck_routes):
        length = 0
        for (i,node) in enumerate (route):
            if i == len(route) -1:
                break
            else:
                nextnode = route[i+1]
                length = length + dist[node, nextnode]
        VMTTruckPerRoute[indr] = length*factor_meters_to_miles
    VMTTruck = sum(VMTTruckPerRoute.values())
    return (VMTTruckPerRoute, VMTTruck)   
    
def printRes (experiments, wb_name, sheet_name):
    exceldoc = openpyxl.load_workbook(wb_name)
    ws = exceldoc.create_sheet(sheet_name)
    experiment = {exp: {} for exp in experiments} # Initialize a dictionary 
    for exp in experiments:
        experiment[exp]['Name'] = exp.Name
        experiment[exp]['Case Study'] = exp.CaseStudy.Name
        experiment[exp]['only_one_tw_per_node'] = exp.CaseStudy.only_one_tw_per_node
        experiment[exp]['Average Demand'] = exp.CaseStudy.demav
        experiment[exp]['Bus Capacity'] = str(tuple(exp.CaseStudy.cap_bus.values()))
        experiment[exp]['Operating scheme'] = exp.Scheme
        experiment[exp]['VMT_Bus'] = exp.VMT_Bus
        experiment[exp]['VMT_tr'] = exp.VMT_tr
        experiment[exp]['VMT_Total'] = exp.VMT_Total
        experiment[exp]['Truck_Fuel_Gallons_Used'] = exp.truck_fuel
        experiment[exp]['Bus_Fuel_Gallons_Used'] = exp.bus_fuel
        experiment[exp]['Total_Fuel_Gallons_Used'] = exp.total_fuel
        experiment[exp]['Truck_CO2_Grams_Emitted'] = exp.truck_co2
        experiment[exp]['Bus_CO2_Grams_Emitted'] = exp.bus_co2
        experiment[exp]['Total_CO2_Grams_Emitted'] = exp.total_co2
        experiment[exp]['Time'] = exp.Time
    row_lists = []
    title_list = ['Name', 'Case Study','only_one_tw_per_node', 'Average Demand', 'Bus Capacity',
                 'Operating scheme', 'VMT_Bus', 'VMT_tr', 'VMT_Total',
                 'Truck_Fuel_Gallons_Used','Bus_Fuel_Gallons_Used','Total_Fuel_Gallons_Used',
                 'Truck_CO2_Grams_Emitted','Bus_CO2_Grams_Emitted','Total_CO2_Grams_Emitted',
                 'Time']
                 
    for exp in experiments:
        row = [experiment[exp][i] for i in title_list]
        row_lists.append(row)
    
    ws.append(title_list)
    for row in row_lists:
        ws.append(row)
    exceldoc.save(wb_name)
    return ()
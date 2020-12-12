# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:55:30 2017

Problem: A feasibility study for last-mile synergies between
passenger and freight transport for an urban area.

We test two operating schemes:
    1) Truck only ("SchemeTruck"), where a truck delivers packages (as is usually done)
    2) Truck for nodes that are "far" from bus routes, and buses for nearby nodes ("SchemeBus")

In the integrated scheme (2), nodes that are within some range of a bus route
are served by the the bus. The parcels are loaded at one of the two terminal stops,
and unloaded in certain bus stops, where thet are picked up by a delivery person 
and distributed using a light vehicle or on foot.

A) HOW TO REPLICATE THE PROPOSED SCHEME:
To replicate the scheme of integrating city buses in the last mile delivery problem,
we use a MILP model to determine the optimal assignment of parcels to bus stops, considering
DIRECT and EXTERNAL costs (associated with Level of Service). 

B) HOW TO DETERMINE IF BUS FREQUENCY IS SUFFICIENT:
Treat frequency as a decision variable and adjust it, if needed.

C) HOW TO DETERMINE IF THE SCHEME IS BENEFICIAL:
For different cases, we calculate the following results:
    a) Number of vehicle-miles traveled
    b) Amount of carbon dioxide emitted.


"""

import synergies
import utilities
import vehicle_routing
import time



                      
                      
def getmain(args):
    
    options = {"truck_fuel_gallons_per_100_miles" : 6,
               "grams_co2_per_fuel_gallon" : 8.887,
               "bus_miles_per_fuel_gallon" : 7.2,
               "f_meters_to_miles" : 0.000621371,
               "peak_to_off" : 1.5
               }

    options.update(args)     
    p_res = args ["p_res"] # Percentage of residential areas.
    Lx = args["Lx"] # Grid length
    Ly = args["Ly"] # Grid height
    unit_distance = args["unit_distance"] # Unit of distance in which Lx, Ly are measured
    dstops = args["dstops"] # Distance between consecutive stops.
    num_areas_per_dimension = args["num_areas_per_dimension"] # Number of large squares per dimension
    rhor_y = args["rhor_y"] # The horizontal route is located at half the height of the area
    rvert_x = args["rvert_x"] # The vertical route is located at half the width of the area.
    r3x = args["r3x"] # r3x and r3y indicate where the 3rd route is located.
    r3y = args["r3y"]

    dem_av = args ["dem_av"]
    peak_to_off = args["peak_to_off"] # High Vs Low average demand according to node type and time of day.
    c_tm = args["c_tm"] #3  Cost per tonne mile traveled for each bus.
    c_dwell = args["c_dwell"] # 4 Cost per package to unload at a stop.
    c_local = args["c_local"] # 5 Cost of assigning a demand node to a stop for local delivery.
    c_late = args["c_late"] # 6 Cost of satisfying demand after its time window.
    c_freq = args["c_freq"] # 7 Cost of frequency (now I use # 9 instead, as an approximation for operational cost)
    bus_types = args["bus_types"] # Available bus types

    cap_bus = args ["cap_bus"] # Capacity per bus type
    fleet_bus = args["fleet_bus"] # Available fleet for each bus type
    cost_fixed_bus = args["cost_fixed_bus"] # 8, Fixed cost for each bus (dollars per vehicle)
    cost_km_bus = args["cost_km_bus"] # 9 Operating cost for each bus (dollars per km)

    c_route = args["c_route"]  #1 Cost of using a route.
    c_stop = args["c_stop"] #2 Cost of using a stop
    vehicle_route = args["vehicle_route"] # Vehicle type serving each route
    b_freq_peak_route = args["b_freq_peak_route"] # Existing ("basic") route frequency, peak
    b_freq_off_route = args["b_freq_off_route"] # Existing ("basic") route frequency, off-peak

    distlim = args["distlim"] # Maximum range within which a node can be served by a bus stop.
    start = args["start"]  # Beginning of first time window
    end = args["end"] # End of last time window.
    twlen = args["twlen"] # Length of time window (hours)

    bus_start = args ["bus_start"] # start of business period
    bus_end = args["bus_end"] # end of business period
    peak_periods = args["peak_periods"] # start and end of peak periods
    bus_speed = args["bus_speed"] # Average bus speed in km/h #
    one_tw_per_node = args["one_tw_per_node"] # The demand of a node can be accumulated in 1 time window (1) or in many (0).
    num_trucks_init = args["num_trucks_init"] # Number of available trucks
    time_per_dem_unit = args["time_per_dem_unit"] # Time ro serve per demand unit
    horizon = args["horizon"] # Maximum allowed trip duration (minutes)
    speed = args["speed"] # Vehicle speed in meters per second
    truck_capacity = args["truck_capacity"] # Capacity of each vehicle in packages
    single_route_truck = args["single_route_truck"] # 1 if truck cannot return to  depot for reloading, 0 if yes.
    case_name_ind = args["case_name_ind"]
    f_meters_to_miles = args["f_meters_to_miles"]
    fmax = args["fmax"] # Maximum allowed bus frequency (buses/hour)
    num_reloads_per_vehicle = args["num_reloads_per_vehicle"] # Maximum number of allowed reloads per truck.
    truck_fuel_gallons_per_100_miles = args["truck_fuel_gallons_per_100_miles"]
    grams_co2_per_fuel_gallon = args["grams_co2_per_fuel_gallon"] 
    bus_miles_per_fuel_gallon = args["bus_miles_per_fuel_gallon"] 
    f_meters_to_miles = args["f_meters_to_miles"] 
    peak_to_off = args["peak_to_off"] # Ratio of peak demand to off-peak demand
    
    # Create the network.
    (nodes,nodes_res, nodes_bus, routes, stops, dx, dy, stops_per_route,
         areas_res, areas_com, areas, area_node, nodes_area, depot) = utilities.getGrid\
        (Lx, Ly, unit_distance, dstops, num_areas_per_dimension, p_res, rhor_y, rvert_x, r3x, r3y)
        
        
    print ('Now in areas', areas)
    dem_nodes = nodes
    network= synergies.Network(nodes, dem_nodes, nodes_res, nodes_bus, routes, stops, dx, dy,
                      stops_per_route, areas_res, areas_com, areas, area_node, nodes_area,
                      depot, p_res)
    
    ## For levels of average demand per node: 
    case_study = synergies.CaseStudy(network, dem_av, peak_to_off,
                  c_tm, c_dwell, c_local, c_late, c_freq,
                  bus_types, cap_bus, fleet_bus, cost_fixed_bus, cost_km_bus, 
                  c_route, c_stop, vehicle_route, b_freq_peak_route, b_freq_off_route, distlim,
                  start, end, twlen, bus_start, bus_end, peak_periods, bus_speed, one_tw_per_node,
                  num_trucks_init, time_per_dem_unit, horizon, speed, truck_capacity, single_route_truck,
                  case_name_ind, f_meters_to_miles)
    
    ################################# BUS ONLY ####################################
    ### Check if demand can be served by passenger bus only.
    if case_study.tot_cap_basic < case_study.tot_dem_in:
        # Then we cannot serve the demand by bus
        print ('Total demand in range exceeds basic bus capacity')
    else:
        for tw in case_study.twindows:
            if case_study.tot_dem_nodes_bus_tw[tw] > case_study.tot_cap_basic_tw[tw]:
                print('Total business demand  cannot be served with basic capacity')
    # This is a sufficient but not necesary condition for infeasibility. We
    # have not checked which stops serve which nodes, so the total capacity could be 
    # sufficient in general but the specific requirements might make the
    # problem infeasible.
    new_exp_bus = synergies.Experiment(case_study, 'bus_only')
    case_study.exp_bus = new_exp_bus
    # Unwrap everything:
    Network = case_study.network
    
    Routes = Network.routes
    dem_nodes = Network.dem_nodes
    nodes_bus = Network.nodes_bus
    
    stops_per_route = Network.stops_per_route
    links_route = Network.links_route
    prec_stops_route_link = Network.prec_stops_route_link
    len_route_link = Network.len_route_link
    dist = Network.dist
    
    nodesInRange = case_study.nodes_range
    nodes_bus_range = case_study.nodes_bus_range
    pot_nodes = case_study.pot_nodes
    pot_stops = case_study.pot_stops
    dem_per_node = case_study.dem_per_node
    cap_basic = case_study.cap_basic
    twindows = case_study.twindows
    twindows_bus = case_study.twindows_bus
    twindows_peak = case_study.twindows_peak
    twindows_off = case_study.twindows_off
    demand = case_study.demand
    b_freq_tw = case_study.b_freq_tw
    b_freq_any_peak_tw = case_study.b_freq_any_peak_tw
    b_freq_any_off_tw = case_study.b_freq_any_off_tw
    vehicle_route = case_study.vehicle_route
    cap_bus = case_study.cap_bus
    fleet_bus = case_study.fleet_bus
    cost_fixed_bus = case_study.cost_fixed_bus
    time_route = case_study.time_route
    routes_type = case_study.routes_type
    cost_route_round = case_study.cost_route_round
    c_route = case_study.c_route
    c_stop = case_study.c_stop
    c_tm = case_study.c_tm
    c_dwell = case_study.c_dwell
    c_stop = case_study.c_stop
    c_local = case_study.c_local
    c_late = case_study.c_late  
    c_freq = case_study.c_freq
    single_route_truck = case_study.single_route_truck
    num_trucks_init = case_study.num_trucks_init
    time_per_dem_unit = case_study.time_per_dem_unit
    horizon = case_study.horizon
    speed = case_study.speed
    truck_capacity = case_study.truck_capacity
    tot_dem_out = case_study.tot_dem_out
    tot_truck_cap_init = case_study.tot_truck_cap_init
    
    print ('#########   NOW IN Experiment', new_exp_bus.name, '########')
    print ('################## Bus Only ##################')
     
    ### Get results.
    rvector = utilities.getModelBus(Routes,dem_nodes,  nodesInRange,
         nodes_bus_range, stops_per_route, pot_nodes, 
         pot_stops, dem_per_node, cap_basic, twindows, twindows_bus,
         twindows_peak, twindows_off, vehicle_route, cap_bus, fleet_bus,
         routes_type, links_route, prec_stops_route_link, len_route_link,
         b_freq_any_peak_tw, b_freq_any_off_tw,time_route, twlen, bus_types,
         demand, b_freq_tw,dist, c_route, c_stop, c_tm, c_dwell, c_local, c_late,c_freq,
         fmax, cost_route_round, b_freq_tw, cost_fixed_bus)

    if rvector is not None: # Model is feasible
        new_exp_bus.feasible_bus = True
        (m, X, df_peak_tw, df_off_tw, SUM1, SUM2, SUM3, SUM4, SUM5, SUM6, SUM8, SUM9) = rvector
        (stops_used, routes_used, stop_node_tw_route, stops_node, routes_node,
        df_peak_tw, df_off_tw) = utilities.getResults(X, df_peak_tw, df_off_tw,
           Routes, twindows, dem_nodes)
        case_study.stops_used = stops_used
        case_study.routes_used = routes_used
        case_study.stops_node_tw_route = stop_node_tw_route
        case_study.stops_node = stops_node
        case_study.routes_node = routes_node
        case_study.df_peak_tw = df_peak_tw
        case_study.df_off_tw = df_off_tw
        (case_study.freq_tw, case_study.freq_peak_hour,
                 case_study.freq_off_hour) = case_study.getFreqQ()
        (case_study.vmt_per_route, case_study.vmt_bus_total) = case_study.getVmtBus\
        (f_meters_to_miles, case_study.freq_tw)

    
        print('Capacity:', case_study.tot_cap_basic)
        print('Total demand in range:', case_study.tot_dem_in)
        print(df_peak_tw, df_off_tw)
       # Calculate VMT for nodes that cannot be served by buses and are served by trucks.  
        nodes_not_range = case_study.nodes_not_range
        if len(nodes_not_range) > 0:
            print ('###### Now we have out-of-range demand ########')
            if single_route_truck == 1:
                if tot_dem_out > tot_truck_cap_init:
                    print ('Total out-of-range demand exceeds truck capacity')
                    print ('Updating fleet size for out-of-range demand')
                    case_study.num_trucks_out_of_range = case_study.update_vehicles('out_of_range')
                    case_study.total_truck_cap_out_of_range = case_study.truck_cap('out_of_range')
                    print ('Updated number of trucks for out-of-range demand to', case_study.num_trucks_out_of_range)                    
                    print ('Now using trucks for out-of-range demand')
                else:
                    case_study.num_trucks_out_of_range = case_study.num_trucks_init
                    case_study.total_truck_cap_out_of_range = case_study.truck_cap('out_of_range')
            else:
                case_study.num_trucks_out_of_range = case_study.num_trucks_init
                case_study.total_truck_cap_out_of_range = case_study.truck_cap('out_of_range')
                
            num_trucks_out_of_range = case_study.num_trucks_out_of_range
            dx = case_study.network.dx
            dy = case_study.network.dy
            tw_start_end = case_study.tw_start_end
            depot = case_study.network.depot
            (locations_not_range, start_times_not_range, end_times_not_range, 
                 demands_not_range, node_tw_truck_node,nodes_not_range_All, 
                 nodes_not_range_reload, first_reload_off_id)\
                 = utilities.transformInput(nodes_not_range, dx, dy,
                 twindows, demand, twlen, tw_start_end, depot,
                 nodes_bus, twindows_bus, start, num_trucks_init, num_reloads_per_vehicle, horizon)

            data = [locations_not_range, demands_not_range, start_times_not_range, end_times_not_range]
                
            ### Start timing here:
            time_start = time.time()
            res = vehicle_routing.vrp(data, num_trucks_out_of_range, time_per_dem_unit,horizon,speed,
                       truck_capacity,first_reload_off_id)
            time_end = time.time()
            extratime = time_end - time_start
            (truck_routes_not_range, route_not_range_single, reload_off_range) = \
                    utilities.getTruckSolution(res, node_tw_truck_node, depot, 
                                    one_tw_per_node, nodes_not_range_reload)
            (vmt_tr_not_range_route, vmt_tr_not_range) =  utilities.getVmtTruck \
            (truck_routes_not_range, dist, f_meters_to_miles)
        else:
            extratime = 0
            vmt_tr_not_range = 0
            
        new_exp_bus.time = m.runtime + extratime ## Total solution time from gurobi and or tools
        new_exp_bus.vmt_truck = vmt_tr_not_range
        new_exp_bus.vmt_bus = case_study.vmt_bus_total
        new_exp_bus.vmt_total = new_exp_bus.vmt_truck + new_exp_bus.vmt_bus
        
        # Calculate emissions:
        (new_exp_bus.truck_fuel_gallons_used, new_exp_bus.bus_fuel_gallons_used, new_exp_bus.total_fuel_gallons_used,
        new_exp_bus.truck_co2_grams_emitted,new_exp_bus.bus_co2_grams_emitted, new_exp_bus.total_co2_grams_emitted) = \
             new_exp_bus.getEmissions(truck_fuel_gallons_per_100_miles, grams_co2_per_fuel_gallon,
             bus_miles_per_fuel_gallon, f_meters_to_miles)
#            experiments_feasible.append(new_exp_bus)
    else: #If rvector is None, i.e., the MILP was infeasible:
        new_exp_bus.feasible_bus = False
        
    # Calculate the "truck only" solution.
    print ('################## Truck Only ##################')
    if single_route_truck == 1:
        if case_study.tot_dem > case_study.tot_truck_cap_init:
             print ('Total demand exceeds total truck capacity')
             print ('Updating fleet size for total demand')
             case_study.num_trucks_only = case_study.update_vehicles('truck_only')
             case_study.total_truck_cap_only = case_study.truck_cap('truck_only')
             print ('Updated number of trucks for total demand to', case_study.num_trucks_only)
             print ('Now using trucks for total demand')
        else:
             case_study.num_trucks_only = case_study.num_trucks_init
             case_study.total_truck_cap_only = case_study.getTruckCap('initial')
    else:
        case_study.num_trucks_only = case_study.num_trucks_init
        case_study.total_truck_cap_only = case_study.getTruckCap('initial')
        
    num_trucks_only = case_study.num_trucks_only
#        total_truck_cap_only = case_study.total_truck_cap_only   
    new_exp_truck = synergies.Experiment(case_study, 'truck_only')
    print ('Now at experiment', new_exp_truck.name)
#        experiments.append(new_exp_truck)
    case_study.exp_truck = new_exp_truck
    dx = case_study.network.dx
    dy = case_study.network.dy
    tw_start_end = case_study.tw_start_end
    depot = case_study.network.depot
    
    # Modify the input appropriately:
    (locations, start_times,end_times, demands, node_tw_truck_node,
     nodes_All,nodes_Reload,first_reload_node_index) = utilities.transformInput(dem_nodes, dx, dy,
    twindows,demand,twlen,tw_start_end,depot,nodes_bus,twindows_bus, start, num_trucks_only, num_reloads_per_vehicle,
           horizon)

    data = [locations, demands, start_times, end_times]
    time_start_vrptw = time.time()
    res = vehicle_routing.vrp(data, num_trucks_only, time_per_dem_unit, horizon, speed,
                               truck_capacity, first_reload_node_index)
    time_end_vrptw = time.time()
    new_exp_truck.time = time_end_vrptw - time_start_vrptw
    (truck_routes, route_no_double_route_name, reload_route)\
        = utilities.getTruckSolution(res, node_tw_truck_node, depot, one_tw_per_node, nodes_Reload)
    (vmt_truck_onlyPerRoute, vmt_truck_only) =  utilities.getVmtTruck(truck_routes,dist,f_meters_to_miles)
    new_exp_truck.vmt_truck = vmt_truck_only
    new_exp_truck.vmt_bus = new_exp_truck.case_study.vmt_bus_basic_total
    new_exp_truck.vmt_total = new_exp_truck.vmt_truck + new_exp_truck.vmt_bus
    
    # Now we can calculate emissions:
    (new_exp_truck.truck_fuel_gallons_used, new_exp_truck.bus_fuel_gallons_used, new_exp_truck.total_fuel_gallons_used,
        new_exp_truck.truck_co2_grams_emitted,new_exp_truck.bus_co2_grams_emitted, new_exp_truck.total_co2_grams_emitted) = \
         new_exp_truck.getEmissions(truck_fuel_gallons_per_100_miles, grams_co2_per_fuel_gallon,
         bus_miles_per_fuel_gallon, f_meters_to_miles)
        
    return new_exp_bus, new_exp_truck, case_study

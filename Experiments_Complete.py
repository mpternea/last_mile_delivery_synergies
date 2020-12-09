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
import utilities
import Synergies_Classes
import Vehicle_Routing

import time
import matplotlib.pyplot as plt
import pickle
import os
#### Create a folder to temporarily save the experiments (as objects).
pickle_folder = 'pickled_stuff'
if not os.path.exists (pickle_folder):
    os.makedirs (pickle_folder)
    
#########################################################################
########################### PROBLEM PARAMETERS #########################
#########################################################################

#%% Parameters
### Standard  on Emissions
truck_fuel_gallons_per_100_miles = 6
grams_co2_per_fuel_gallon = 8.887
co2_per_CH4 = 25
co2_per_N20 = 298
co2_per_refrigerant = 1430
bus_miles_per_fuel_gallon = 7.2
f_meters_to_miles = 0.000621371

################ Network Information #####################
unit_distance = 100 # meters between consecutive nodes on the grid.
dstops = 1000 # Distance between consecutive bus stops on a route.
num_areas_per_dimension = 2 # number of large squares per dimension
# (borders not necessarily on nodes) - Squares define areas.
#p_res = 0.50 # percentage of residential areas (the remaining percentage is 
# for commercial - business areas)
rhor_y = [0.5] # The horizontal route is located at half the height of the area
rvert_x = [0.5] # The vertical route is located at half the width of the area.
r3x = 0.5 # r3x and r3y indicate where the 3rd route is located.
r3y = [0.1,0.8]
# The third route is shaped as
#       _______
# _____|

################ Time Windows #############################
start = 8 # Beginning of first time window
end = 12 # end of last time window.
twlen = 2 # Length of time window (hours)
bus_start = 10 # start of business period
bus_end = 16 # end of business period
peak_periods = [(8,10), (16,18)] # start and end of peak periods
peak_to_off = 1.5 # High Vs Low average demand according to node type and time of day.
# "Peak" here does not refer to traffic or bus frequency peak.

################ Objective Function Costs #####################
route_costs = [30000,35000,40000]  #1 Cost of using a route.
stop_costs = [7000,7500,8000] #2 Cost of using a stop
c_tm = 455 #3  Cost per tonne mile traveled for each bus.
c_dwell = 15 # 4 Cost per package to unload at a stop.
c_local = 5 # 5 Cost of assigning a demand node to a stop for local delivery.
c_late = 10 # 6 Cost of satisfying demand after its time window.
c_freq = 500 # 7 Cost of frequency (now I use # 9 instead, as an approximation for operational cost)
cost_fixed_bus = {'BusType1':300000} # 8, Fixed cost for each bus (dollars per vehicle)
cost_km_bus = {'BusType1': 50} # 9 Operating cost for each bus (dollars per km)

################### Bus Information ###############################
bus_types = ['BusType1'] # Available bus types
cap_bus = {'BusType1':30}
fleet_bus = {'BusType1': 40} # Available fleet for each bus type
f_peak_range = [4,5] # Bus frequency for peak hours
f_off_range = [2,3] # Bus frequency for off peak hours.
fmax = 10 # Maximum hourly frequency
bus_speed = 30 # km/h #
seed = 1234 #Random seed, used for generating case studies
distlim = 600 # Maximum range within which a node can be served by a bus stop.
one_tw_per_node = 0 # The demand of a node can be accumulated in 1 time window (1) or in many (0).

################### Truck Information ###############################
single_route_truck = 1 # Indicates whether each truck can return to the depot.
num_trucks_init = 30 # Number of available trucks
time_per_dem_unit = 10 # time per demand unit
horizon = 20*3600 # Maximum allowed trip length
speed = 25 # Vehicle speed in meters per second
truck_capacity = 150 # capacity of each vehicle in packages
num_reloads_per_vehicle = 2

################# Potential parameter ranges ###################################### 
# A list contains the values of each parameter during sensitivity analysis.
#L_Range = [20] #Lx and Ly, in nodes
#DistLimRange = [150,200,250,300,350] # meters: How far from a stop can a node be to be considered for bus service?
Lx = 10
Ly = 10
#dem_av_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # # Average demand per node and day
dem_av_range = [1,2,3,4,5]
p_res_range = [0.5]

#%% Experiments

casenameind_init = 1 # Index for the name of the 1st Experiment
#DictRes = {} # contains results
solution = {} # feasible or not
runtime = {} # Will contain running time for every Experiment
gap = {} # Will contain MIP gap for every experiment

case_studies = []
Experiments = []
experiments_feasible = []
casenameind = casenameind_init # 
netname_ind = 1

## For loop: Change the percentage of residential areas.
for p_res in p_res_range:
    ################################# GRID AND NETWORK ################################
    
    ### Create the network.
    (nodes,nodes_res, nodes_bus, routes, stops, DX, DY, stops_per_route,
         areas_res, areas_com, areas, area_node, nodes_area, depot) = utilities.getGrid\
        (Lx, Ly, unit_distance, dstops, num_areas_per_dimension, p_res, rhor_y, rvert_x, r3x, r3y)
        
        
    print ('Now in areas', areas)
    dem_nodes = nodes
    network= Synergies_Classes.Network (nodes, dem_nodes, nodes_res, nodes_bus, routes, stops, DX, DY,
                      stops_per_route, areas_res, areas_com, areas, area_node, nodes_area,
                      depot, p_res)
    
    network.name = 'Network'+str(netname_ind)
    netname_ind = netname_ind + 1
    ## For levels of average demand per node: 
    for demav in dem_av_range:
        CaseStudy = Synergies_Classes.CaseStudy (network, demav, peak_to_off, route_costs,
                      stop_costs, c_tm, c_dwell, c_local, c_late, c_freq,
                      bus_types, cap_bus, fleet_bus, cost_fixed_bus, cost_km_bus, distlim,
                      f_peak_range, f_off_range, start, end, twlen,
                      bus_start, bus_end, peak_periods, seed, bus_speed, one_tw_per_node,
                      num_trucks_init, time_per_dem_unit, horizon, speed, truck_capacity, single_route_truck,
                      casenameind, f_meters_to_miles)
        case_studies.append(CaseStudy)
        casenameind = casenameind + 1    
        ################################# BUS ONLY ####################################
        ### Check if demand can be served by passenger bus only.
        if CaseStudy.tot_cap_basic < CaseStudy.tot_dem_in:
            # Then we cannot serve the demand by bus
            print ('Total demand in range exceeds basic bus capacity')
        else:
            for tw in CaseStudy.twindows:
                if CaseStudy.tot_dem_nodes_bus_tw[tw] > CaseStudy.tot_cap_basic_tw[tw]:
                    print('Total business demand  cannot be served with basic capacity')
        # This is a sufficient but not necesary condition for infeasibility. We
        # have not checked which stops serve which nodes, so the total capacity could be 
        # sufficient in general but the specific requirements might make the
        # problem infeasible.
        new_exp = Synergies_Classes.Experiment (CaseStudy, 'BUS_ONLY')
        Experiments.append(new_exp)
        CaseStudy.exper_list.append(new_exp)
        # Unwrap everything:
        Scheme =  new_exp.scheme 
        Network = CaseStudy.network
        
        Routes = Network.routes
        dem_nodes = Network.dem_nodes
        nodes_bus = Network.nodes_bus
        
        stops_per_route = Network.stops_per_route
        links_route = Network.links_route
        prec_stops_route_link = Network.prec_stops_route_link
        len_route_link = Network.len_route_link
        dist = Network.dist
        
        nodesInRange = CaseStudy.nodes_range
        nodes_bus_range = CaseStudy.nodes_bus_range
        pot_nodes = CaseStudy.pot_nodes
        pot_stops = CaseStudy.pot_stops
        dem_per_node = CaseStudy.dem_per_node
        cap_basic = CaseStudy.cap_basic
        twindows = CaseStudy.twindows
        twindows_bus = CaseStudy.twindows_bus
        twindows_peak = CaseStudy.twindows_peak
        twindows_off = CaseStudy.twindows_off
        demand = CaseStudy.demand
        tw_w_demand = CaseStudy.tw_w_demand
        b_freq_tw = CaseStudy.b_freq_tw
        b_freq_any_peak_tw = CaseStudy.b_freq_any_peak_tw
        b_freq_any_off_tw = CaseStudy.b_freq_any_off_tw
        vehicle_route = CaseStudy.vehicle_route
        cap_bus = CaseStudy.cap_bus
        fleet_bus = CaseStudy.fleet_bus
        cost_fixed_bus = CaseStudy.cost_fixed_bus
        time_route = CaseStudy.time_route
        routes_type = CaseStudy.routes_type
        cost_route_round = CaseStudy.cost_route_round
        c_route = CaseStudy.c_route
        c_stop = CaseStudy.c_stop
        c_tm = CaseStudy.c_tm
        c_dwell = CaseStudy.c_dwell
        c_stop = CaseStudy.c_stop
        c_local = CaseStudy.c_local
        c_late = CaseStudy.c_late  
        c_freq = CaseStudy.c_freq
        # Unwrap truck info too, it does not take long and also why do it twice later?
        single_route_truck = CaseStudy.single_route_truck
        num_trucks_init = CaseStudy.num_trucks_init
        time_per_dem_unit = CaseStudy.time_per_dem_unit
        horizon = CaseStudy.horizon
        speed = CaseStudy.speed
        truck_capacity = CaseStudy.truck_capacity
        tot_dem_out = CaseStudy.tot_dem_out
        tot_truck_cap_init = CaseStudy.tot_truck_cap_init
        
        tot_dem = CaseStudy.tot_dem
        print ('#########   NOW IN Experiment', new_exp.name, '########')
        print ('################## Bus Only ##################')
         
        ### Get results.
        rvector = utilities.getModelBus (Routes,dem_nodes,  nodesInRange, nodes_bus_range, stops_per_route, pot_nodes, 
             pot_stops, dem_per_node, cap_basic, twindows, twindows_bus,
             twindows_peak, twindows_off, vehicle_route, cap_bus, fleet_bus,
             routes_type, links_route, prec_stops_route_link, len_route_link,
             b_freq_any_peak_tw, b_freq_any_off_tw,time_route, twlen, bus_types,
             demand, b_freq_tw,dist, c_route, c_stop, c_tm, c_dwell, c_local, c_late,c_freq,
             fmax, cost_route_round, b_freq_tw, cost_fixed_bus)
        
        
        
        ################################     RESULTS      #############################
        
        if rvector is not None: # Model is feasible
            (m, X, df_peak_tw, df_off_tw, SUM1, SUM2, SUM3, SUM4, SUM5, SUM6, SUM8, SUM9) = rvector
            (stops_used, routes_used, stop_node_tw_route, stops_node, routes_node,
            df_peak_tw, df_off_tw) = utilities.getResults (X, df_peak_tw, df_off_tw,
               Routes, twindows, dem_nodes)
            CaseStudy.stops_used = stops_used
            CaseStudy.routes_used = routes_used
            CaseStudy.stops_node_tw_route = stop_node_tw_route
            CaseStudy.stops_node = stops_node
            CaseStudy.routes_node = routes_node
            CaseStudy.df_peak_tw = df_peak_tw
            CaseStudy.df_off_tw = df_off_tw
            (CaseStudy.freq_tw, CaseStudy.freq_peak_hour,
                     CaseStudy.freq_off_hour) = CaseStudy.getFreqQ()
            (CaseStudy.vmt_per_route, CaseStudy.vmt_bus_total) = CaseStudy.getVmtBus(f_meters_to_miles, CaseStudy.freq_tw)
    
            solution[new_exp] = 'OK'
            runtime[new_exp] = m.runtime
            gap[new_exp] = m.MIPGap # The final gap after optimization, not the allowed
        

            print ('Capacity:', CaseStudy.tot_cap_basic)
            print ('Total demand in range:', CaseStudy.tot_dem_in)
            print (df_peak_tw, df_off_tw)
#          Calculate VMT for nodes that cannot be served by buses and are served by trucks.  
            nodes_not_range = CaseStudy.nodes_not_range
            if len(nodes_not_range) > 0:
                print ('###### Now we have out-of-range demand ########')
                if single_route_truck == 1:
                    if tot_dem_out > tot_truck_cap_init:
                        print ('TOTAL OUT-OF-RANGE-DEMAND EXCEEDS INITIAL TRUCK CAPACITY')
                        print ('Updating fleet size for out-of-range demand')
                        CaseStudy.num_trucks_out_of_range = CaseStudy.update_vehicles ('out_of_range')
                        CaseStudy.total_truck_cap_out_of_range = CaseStudy.truck_cap ('out_of_range')
                        print ('Updated number of trucks for out-of-range demand to', CaseStudy.num_trucks_out_of_range )                    
                        print ('Now using trucks for out-of-range demand')
                    else:
                        CaseStudy.num_trucks_out_of_range = CaseStudy.num_trucks_init
                        CaseStudy.total_truck_cap_out_of_range = CaseStudy.truck_cap('out_of_range')
                else:
                    CaseStudy.num_trucks_out_of_range = CaseStudy.num_trucks_init
                    CaseStudy.total_truck_cap_out_of_range = CaseStudy.truck_cap('out_of_range')
                    
                num_trucks_out_of_range = CaseStudy.num_trucks_out_of_range
                total_truck_cap_out_of_range = CaseStudy.total_truck_cap_out_of_range
                DX = CaseStudy.network.DX
                DY = CaseStudy.network.DY
                tw_start_end = CaseStudy.tw_start_end
                depot = CaseStudy.network.depot
                (locations_not_range, start_times_not_range, end_times_not_range, 
                     demands_not_range, node_tw_truck_node,nodes_not_range_All, 
                     nodes_not_range_reload, first_reload_off_id)\
                     = utilities.transformInput (nodes_not_range, DX, DY,
                     twindows, demand, twlen, tw_start_end, depot,
                     nodes_bus, twindows_bus, start, num_trucks_init, num_reloads_per_vehicle, horizon)

                data = [locations_not_range, demands_not_range, start_times_not_range, end_times_not_range]
                    
                ### Start timing here:
                time_start = time.time()
                RES = Vehicle_Routing.main (data, num_trucks_out_of_range, time_per_dem_unit,horizon,speed,
                           truck_capacity,first_reload_off_id)
                time_end = time.time()
                extratime = time_end - time_start
                (truck_routes_not_range, route_not_range_single, reload_off_range) = \
                        utilities.getTruckSolution (RES, node_tw_truck_node, depot, 
                                        one_tw_per_node, nodes_not_range_reload)
                (vmt_tr_not_range_route, vmt_tr_not_range) =  utilities.getVmtTruck \
                (truck_routes_not_range, dist, f_meters_to_miles)
            else:
                extratime = 0
                vmt_tr_not_range = 0
            new_exp.time = m.runtime + extratime ## Total solution time from gurobi and or tools
            new_exp.vmt_truck = vmt_tr_not_range
            new_exp.vmt_bus = CaseStudy.vmt_bus_total
            new_exp.vmt_total = new_exp.vmt_truck + new_exp.vmt_bus
            
            # Calculate emissions:
            (new_exp.truck_fuel_gallons_used, new_exp.bus_fuel_gallons_used, new_exp.total_fuel_gallons_used,
            new_exp.truck_co2_grams_emitted,new_exp.bus_co2_grams_emitted, new_exp.total_co2_grams_emitted) = \
                 new_exp.getEmissions (truck_fuel_gallons_per_100_miles, grams_co2_per_fuel_gallon,
                 bus_miles_per_fuel_gallon, f_meters_to_miles)
            experiments_feasible.append(new_exp)
            pickle.dump (new_exp, open (pickle_folder + '/' + new_exp.name +'.p', 'wb'))
        else:
            solution[new_exp] = 'NONE'
        # Calculate the "truck only" solution.
        print ('################## Truck Only ##################')
        if single_route_truck == 1:
            if CaseStudy.tot_dem > CaseStudy.tot_truck_cap_init:
                 print ('Total demand exceeds total truck capacity')
                 print ('Updating fleet size for total demand')
                 CaseStudy.num_trucks_only = CaseStudy.update_vehicles('truck_only')
                 CaseStudy.total_truck_cap_only = CaseStudy.truck_cap('truck_only')
                 print ('Updated number of trucks for total demand to', CaseStudy.num_trucks_only)
                 print ('Now using trucks for total demand')
            else:
                 CaseStudy.num_trucks_only = CaseStudy.num_trucks_init
                 CaseStudy.total_truck_cap_only = CaseStudy.getTruckCap('initial')
        else:
            CaseStudy.num_trucks_only = CaseStudy.num_trucks_init
            CaseStudy.total_truck_cap_only = CaseStudy.getTruckCap('initial')
            
        num_trucks_only = CaseStudy.num_trucks_only
        total_truck_cap_only = CaseStudy.total_truck_cap_only   
        new_exp = Synergies_Classes.Experiment (CaseStudy, 'TRUCK_ONLY')
        print ('Now at experiment', new_exp.name)
        Experiments.append(new_exp)
        CaseStudy.exper_list.append(new_exp)
        DX = CaseStudy.network.DX
        DY = CaseStudy.network.DY
        tw_start_end = CaseStudy.tw_start_end
        depot = CaseStudy.network.depot
        
        # Modify the input appropriately:
        (locations, start_times,end_times, demands, node_tw_truck_node,
         nodes_All,nodes_Reload,first_reload_node_index) = utilities.transformInput (dem_nodes, DX, DY,
        twindows,demand,twlen,tw_start_end,depot,nodes_bus,twindows_bus, start, num_trucks_only, num_reloads_per_vehicle,
               horizon)

        data = [locations, demands, start_times, end_times]
        time_start_vrptw = time.time()
        RES = Vehicle_Routing.main(data, num_trucks_only, time_per_dem_unit, horizon, speed,
                                   truck_capacity, first_reload_node_index)
        time_end_vrptw = time.time()
        experiments_feasible.append(new_exp)
        new_exp.time = time_end_vrptw - time_start_vrptw
        (truck_routes, DictRouteNoDoublesPerRouteName, DictReloadPerRoute)\
            = utilities.getTruckSolution (RES, node_tw_truck_node, depot, one_tw_per_node, nodes_Reload)
        (vmt_truck_onlyPerRoute, vmt_truck_only) =  utilities.getVmtTruck (truck_routes,dist,f_meters_to_miles)
        new_exp.vmt_truck = vmt_truck_only
        new_exp.vmt_bus = new_exp.CaseStudy.vmt_bus_basic_total
        new_exp.vmt_total = new_exp.vmt_truck + new_exp.vmt_bus
        
        # Now we can calculate emissions:
        (new_exp.truck_fuel_gallons_used, new_exp.bus_fuel_gallons_used, new_exp.total_fuel_gallons_used,
            new_exp.truck_co2_grams_emitted,new_exp.bus_co2_grams_emitted, new_exp.total_co2_grams_emitted) = \
             new_exp.getEmissions (truck_fuel_gallons_per_100_miles, grams_co2_per_fuel_gallon,
             bus_miles_per_fuel_gallon, f_meters_to_miles)
        pickle.dump (new_exp, open (pickle_folder + '/' + new_exp.name +'.p', 'wb'))
    
###### Group Experiments
exp_truck_only = []
exp_bus_only = [] # Plus truck for unreachable nodes
for exp in Experiments:
    scheme = exp.scheme
    if scheme == 'TRUCK_ONLY':
        exp_truck_only.append(exp)
    elif scheme == 'BUS_ONLY':
        exp_bus_only.append(exp)
       
############################ Plot #################################
### Plots VMT as a function of average demand.
x = [case.tot_dem for case in case_studies]

vmt_total_schemebus_list = [exp.vmt_total for exp in exp_bus_only]
vmt_bus_schemebus_list = [exp.vmt_bus for exp in exp_bus_only]
vmt_truck_not_range_scheme_bus_list = [exp.vmt_truck for exp in exp_bus_only]

vmt_total_SchemeTruck_List = [exp.vmt_total for exp in exp_truck_only]
vmt_bus_SchemeTruck_List = [exp.vmt_bus for exp in exp_truck_only]
vmt_truck_SchemeTruck_List = [exp.vmt_truck for exp in exp_truck_only]


plt.plot (x, vmt_total_schemebus_list, 'ro', label = 'New Scheme: VMT Total')
plt.plot (x, vmt_bus_schemebus_list, 'bo', label = 'New Scheme: VMT Bus')
plt.plot (x, vmt_truck_not_range_scheme_bus_list, 'go', label = 'New Scheme: VMT Truck')

plt.plot (x, vmt_total_SchemeTruck_List, 'rs', label = 'Truck Scheme: VMT Total')
plt.plot (x, vmt_bus_SchemeTruck_List, 'bs',  label = 'Truck Scheme: VMT Bus')
plt.plot (x, vmt_truck_SchemeTruck_List, 'gs', label = 'Truck Scheme: VMT Truck')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           mode="expand", borderaxespad=0)
plt.xlabel('Total Demand')
plt.ylabel('Total Vehicle Meters Travelled')
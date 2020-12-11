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
import matplotlib.pyplot as plt
import main
    
#########################################################################
########################### PROBLEM PARAMETERS #########################
#########################################################################

#%% Parameters
### Standard  on Emissions
truck_fuel_gallons_per_100_miles = 6
grams_co2_per_fuel_gallon = 8.887
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


c_route = {'R3': 35000, 'R2': 30000, 'R1': 30000} # Cost for using each route
c_stop = {('R3', 12): 7500, ('R2', 6): 8000, ('R1', 56): 8000, ('R3', 94): 8000, ('R2', 116): 7500, ('R1', 66): 7500}
vehicle_route = {'R3': 'BusType1', 'R2': 'BusType1', 'R1': 'BusType1'}
b_freq_peak_route = {'R3': 4, 'R2': 5, 'R1': 4}
b_freq_off_route = {'R3': 2, 'R2': 2, 'R1': 2}
p_res = 0.5 # Proportion of residential areas (for random generation of area)

################### Bus Information ###############################
bus_types = ['BusType1', 'BusType2'] # Available bus types
cap_bus = {'BusType1':30, 'BusType2': 34}
fleet_bus = {'BusType1': 40, 'BusType2':20} # Available fleet for each bus type
cost_fixed_bus = {'BusType1':300000, 'BusType2': 150000} # 8, Fixed cost for each bus (dollars per vehicle)
cost_km_bus = {'BusType1': 50, 'BusType2':100} # 9 Operating cost for each bus (dollars per km)
f_peak_range = [4,5] # Bus frequency for peak hours
f_off_range = [2,3] # Bus frequency for off peak hours.
fmax = 10 # Maximum hourly frequency
bus_speed = 30 # km/h #
distlim = 600 # Maximum range within which a node can be served by a bus stop.
one_tw_per_node = 0 # The demand of a node can be accumulated in 1 time window (1) or in many (0).
################### Truck Information ###############################
single_route_truck = 1 # Indicates whether each truck can return to the depot.
num_trucks_init = 30 # Number of available trucks
time_per_dem_unit = 10 # time per demand unit
horizon = 20*3600 # Maximum allowed trip duration (minutes)
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
dem_av_range = [1]


#%% experiments
dem_av_range = [0.2, 0.5, 1, 2, 3, 4, 10]
solution = {} # feasible or not
runtime = {} # Will contain running time for every Experiment
gap = {} # Will contain MIP gap for every experiment
case_studies = []
exp_bus_feasible = []

## For loop: Change the percentage of residential areas.
case_name_ind = 1 # Index for the name of the 1st Experiment
net_name_ind = 1



args  = {"Lx": Lx, "Ly": Ly, "unit_distance": unit_distance, "dstops": dstops,
         "num_areas_per_dimension":num_areas_per_dimension,
         "rhor_y": rhor_y, "rvert_x": rvert_x, "r3x": r3x, "r3y": r3y,
         "p_res": p_res, "peak_to_off": peak_to_off, "route_costs": route_costs, "stop_costs": stop_costs,
         "c_tm": c_tm, "c_dwell": c_dwell, "c_local": c_local, "c_late": c_late, "c_freq": c_freq,
          "bus_types": bus_types, "cap_bus": cap_bus, "fleet_bus": fleet_bus, "cost_fixed_bus": cost_fixed_bus,
          "cost_km_bus": cost_km_bus, "c_route": c_route, "c_stop": c_stop, "vehicle_route": vehicle_route,
          "b_freq_peak_route": b_freq_peak_route, "b_freq_off_route":b_freq_off_route,
          "distlim": distlim,"f_peak_range": f_peak_range, "f_off_range": f_off_range,
          "start": start, "end": end, "twlen": twlen,"bus_start": bus_start, "bus_end": bus_end,
          "peak_periods": peak_periods, "bus_speed": bus_speed,"one_tw_per_node": one_tw_per_node,
          "num_trucks_init": num_trucks_init, "time_per_dem_unit": time_per_dem_unit, "horizon": horizon,
          "speed": speed, "truck_capacity": truck_capacity, "single_route_truck": single_route_truck,
          "f_meters_to_miles": f_meters_to_miles, "fmax": fmax,
          "num_reloads_per_vehicle": num_reloads_per_vehicle,
          "truck_fuel_gallons_per_100_miles": truck_fuel_gallons_per_100_miles,
          "grams_co2_per_fuel_gallon": grams_co2_per_fuel_gallon,
          "bus_miles_per_fuel_gallon": bus_miles_per_fuel_gallon}
          
exp_truck_only = []
exp_bus_only = [] # Plus truck for unreachable nodes
exp_bus_feasible = []                   
for (i, dem_av) in enumerate(dem_av_range):
    args["dem_av"] = dem_av
    args["case_name_ind"] = i 
    new_exp_bus, new_exp_truck, case_study =  main.getmain(args)
    exp_truck_only.append(new_exp_truck)
    exp_bus_only.append(new_exp_bus)
    case_studies.append(case_study)
    if new_exp_bus.feasible_bus:
        exp_bus_feasible.append(new_exp_bus)
    
    

       
## Plot VMT as a function of average demand for case studies with a feasible bus scheme.
cases_plot = [case for case in case_studies if case.exp_bus.feasible_bus == True]
x = [case.tot_dem for case in cases_plot]

vmt_total_schemebus_list = [exp.vmt_total for exp in exp_bus_only]
vmt_bus_schemebus_list = [exp.vmt_bus for exp in exp_bus_only]
vmt_truck_not_range_scheme_bus_list = [exp.vmt_truck for exp in exp_bus_only]

vmt_total_SchemeTruck_List = [exp.vmt_total for exp in exp_truck_only]
vmt_bus_SchemeTruck_List = [exp.vmt_bus for exp in exp_truck_only]
vmt_truck_SchemeTruck_List = [exp.vmt_truck for exp in exp_truck_only]


plt.plot(x, vmt_total_schemebus_list, 'ro', label = 'New Scheme: VMT Total')
plt.plot(x, vmt_bus_schemebus_list, 'bo', label = 'New Scheme: VMT Bus')
plt.plot(x, vmt_truck_not_range_scheme_bus_list, 'go', label = 'New Scheme: VMT Truck')

plt.plot(x, vmt_total_SchemeTruck_List, 'rs', label = 'Truck Scheme: VMT Total')
plt.plot(x, vmt_bus_SchemeTruck_List, 'bs',  label = 'Truck Scheme: VMT Bus')
plt.plot(x, vmt_truck_SchemeTruck_List, 'gs', label = 'Truck Scheme: VMT Truck')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc = 3,
           mode="expand", borderaxespad=0)
plt.xlabel('Total Demand')
plt.ylabel('Total Vehicle Meters Travelled')
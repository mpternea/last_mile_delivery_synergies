# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 08:38:15 2017

Containts the 3 main classes: EXPERIMENT, CaseStudy and NETWORK.

A Network object is determined by the underlying grid, with given demand nodes,
distances, bus routes and stops. The grid also contains the location of the 
depot.

A CaseStudy object contains all the operational characteristics of the operating
schemes, for a given network. For the "bus only" scheme, it has properties like average demand,
costs, frequencies, time window information, etc. For the "truck only" scheme,
it has properties like truck capacity, truck speed, etc.

An Experiment object is identified by the CaseStudy it refers to, as well as 
the operating scheme under consideration.

So, for example, when we are performing sensitivity analysis in a given network
to test the impact of demand fluctuation, we create new CaseStudy objects and,
absed on them, one EXPERIMENT object for each scheme.


"""
import random 
import copy
import math
import utilities

class Experiment:
    def __init__ (self, CaseStudy, scheme):
        self.CaseStudy = CaseStudy # CaseStudy: Specific, given data (not just the network)
        self.scheme = scheme
        self.name = 'EXP_' + str(CaseStudy.name) + '_' + scheme
        self.vmt_total = None
        self.vmt_truck = None
        self.vmt_bus = None
        self.demand = CaseStudy.tot_dem
        self.time = None
        self.feasible = None
        self.truck_fuel = None
        self.bus_fuel = None
        self.total_fuel = None
        self.truck_co2 = None
        self.bus_co2 = None
        self.total_co2 = None

    def __repr__(self):
            return "<Exp %s>" % (self.name)
    def getEmissions (self,truck_fuel_gallons_per_100_miles, grams_co2_per_fuel_gallon,
                 bus_miles_per_fuel_gallon, f_meters_to_miles):

    # Unwrap properties
        vmt_truck = self.vmt_truck
        vmt_bus = self.vmt_bus
        bus_fuel_gallons_per_mile = 1 / bus_miles_per_fuel_gallon
        truck_fuel = vmt_truck * truck_fuel_gallons_per_100_miles / 100
        bus_fuel = vmt_bus * bus_fuel_gallons_per_mile
        total_fuel = truck_fuel + bus_fuel
        truck_co2 = grams_co2_per_fuel_gallon * truck_fuel
        bus_co2 = grams_co2_per_fuel_gallon * bus_fuel
        total_co2 = truck_co2 + bus_co2
        return (truck_fuel, bus_fuel, total_fuel, truck_co2, bus_co2, total_co2)

        
class CaseStudy:
    def __init__ (self, network, demav, peak_to_off, route_costs, stop_costs, c_tm, c_dwell, c_local, c_late, c_freq,
                  bus_types, cap_bus, fleet_bus, cost_fixed_bus, cost_km_bus, distlim, 
                  f_peak_range, f_off_range, start, end, tw_len,
                  bus_start, bus_end, peak_periods,seed,bus_speed,only_one_tw_per_node,
                  num_trucks_init, time_per_dem_unit, horizon, speed, truck_capacity,single_route_truck,
                  case_id, f_meters_to_miles): # specific values for costs, not the random ranges
    
        self.network = network
        self.demav = demav
        self.peak_to_off = peak_to_off
        self.route_costs = route_costs
        self.stop_costs = stop_costs
        self.c_tm = c_tm
        self.c_dwell = c_dwell
        self.c_local = c_local
        self.c_late = c_late
        self.c_freq = c_freq
        self.bus_types = bus_types
        self.cap_bus = cap_bus
        self.fleet_bus = fleet_bus
        self.cost_fixed_bus = cost_fixed_bus
        self.cost_km_bus = cost_km_bus
        self.distlim = distlim
        self.f_peak_range = f_peak_range
        self.f_off_range = f_off_range
        self.start = start
        self.end = end
        self.tw_len = tw_len
        self.bus_start = bus_start 
        self.bus_end = bus_end 
        self.peak_periods = peak_periods # peak_periods = [(7,9), (16,18)]      
        self.seed = seed
        self.bus_speed = bus_speed
        self.one_tw_per_node = only_one_tw_per_node
        # Truck data
        self.num_trucks_init = num_trucks_init # Number of trucks
        self.time_per_dem_unit = time_per_dem_unit 
        self.horizon = horizon # Longest truck trip duration
        self.speed = speed # Truck speed
        self.truck_capacity = truck_capacity # Truck Capacity
        self.single_route_truck = single_route_truck
        self.tot_truck_cap_init = self.getTruckCap('initial')
        
        self.name = 'CASE' + str(case_id)
        (self.pot_nodes, self.pot_stops) = self.getPotential()
        (self.nodes_range, self.nodes_not_range) = self.getInrange()
        (self.twindows, self.twindows_bus, self.twindows_peak, self.twindows_off, self.tw_start_end) = self.getWindows()
        (self.demand,self.dem_per_node,self.tw_w_demand) = self.getDemand()
        self.time_route = self.getTimeR() # Hours
        self.tot_dem = sum(self.demand[k,t] for k in network.nodes for t in self.twindows)
        self.tot_dem_in = sum(self.demand[k,t] for k in self.nodes_range for t in self.twindows)
        self.tot_dem_out = self.tot_dem - self.tot_dem_in
        self.nodes_bus_range = set(self.nodes_range) & set(self.network.nodes_bus)
        self.TotDemBusInRange = sum(self.demand[k,t] for k in  self.nodes_bus_range for t in self.twindows) 
        self.tot_dem_nodes_bus_tw = self.getDemBusTW() # I might not be using this
        (self.c_route, self.c_stop, self.vehicle_route, self.b_freq_peak_route, self.b_freq_off_route) = self.getRandomStuff()
        self.routes_type = self.getRouteType()
        (self.b_freq_tw, self.b_freq_any_peak_tw, self.b_freq_any_off_tw) = self.getFreqQ ()
        (self.cap_basic, self.tot_cap_basic_tw, self.tot_cap_basic) = self.getCap()
        self.cost_route_round = self.getCostRr ()
        (self.vmt_basic_route, self.vmt_bus_basic_total) = self.getVmtBus(f_meters_to_miles, self.b_freq_tw)
        self.exper_list  = []
#        (self.vmt_route, self.vmt_bus_total) = self.getVmtB ()

    def __repr__(self):
        return "<Case %s>" % (self.name)
        
    def updateVehicles (self,case):
        # Increases total truck capacity by updating the necessary number
        # of trucks (or by using larger trucks --> Not good for comparison).
        truck_capacity = self.truck_capacity
        if case == 'out_of_range':
            tot_dem = self.tot_dem_out
        elif case == 'truck_only':
            tot_dem = self.tot_dem
        total_truckCap_current = self.tot_truck_cap_init
        # Find the difference:
        extra_dem = tot_dem - total_truckCap_current
        # Find the required number of extra trucks
        extra_trucks = math.ceil(extra_dem / truck_capacity)
        new_num_trucks = self.num_trucks_init + extra_trucks
        return (new_num_trucks)  
        
    def getTruckCap (self, case): # case: 'initial', 'out_of_range', 'truck_only'
        if case == 'initial':
            num_vehicles = self.num_trucks_init
        elif case == 'out_of_range':
            num_vehicles = self.num_trucks_out_of_range
        elif case == 'truck_only':
            num_vehicles = self.num_trucks_only
        truck_capacity = self.truck_capacity
        total_truckCap = num_vehicles * truck_capacity
        return (total_truckCap)
        
    def getWindows (self):
        start = self.start
        end = self.end
        tw_len = self.tw_len
        bus_start = self.bus_start
        bus_end = self.bus_end
        peak_periods = self.peak_periods
        # Calculate stime windows, business, peak, according to data
        if (end-start)%tw_len !=0:
            numtw = int((end-start)/tw_len) +1 # The last time window will be smaller
        else:
            numtw = int((end-start)/tw_len)
        twindows = [i for i in range (1, int(numtw+1))]
        tw_start = {}
        tw_end=  {}
        for (ind,t) in enumerate(twindows):
            tw_start[t] = start + ind*tw_len
            tw_end[t] = tw_start[t] + tw_len
        if tw_end[twindows[-1]] > end: # We have reached the final time windows
                tw_end[twindows[-1]] = end
        tw_start_end = {tw: (tw_start[tw], tw_end[tw]) for tw in twindows}      
        twindows_bus = [i for i in twindows if tw_start[i] >= bus_start and  tw_end[i] <= bus_end]
        twindows_peak = []
        for period in peak_periods:
            start = period[0]
            end = period[1]
            twindows_peak.extend ([tw for tw in twindows if tw_start[tw] >= start and tw_end[tw] <= end])
        twindows_off = [tw for tw in twindows if tw not in twindows_peak]
        return (twindows, twindows_bus, twindows_peak, twindows_off, tw_start_end)
    
    def getPotential (self):
        network = self.network
        distlim = self.distlim
        dem_nodes = network.dem_nodes
        dist = network.dist
        stops = network.stops
        pot_nodes = {stop:[i for i in dem_nodes if dist[stop,i] <= distlim] for stop in stops}
        pot_stops = {node:[stop for stop in stops if node in pot_nodes[stop]] for node in dem_nodes}
        return (pot_nodes, pot_stops)
        
    def getInrange (self):
        network = self.network
        dem_nodes = network.dem_nodes
        pot_stops = self.pot_stops
        nodes_not_range = []
        for demnode in dem_nodes:
            if pot_stops[demnode] == []:
#                print ('BE CAREFUL! NODE', demnode, 'UNSERVED DUE TO DISTANCE!')
                nodes_not_range.append(demnode)
        nodes_range = list(set(dem_nodes) - set(nodes_not_range))
        return (nodes_range, nodes_not_range)
              
    def getDemand (self):
        one_tw_per_node = self.one_tw_per_node
        network = self.network
        demav = self.demav
        peak_to_off = self.peak_to_off
        twindows = self.twindows
        twindows_bus = self.twindows_bus
        dem_nodes = network.dem_nodes
        areas = network.areas
        AreasCom = network.AreasCom
        areas_res = network.areas_res
        nodes_area = network.nodes_area
        tw_no_bus = [tw for tw in twindows if tw not in twindows_bus]  
        tot_dem = round(demav*len(dem_nodes),0)
        # If the demand of a node is spread across multiple time windows
        if one_tw_per_node == 0:
            ptwbus = len(twindows_bus)*peak_to_off/(len(twindows_bus)*peak_to_off+(len(twindows)-len(twindows_bus)))
            # Returns demand per node and time window.
            nums = [random.randint(1,4) for area in areas] ### Why 4?
            pdem = [i/sum(nums) for i in nums]
            p_dem_per_area = dict(zip(areas, pdem))
            #TotDemPerTW = {tw:int (random.uniform (DemLB,DemUB)*len(dem_nodes)) for tw in twindows}
            daily_p_dem_per_area = {area: p_dem_per_area[area]*tot_dem for area in areas}
            peak_dem_bus_area = {area:daily_p_dem_per_area[area]*ptwbus for area in AreasCom}
            peak_dem_res_area = {area:daily_p_dem_per_area[area]*(1-ptwbus) for area in areas_res}
            peak_dem_area = {**peak_dem_bus_area, **peak_dem_res_area}   
            off_dem_area = {area: daily_p_dem_per_area[area] - peak_dem_area[area] for area in areas}
            
            dem_tw_bus_area = {}
            dem_tw_nonbus_area = {}
            for area in areas:
                totbusdem = peak_dem_area[area]
                dem_tw_bus_area[area] = utilities.getSum (twindows_bus, totbusdem) # For business time windows
                totnonbusdem = off_dem_area[area]
                dem_tw_nonbus_area[area] = utilities.getSum (tw_no_bus, totnonbusdem) #
            dem_tw_area = {}
            for area in areas:
                dem_tw_area[area] = {}
                for tw in twindows:
                    if tw in twindows_bus:
                        dem_tw_area[area][tw] = dem_tw_bus_area[area][tw]
                    else:
                        dem_tw_area[area][tw] = dem_tw_nonbus_area[area][tw]
            demand = {}
            for area in areas:
                dtemp = {}
                area_nodes = nodes_area[area]
                for tw in twindows:
                    areadem = dem_tw_area[area][tw]
                    dtemp[tw] = utilities.getSum (area_nodes, areadem)
                for tw in dtemp:
                    for node in dtemp[tw]:
                        demand[node,tw] = dtemp[tw][node]
            dem_per_node = {node: sum(demand[node,tw] for tw in twindows) for node in dem_nodes}
            tw_w_demand = {node: 'Multiple' for node in dem_nodes}
        else: # If we only have one time window with demand for each node     
            tw_w_demand = {}
            nodes_bus = network.nodes_bus
            # Returns demand per node and time window.
            tw_no_bus = [tw for tw in twindows if tw not in twindows_bus]  
            tot_dem = round(demav*len(dem_nodes),0)
            dem_per_node  = utilities.F_N_SUM (dem_nodes,tot_dem)
            demand = {}
            for node in dem_nodes:
                if node in nodes_bus:
                    tw = random.choice(twindows_bus)
                else:
                    tw = random.choice(tw_no_bus)
                demand[node,tw] = dem_per_node[node]
                tw_w_demand[node] = tw
                tw_with_0_demand = copy.deepcopy(twindows)
                tw_with_0_demand.remove((tw))
                for tw2 in tw_with_0_demand:
                    demand[node,tw2] = 0
        return (demand, dem_per_node,tw_w_demand)

    def getTimeR(self):
        bus_speed = self.bus_speed # km/h
        routes = self.network.routes
        length_route = self.network.length_route # meters
        time_route = {r:length_route[r] / 1000 / bus_speed for r in routes} # Hours
        return (time_route)
        
    def getFreqQ (self):
        routes = self.network.routes
        twindows = self.twindows
        twindows_peak = self.twindows_peak
        b_freq_peak_route = self.b_freq_peak_route
        b_freq_off_route = self.b_freq_off_route
        tw_len = self.tw_len
        b_freq_tw = {}
        b_freq_any_peak_tw = {}
        b_freq_any_off_tw = {}
        for i in routes:
            for t in twindows:
                if t in twindows_peak:
                    b_freq_tw[i,t] = b_freq_peak_route[i] * tw_len
                    # fpeak in buses per hour, tw_len in hours/time window
                else:
                    b_freq_tw[i,t] = b_freq_off_route[i] * tw_len
        b_freq_any_peak_tw = {r : b_freq_peak_route[r] * tw_len for r in routes}
        b_freq_any_off_tw = {r : b_freq_off_route[r] * tw_len for r in routes}
        return (b_freq_tw, b_freq_any_peak_tw, b_freq_any_off_tw)
        
    def getCostRr (self):
        routes = self.network.routes
        length_route = self.network.length_route
        cost_km_bus = self.cost_km_bus
        vehicle_route = self.vehicle_route
        cost_route_round = {}
        for i in routes:
            cost_route_round[i] = 2 / 1000 * length_route[i] * cost_km_bus[vehicle_route[i]]
        return (cost_route_round)    
        
    def getCap (self):
        network = self.network
        routes = network.routes
        twindows = self.twindows
        vehicle_route = self.vehicle_route
        cap_bus = self.cap_bus
        b_freq_tw = self.b_freq_tw
        cap_basic = {}     
        for i in routes:
            for t in twindows:
                cap_basic[i, t] = b_freq_tw[i,t] *cap_bus[vehicle_route[i]]
        tot_cap_basic = sum(cap_basic.values())
        tot_cap_basic_tw = {tw: sum(cap_basic[i,tw] for i in routes) for tw in twindows}
        return (cap_basic, tot_cap_basic_tw, tot_cap_basic)
    
    def getDemBusTW(self): # Do I use it anywhere?
        nodes_bus_range = self.nodes_bus_range
        demand = self.demand
        twindows = self.twindows
        dem_nodes_bus = {i:demand[i] for i in demand if i[0] in nodes_bus_range}
        tot_dem_nodes_bus_tw = {tw: sum(dem_nodes_bus[i,tw] for i in nodes_bus_range) for tw in twindows}
        return (tot_dem_nodes_bus_tw)
    
    def getRouteType (self):
        # Creates a dictionary: {Bus: routes_type}
        bus_types = self.bus_types
        vehicle_route = self.vehicle_route
        routes = self.network.routes
        routes_type = {t: [] for t in bus_types}
        for route in routes:
            bustype = vehicle_route[route]
            routes_type[bustype].append(route)
        return (routes_type)

    
    def getRandomStuff (self):
        ### Contains all parameters that are created randomly.
        random.seed(self.seed)
        routes = self.network.routes
        stops_per_route = self.network.stops_per_route
        route_costs = self.route_costs
        stop_costs = self.stop_costs
        bus_types = self.bus_types
        f_peak_range = self.f_peak_range
        f_off_range = self.f_off_range
        
        c_route = {r: random.choice(route_costs) for r in routes}
        vehicle_route = {r: random.choice(bus_types) for r in routes}
        b_freq_peak_route = {r: random.choice(f_peak_range) for r in routes}
        b_freq_off_route = {r: random.choice(f_off_range) for r in routes}
        c_stop = {}
        for r in routes:
            for s in stops_per_route[r]:
                c_stop[r,s] = random.choice(stop_costs)
        return (c_route, c_stop, vehicle_route,b_freq_peak_route,b_freq_off_route)

    def get_freq (self):
        routes = self.network.routes
        b_freq_any_peak_tw = self.b_freq_any_peak_tw
        b_freq_any_off_tw = self.b_freq_any_off_tw
        df_peak_tw = self.df_peak_tw
        df_off_tw = self.df_off_tw
        twindows_off = self.twindows_off
        twindows_peak = self.twindows_peak
        tw_len = self.tw_len
        freq_tw = {}
        freq_peak_h = {}
        freq_off_h = {}
        for i in routes:
            for tw in twindows_peak:
                freq_tw[i,tw] = b_freq_any_peak_tw[i] + df_peak_tw[i]
            for tw in twindows_off:
                freq_tw[i,tw] = b_freq_any_off_tw[i] + df_off_tw[i]
        for i in routes:
            freq_peak_h[i] = (b_freq_any_peak_tw[i] + df_peak_tw[i]) / tw_len
            freq_off_h[i] = (b_freq_any_off_tw[i] + df_off_tw[i]) / tw_len
        return (freq_tw, freq_peak_h, freq_off_h )
        
    def getVmtBus (self, f_meters_to_miles, freq_tw):
        routes = self.network.routes
        twindows = self.twindows
        length_route = self.network.length_route
        vmt_route = {}
        for route in routes:
            len_total = 2 * length_route[route]
            passes = 0
            for tw in twindows:
                passes = passes + freq_tw[route,tw]
            vmt_route[route] = passes * len_total * f_meters_to_miles
        vmt_bus_total = sum(vmt_route.values())
        return (vmt_route, vmt_bus_total)
        

class Network:
    def __init__ (self,nodes,dem_nodes,nodes_res, nodes_bus, routes, stops, dx, dy,
                  stops_per_route, areas_res, AreasCom, areas, area_per_node,nodes_area,
                  depot, p_res):
        """ Initializes an object of type Network, which represents the area of interest
        in terms of nodes, links, bus routes, stops, and land use. """
        self.nodes = nodes
        self.dem_nodes = dem_nodes
        self.nodes_res = nodes_res
        self.nodes_bus = nodes_bus
        self.routes = routes
        self.stops = stops
        self.dx = dx
        self.dy = dy
        self.stops_per_route = stops_per_route
        self.areas_res = areas_res
        self.AreasCom = AreasCom
        self.areas = areas
        self.area_per_node = area_per_node
        self.nodes_area = nodes_area
        self.depot = depot
        self.name = 'Unnamed_Network'
        self.dist = self.getDist()
        self.p_res = p_res
        (self.links_route, self.prec_stops_route_link, self.len_route_link,
             self.length_route) = self.getLinks ()
        
    def __repr__(self):
        return "<Case %s>" % (self.name)    
        
    def getDist (self):
        # Calculates Manhattan distance.
        nodes = self.nodes
        dx = self.dx
        dy = self.dy
        dist = {}
        for node1 in nodes:
            for node2 in nodes:
                 dist[node1,node2] = abs(dx[node1]-dx[node2]) + abs (dy[node1] - dy[node2])
        return (dist)       
        
    def getLinks (self):
        # Crates the network links for each route.
        routes = self.routes
        stops_per_route = self.stops_per_route
        dist = self.dist       
        links_route = {route:[] for route in routes}
        prec_stops_route_link = {route:{} for route in routes} # Including first stop of link
        for route in routes:
            stops = stops_per_route[route]
            for (ind,i) in enumerate(stops):
                if ind == len(stops)-1:
                    break
                stop1 = stops[ind]
                stop2 = stops[ind+1]
                newlink = (stop1, stop2)
                links_route[route].append(newlink)
                prec_stops_route_link[route][newlink] = stops[0:stops.index(stop2)]
        len_route_link = {}    
        length_route = {}
        for route in routes:
            length = 0
            for link in links_route[route]:
                newl = dist[link[0], link[1]]
                len_route_link[route,link] = newl
                length = length + newl
            length_route[route] = length
        return (links_route, prec_stops_route_link, len_route_link, length_route)




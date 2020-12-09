"""
Created on Sun May 28 09:17:22 2017

Contains the VRPTW example code from Google's OR tools
(https://developers.google.com/optimization/routing/vrp)

We modify the INPUT of the model to account for some default assumptions of
OR tools (e.g., the demand of each node is concentrated in one time window
during the day, and the time windows are strict).

We modify the code of OR-Tools to allow for truck reloading at the depot.

"""

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def distance(x1, y1, x2, y2):
    # Manhattan distance
    dist = abs(x1 - x2) + abs(y1 - y2)
    return dist

    
class CreateDistanceCallback(object):
  """Create callback to calculate distances and travel times between points."""

  def __init__(self, locations):
    """Initialize distance array."""
    num_locations = len(locations)
    self.matrix = {}

    for from_node in range(num_locations):
      self.matrix[from_node] = {}
      for to_node in range(num_locations):
        x1 = locations[from_node][0]
        y1 = locations[from_node][1]
        x2 = locations[to_node][0]
        y2 = locations[to_node][1]
        self.matrix[from_node][to_node] = distance(x1, y1, x2, y2)

  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]


# Demand callback
class CreateDemandCallback(object):
  """Create callback to get demands at location node."""

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return -self.matrix[from_node]

# Service time (proportional to demand) callback.
class CreateServiceTimeCallback(object):
  """Create callback to get time windows at each location."""

  def __init__(self, demands, time_per_demand_unit):
    self.matrix = demands
    self.time_per_demand_unit = time_per_demand_unit

  def ServiceTime(self, from_node, to_node):
    return self.matrix[from_node] * self.time_per_demand_unit
# Create the travel time callback (equals distance divided by speed).
class CreateTravelTimeCallback(object):
  """Create callback to get travel times between locations."""

  def __init__(self, dist_callback, speed):
    self.dist_callback = dist_callback
    self.speed = speed

  def TravelTime(self, from_node, to_node):
    travel_time = self.dist_callback(from_node, to_node) / self.speed
    return travel_time
    
    
class CreateTotalTimeCallback(object):
  """Create callback to get total times between locations.
   Create total_time callback (equals service time plus travel time).
  """

  def __init__(self, service_time_callback, travel_time_callback):
    self.service_time_callback = service_time_callback
    self.travel_time_callback = travel_time_callback

  def TotalTime(self, from_node, to_node):
    service_time = self.service_time_callback(from_node, to_node)
    travel_time = self.travel_time_callback(from_node, to_node)
    return service_time + travel_time
 
    
def main(data, num_vehicles, time_per_demand_unit, horizon, speed, VehicleCapacity,
         first_reload_node_index):

  res = [] 
  # Create the data.
  locations = data[0]
  demands = data[1]
  start_times = data[2]
  end_times = data[3]
  num_locations = len(locations)
  depot = 0
  
  # Create routing model.
  if num_locations > 0:
    # The number of nodes of the VRP is num_locations.
    # Nodes are indexed from 0 to num_locations - 1. By default the start of
    # a route is node 0.
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    # Setting first solution heuristic: the
    # method for finding a first solution to the problem.
    
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # The 'PATH_CHEAPEST_ARC' method does the following:
    # Starting from a route "start" node, connect it to the node which produces the
    # cheapest route segment, then extend the route by iterating on the last
    # node added to the route.
    # Put callbacks to the distance function and travel time functions here.
    dist_between_locations = CreateDistanceCallback(locations)
    dist_callback = dist_between_locations.Distance
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    demands_at_locations = CreateDemandCallback(demands)
    demands_callback = demands_at_locations.Demand

    fix_start_cumul_to_zero = True
    capacity = "Capacity"
    
    routing.AddDimension(demands_callback, VehicleCapacity, VehicleCapacity,  
                         fix_start_cumul_to_zero, capacity)
    
    capacity_dimension = routing.GetDimensionOrDie(capacity)  
    for order in range(1,num_locations):
        if  order < first_reload_node_index:
            capacity_dimension.SlackVar(order).SetValue(0)
        routing.AddVariableMaximizedByFinalizer(capacity_dimension.CumulVar(order))

    time = "Time"

    service_times = CreateServiceTimeCallback(demands, time_per_demand_unit)
    service_time_callback = service_times.ServiceTime

    travel_times = CreateTravelTimeCallback(dist_callback, speed)
    travel_time_callback = travel_times.TravelTime

    total_times = CreateTotalTimeCallback(service_time_callback, travel_time_callback)
    total_time_callback = total_times.TotalTime

    routing.AddDimension(total_time_callback,  # total time function callback
                         horizon,
                         horizon,
                         fix_start_cumul_to_zero,
                         time)

    time_dimension = routing.GetDimensionOrDie(time)

    for location in range(2, first_reload_node_index): ### Without depot and re-loading
      start = start_times[location]
      end = end_times[location]
      time_dimension.CumulVar(location).SetRange(start, end)
      
    for location in range (first_reload_node_index, num_locations):
        time_dimension.CumulVar(location).SetRange(0, horizon)

    for location in range (first_reload_node_index, num_locations):
        routing.AddDisjunction([location], 0)
    
    penalty = 10000000
    for order in range(1, first_reload_node_index):  ### Without depot and re-loading
        routing.AddDisjunction([order], penalty)
      
    # Solve displays a solution, if any.
    print ('Now doing the assignment for the truck -- this might take time...')
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
      # Solution cost.
      print ("Total distance of all routes: " + str(assignment.ObjectiveValue()) + "\n")
      # Inspect solution.
      capacity_dimension = routing.GetDimensionOrDie(capacity);
      time_dimension = routing.GetDimensionOrDie(time);

      for vehicle_nbr in range(num_vehicles):
        index = routing.Start(vehicle_nbr)
        plan_output = 'Route {0}:'.format(vehicle_nbr)

        while not routing.IsEnd(index):
          node_index = routing.IndexToNode(index)
          load_var = capacity_dimension.CumulVar(index)
          time_var = time_dimension.CumulVar(index)
          plan_output += \
                    " {node_index} Load({load}) Time({tmin}, {tmax}) -> ".format(
                        node_index=node_index,
                        load=assignment.Value(load_var),
                        tmin=str(assignment.Min(time_var)),
                        tmax=str(assignment.Max(time_var)))
          index = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)
        load_var = capacity_dimension.CumulVar(index)
        time_var = time_dimension.CumulVar(index)
        plan_output += \
                  " {node_index} Load({load}) Time({tmin}, {tmax})".format(
                      node_index=node_index,
                      load=assignment.Value(load_var),
                      tmin=str(assignment.Min(time_var)),
                      tmax=str(assignment.Max(time_var)))
        res.append(plan_output)
    else:
      print ('No solution found.')
  else:
    print ('Specify an instance greater than 0.')

  return(res) 


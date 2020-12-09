# last_mile_delivery_synergies
`last_mile_delivery_synergies` is an implementation of last-mile delivery synergies between passenger and freight transport for urban areas using a MILP model and Google OR-Tools.

This tool implements a synergy scheme between passenger and freight transportation for last-mile deliveries in urban areas. The operating scheme relies on the selection of existing bus routes, where parcels are loaded, and bus stops, where parcels are unloaded, while demand with limited access to the bus system is served by a truck. To replicate the operating scheme, we develop a Mixed Integer Linear Programming model demonstrating the use of buses, and combine it with an algorithm for Vehicle Routing with Time Windows to simulate the operation of trucks. The model designs a minimum cost scheme by taking into account operator, passenger, and external costs.
The MILP Model is implemented using Gurobi solver (https://www.gurobi.com/), while the vehicle routing problem is solved using Google OR-Tools (https://developers.google.com/optimization/routing/vrptw).

Supported functionalities:

- A mixed integer-linear programming model that determines the allocation of demand nodes to bus routes and stops based on the minimization of total cost, including environmental externalities.
- A modified version of Google OR-Tools Vehicle Routing Problem with Time Windows. The modifications enable the consideration of soft time windows, multiple visits to the depot, and for nodes with multiple demand requests corresponding to different time windows within a day.
- Calculation and plot generation of fuel emissions for different values of average demand per node.

License:

If you use `last_mile_delivery_synergies` in your research, please cite our paper:

```
    @inproceedings{pternea2018feasibility,
       title={A feasibility study for last-mile synergies between passenger and freight transport for an urban area},
       author={Pternea, Moschoula and Lan, Chien-Lun and Haghani, Ali and Chin, Shih Miao},  
       booktitle={Proceedings of the Annual Meeting of Transportation Research Board},
       pages={1--5},  
       year={2018} 
   }
```

Our paper can be found here: https://www.researchgate.net/profile/Chien_Lun_Lan/publication/335796051_A_Feasibility_Study_for_Last-Mile_Synergies_between_Passenger_and_Freight_Transport_for_an_Urban_Area/links/5e7ace58299bf1f3873fc612/A-Feasibility-Study-for-Last-Mile-Synergies-between-Passenger-and-Freight-Transport-for-an-Urban-Area.pdf

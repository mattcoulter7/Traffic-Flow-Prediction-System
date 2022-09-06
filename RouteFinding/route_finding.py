from asyncio.windows_events import NULL
from distutils.log import debug
import string
import csv
from enum import Enum
from operator import attrgetter
from typing import List
from typing_extensions import Self

# enum for each type of scats site
class SiteType(Enum):
    INT = 0
    POS = 1
    FLASH_PX = 2
    FIRE_W_W = 3
    RBT_MTR = 4
    FIRE_SIG = 5
    AMBU_SIG = 6
    RAMP_MTR = 7
    BUS_SIG = 8
    TMP_POS = 9
    O_H_LANE = 10

# the node represents each scats site
class Node:
    scats_number: int
    neighbours: list
    longitude: float
    latitude: float
    name: str
    scats_type: SiteType 

# the traffic graph 
class TrafficGraph:
    nodes: list

    def __init__(self) -> None:
        self.nodes = list()

    def get_node_from_scats_number(self, scats_number: int) -> Node:
        n = [x for x in self.nodes if x.scats_number == scats_number]
        if len(n) == 0:
            return NULL

        return n[0]


# the route
class Route:
    nodes: list

    def __init__(self) -> None:
        self.nodes = list()

    def print_route(self):
        for node in self.nodes:
            print(node.name)

# route Node
class RouteNode:
    node: Node
    previous_node: Self

    def __init__(self, node: Self, previous_node: Self) -> None:
        self.node = node
        self.previous_node = previous_node

    def convert_to_route(self) -> Route:
        route = Route()
        cur_node: Self = self
        while cur_node != NULL:
            print (cur_node.node.scats_number)
            route.nodes.append(cur_node.node)
            cur_node = cur_node.previous_node
        route.nodes.reverse()
        print(len(route.nodes))
        return route
    
    def expand_node(self, traffic_network: TrafficGraph) -> list:
        nodes = list()
        for n in self.node.neighbours:
            nodes.append(RouteNode(traffic_network.get_node_from_scats_number(n), self))
        return nodes

    def calcuate_node_cost(self) -> float:
        if self.previous_node == NULL:
            return 0.0
        
        # caclucate the cost to travel to the previous node
        cost = 1 # TODO calculate the actual cost of the paths it prediced time
        
        # add the cost to the current cost of the path
        cost += self.previous_node.calcuate_node_cost()

        return cost

# open the route file 
def open_road_network(file: string) -> TrafficGraph:
    tg = TrafficGraph()
    fieldnames = ['SCATS Number', 'Site Description', 'Site Type', 'Longitude', 'Latitude', 'Neighbours']
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if len(row['Site Description']) == 0: 
                continue

            node = Node()
            node.scats_number = int(row['SCATS Number'])
            node.name = row['Site Description']
            node.latitude = float(row['Latitude'])
            node.longitude = float(row['Longitude'])
            node.scats_type = SiteType[row['Site Type']]
            node.neighbours = [int(x) for x in row['Neighbours'].split(';')]
            tg.nodes.append(node)
            print(node.scats_number, node.name, node.latitude, node.longitude, node.scats_type, node.neighbours)

    return tg

# a-star algorithm 
def find_routes(traffic_network: TrafficGraph, origin: int, destination: int, route_options_count: int = 5) -> list:
    routes = list()
    destination_node = traffic_network.get_node_from_scats_number(destination)
    frontier = list()
    # add origin to frontier
    frontier.append(RouteNode(traffic_network.get_node_from_scats_number(origin), NULL))
    while len(frontier) > 0:
        # sort the frontier by the path cost
        frontier.sort(key=lambda x: x.calcuate_node_cost())

        # selected is the fist node in the list as it has the lowest cost
        selected: RouteNode = frontier[len(routes)]

        # is the frontier at the destination?
        if selected.node == destination_node:
            print ("route found")
            routes.append(selected.convert_to_route())
            # exit search when the desired number of routes are found 
            if len(routes) == route_options_count:
                break
        
        # expand the selected node
        children: list = selected.expand_node(traffic_network)
        
        # remove the selected node from the frontier
        frontier.remove(selected)

        # TODO see if it needs to check for duplicate nodes before expanding
        frontier.extend(children)
    
    return routes

# display the routes 

if __name__ == "__main__":
    traffic_network = open_road_network("traffic_network.csv")
    routes = find_routes(traffic_network, 970, 2825, route_options_count=3)
    for r in routes:
        print ("--ROUTE--")
        r.print_route()
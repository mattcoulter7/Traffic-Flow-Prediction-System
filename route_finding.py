from cmath import sqrt
from distutils.log import debug
import string
import csv
import datetime
import numpy
import random
from sys import float_repr_style
from turtle import distance
import geopy.distance
from TrafficData.TrafficData import predict_traffic_flow
from enum import Enum
from operator import attrgetter
from typing import List
from typing_extensions import Self
import argparse
from renderer import renderMap

MAX_SPEED = 60 # estimate to be the speed limit for all roads
CAPACITY_SPEED = 48
FREE_FLOW_SPEED = 48 * 2
MAX_FLOW_RATE = 1800
JAM_DENSITY = 150
ITERSECTION_WAIT_TIME = 30 / 60 / 60 # approximate an average wait time of 30 seconds for each intersection this is converted to hours
TRAFFIC_NETWORK_FILE = "data/traffic_network.csv"

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
            return None

        return n[0]


# the route
class Route:
    nodes: list
    cost: float

    def __init__(self, cost) -> None:
        self.nodes = list()
        self.cost = cost

    def print_route(self):
        for node in self.nodes:
            print(node.scats_number ,"-" ,node.name)
        print ("Length:\t\t", len(self.nodes))
        print ("Distance:\t", "{:.2f}".format(self.calculate_route_distance()) + "km")
        # convert cost from seconds to minutes
        print ("Cost:\t\t", "{:.2f}".format(self.cost * 60) + "mins")

    def calculate_route_distance(self) -> float:
        dist = 0.0
        for i in range(len(self.nodes) - 1):
            coords_1 = (self.nodes[i].latitude, self.nodes[i].longitude)
            coords_2 = (self.nodes[i + 1].latitude, self.nodes[i + 1].longitude)

            dist += geopy.distance.geodesic(coords_1, coords_2).km
        return dist

    def list_scats(self):
        arr = list()
        for node in self.nodes:
            arr.append(node.scats_number)
        return arr

# route Node wraps up the node class and implements functionality to allow for routing for A* graph search
class RouteNode:
    node: Node
    previous_node: Self
    cost: float
    date: datetime

    def __init__(self, node: Self, previous_node: Self, date: datetime) -> None:
        self.node = node
        self.previous_node = previous_node
        self.date = date
        self.cost = self.calcuate_node_cost(date)
        #print(self.node.name, self.cost)

    def convert_to_route(self) -> Route:
        route = Route(self.cost)
        cur_node: Self = self
        while cur_node != None:
            #print (cur_node.node.scats_number)
            route.nodes.append(cur_node.node)
            cur_node = cur_node.previous_node
        route.nodes.reverse()
        #print(len(route.nodes))
        #print(self.cost, "km")
        return route
    
    def expand_node(self, traffic_network: TrafficGraph) -> list:
        nodes = list()
        for n in self.node.neighbours:
            nodes.append(RouteNode(traffic_network.get_node_from_scats_number(n), self, self.date))
        return nodes

    def calcuate_node_cost(self, date: datetime) -> float:
        if self.previous_node == None:
            return 0.0
        
        # caclucate the cost to travel to the previous node
        coords_1 = (self.node.latitude, self.node.longitude)
        coords_2 = (self.previous_node.node.latitude, self.previous_node.node.longitude)

        dist = geopy.distance.geodesic(coords_1, coords_2).km

        # check the locations are correct
        if dist == 0:
            raise RuntimeError("ERROR in data:", self.node.scats_number, ",", self.node.name)
            
        # calculate the speed of the segment
        # flow the number of vehicles passing over a point over a period of time
        # add the cost to the predition so the traffic times are slightly more accurate
        new_date_time = date + datetime.timedelta(hours=self.previous_node.cost)
        flow = predict_traffic_flow(self.previous_node.node.scats_number, new_date_time)
        #random.seed(self.previous_node.node.scats_number + time.minute)
        #flow = random.randint(0, 1800)

        # clamp the flow value 1800
        flow = numpy.clip(flow, 0, 1800)

        traffic_speed = 48 - ((8 * sqrt(1800 - flow)) / (5 * sqrt(2)))
        traffic_speed2 = 48 + ((8 * sqrt(1800 - flow)) / ( 5 * sqrt(2)))
        #print("flow:", flow, "speed 1:", traffic_speed, "speed 2:", traffic_speed2)

        # select the min speed as traffic can't breach the speed limit
        speed = min(MAX_SPEED, traffic_speed2.real)
        
        # calculate segment time in seconds
        segment_time = dist / speed

        # add the cost to the current cost of the path
        cost = self.previous_node.cost + segment_time + ITERSECTION_WAIT_TIME if self.node.scats_type == SiteType.INT else 0

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
            #print(node.scats_number, node.name, node.latitude, node.longitude, node.scats_type, node.neighbours)

    return tg

# a-star algorithm 
def find_routes(traffic_network: TrafficGraph, origin: int, destination: int, date: datetime, route_options_count: int = 5) -> list:
    routes = list()
    destination_node = traffic_network.get_node_from_scats_number(destination)
    frontier = list()
    # add origin to frontier
    frontier.append(RouteNode(traffic_network.get_node_from_scats_number(origin), None, date))
    while len(frontier) > 0:
        # sort the frontier by the path cost
        frontier.sort(key=lambda x: x.cost)

        # selected is the fist node in the list as it has the lowest cost
        selected: RouteNode = frontier[0]

        # is the frontier at the destination?
        if selected.node == destination_node:
            #print ("route found")
            routes.append(selected.convert_to_route())
            
            # remove destination node from list
            frontier.remove(selected)
            
            # exit search when the desired number of routes are found 
            if len(routes) == route_options_count:
                break

            continue
        
        # expand the selected node
        children: list = selected.expand_node(traffic_network)
        
        # remove the selected node from the frontier
        frontier.remove(selected)

        # remove nodes with loops in it
        for c in children:
            new_node: Node = c.node
            previous_node: RouteNode = c.previous_node
            duplicated = False
            while previous_node != None:
                if new_node == previous_node.node:
                    # duplicated node
                    #print("duplicated")
                    duplicated = True
                    break
                # set the previous node to the previous previous node
                previous_node = previous_node.previous_node
            
            if duplicated:
                children.remove(c)

        # TODO see if it needs to check for duplicate nodes before expanding
        frontier.extend(children)
    
    return routes

# display the routes 


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=970,
        help="Starting SCATS")
    parser.add_argument(
        "--dest",
        default=3001,
        help="Finish SCATS")
    parser.add_argument(
        "--time",
        default=datetime.datetime.now(),
        help="Time of Day")
    parser.add_argument(
        "--day",
        default="6",
        help="Day Index")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    routeList = []
    args = createParser()
    traffic_network = open_road_network(TRAFFIC_NETWORK_FILE)
    routes = find_routes(traffic_network, int(args.src), int(args.dest), args.time, route_options_count=5)
    for i, r in enumerate(routes):
        print (f"--ROUTE {i + 1}--")
        r.print_route()
        routeList.append(r.list_scats())
    renderMap(routeList)
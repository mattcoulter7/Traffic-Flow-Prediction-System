import string
import csv
from enum import Enum
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
    nodes: list = list()


# open the route file 
def open_road_network(file: string) -> TrafficGraph:
    tg = TrafficGraph()
    fieldnames = ['SCATS Number', 'Site Description', 'Site Type', 'Longitude', 'Latitude', 'Neighbours']
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            node = Node()
            node.scats_number = int(row['SCATS Number'])
            node.name = row['Site Description']
            node.latitude = float(row['Latitude'])
            node.longitude = float(row['Longitude'])
            node.scats_type = SiteType[row['Site Type']]
            node.neighbours = [int(x) for x in row['Neighbours'].split('|')]
            tg.nodes.append(node)
            print(node.scats_number, node.name, node.latitude, node.longitude, node.scats_type, node.neighbours)

    return tg

# a-star algorithm 

# display the routes 

if __name__ == "__main__":
    open_road_network("traffic_network.csv")
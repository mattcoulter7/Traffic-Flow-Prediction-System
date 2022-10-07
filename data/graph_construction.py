import csv
from multiprocessing import connection
import os
import math
import geopy.distance
from numpy import positive

from pygorithm.geometry.vector2 import Vector2

# means that each intersection will group connecting intersections by their relative angle between ranges
# the angle range is 360 / angle_segments
angle_segments = 4
segment_angle = 360 / angle_segments

def get_relative_angles(angles,parent_angle,angle_offset = 0):
    relative_angles = []
    for angle in angles:
        relative_angle = angle - parent_angle + angle_offset
        relative_angle = relative_angle % 360
        relative_angles.append(relative_angle)

    return relative_angles

class Location():
    def __init__(self,csv_line):
        self.scats_number = csv_line[0]
        self.site_description = csv_line[1]
        self.site_type = csv_line[2]
        self.latitude = float(csv_line[3])
        self.longitude = float(csv_line[4])
        self.geo = Vector2(self.longitude,self.latitude)
        self.neighbours = csv_line[5]
        
        self.roads = self.get_roads()

    def get_roads(self):
        return list(filter(lambda a: a != "",self.site_description.split('/')))

    def calculate_connecting_sites(self,locations):
        self.connecting_sites = []
        for road in self.roads:
            # 1. find all the sites which connect to the road
            connecting_locations = list(filter(lambda loc: loc is not self and road in loc.roads,locations))
            if len(connecting_locations) == 0: continue

            # 2. determine the angle of the roads
            angles = list(map(lambda loc: self.angle_to_site(loc),connecting_locations))
            distances = list(map(lambda loc: self.distance_to_site(loc),connecting_locations))
            shortest_distance = min(distances)
            closest_site_index = distances.index(shortest_distance)
            road_angle = angles[closest_site_index]


            # 3. separate each road collection by angle segments
            relative_angles = get_relative_angles(angles,road_angle,angle_offset=-segment_angle/2)
            site_angle_debug = list(zip(list(map(lambda s: s.scats_number,connecting_locations)),relative_angles))
            print(site_angle_debug)
            for i in range(angle_segments):
                open_angle = segment_angle * i
                close_angle = segment_angle * (i + 1)

                seg,seg_angles,seg_dist = [],[],[]
                for i in range(len(relative_angles)):
                    relative_angle = relative_angles[i]
                    if relative_angle >= open_angle and relative_angle <= close_angle:
                        seg.append(connecting_locations[i])
                        seg_angles.append(relative_angle)
                        seg_dist.append(distances[i])

                # 4. Determine the closest site in the segment
                segment_closest_site = seg[min(range(len(seg_dist)), key=seg_dist.__getitem__)] if len(seg_dist) > 0 else None
                if segment_closest_site is not None: self.connecting_sites.append(segment_closest_site)
        return self.connecting_sites

    def distance_to_site(self,site):
        return geopy.distance.geodesic((self.latitude,self.longitude), (site.latitude,site.longitude)).km

    def angle_to_site(self,site):
        geo_delta = site.geo - self.geo
        degrees = math.atan2(geo_delta.y, geo_delta.x)*180/math.pi # angle with respect to x
        return degrees


def read_all_locations():
    reader = csv.reader(open(os.path.join(os.path.dirname(__file__),'traffic_network.csv'), 'r'))
    locations = [loc for loc in reader][1:]
    locations = [Location(loc) for loc in locations]
    return locations

def write_all_locations(locations):
    writer = csv.writer(open(os.path.join(os.path.dirname(__file__),'traffic_network2.csv'), 'w', newline=''))
    writer.writerow(['SCATS Number','Site Description','Site Type','Latitude','Longitude','Neighbours'])
    for loc in locations:
        connecting_sites = ";".join(map(lambda s: s.scats_number,loc.connecting_sites))
        writer.writerow([loc.scats_number,loc.site_description,loc.site_type,loc.latitude,loc.longitude,connecting_sites])

def main():
    locations = read_all_locations()
    for loc in locations:
        loc.calculate_connecting_sites(locations)
    write_all_locations(locations)

if __name__ == "__main__":
    main()
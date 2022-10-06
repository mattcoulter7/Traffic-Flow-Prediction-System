import argparse
from sys import prefix
from turtle import color
import folium
import random
import json
import csv



style = lambda x: {
    'color' : x['properties']['stroke'],
    'weight' : x['properties']['stroke-width']
}


data = [[970,3685,2000,3682,3126,2200,4063,4034,4032,4321],[970,3685,2000,3682,3126,2200,4063,4057,4032,4321],[970,3685,2000,3682,3126,3127,4063,4034,4032,4321],[970,3685,2000,3682,3126,3127,4063,4057,4032,4321],[970,3685,2000,4043,4040,3120,4035,4034,4032,4321]]

def getCoords(scat):
    file = open('network.csv', 'r')

    for row in csv.reader(file):
        if row[0] == str(scat):
            return float(row[4]) + 0.0012469, float(row[3]) + 0.0012275

    print("unable to find SCAT location")
    return 0,0
    
def getNodesGeoJson():
    return True


def generateGeoJson(arr):
    data = {}
    data['type'] = 'FeatureCollection'
    data['features'] = []

    for iter, route in reversed(list(enumerate(arr))):
        weight = 5#4 if iter == 0 else 4
        color = "#3484F0" if iter == 0 else "#757575"
        coords = []
        for scat in route:
            lon, lat = getCoords(scat)
            coords.append([lon, lat])
        
        feature = {}
        feature['type'] = 'Feature'

        properties = {}
        properties['stroke'] = color
        properties['stroke-width'] = weight
        feature['properties'] = properties

        geometry = {}
        geometry['type'] = 'LineString'
        geometry['coordinates'] = coords
        feature['geometry'] = geometry

        data["features"].append(feature)

    return json.dumps(data)

def drawMarkers(map, src, dest):
    src_lon, src_lat = getCoords(src)
    dest_lon, dest_lat = getCoords(dest)

    folium.Marker([src_lat, src_lon], popup=f"<strong>Start</strong> SCATS: {src}", icon=folium.Icon(color='blue', icon='circle', prefix='fa')).add_to(map)
    folium.Marker([dest_lat, dest_lon], popup=f"<strong>Finish</strong> SCATS: {dest}", icon=folium.Icon(color='red', icon='flag', prefix='fa')).add_to(map)

def drawNodes(map):
    file = open('network.csv', 'r')

    for iter, row in enumerate(csv.reader(file)):
        if iter == 0:
            continue
        lon, lat = getCoords(row[0])
        folium.Circle(
            radius=5,
            location=[lat, lon],
            popup=f"SCATS: {row[0]}",
            color="#5A5A5A",
            fill=False,
            ).add_to(map)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="0970",
        help="Starting SCATS")
    parser.add_argument(
        "--dest",
        default="3001",
        help="Finish SCATS")
    parser.add_argument(
        "--time",
        default="0.25",
        help="Time of Day")
    parser.add_argument(
        "--day",
        default="6",
        help="Day Index")
    args = parser.parse_args()

    # call search agorithm
    # return best 5 routes
    # generate geojson data
    routes = generateGeoJson(data)

    # create map
    map = folium.Map(location=[-37.831219, 145.056965], zoom_start=13, tiles="cartodbpositron", zoom_control=False,
               scrollWheelZoom=False,
               dragging=False)
    
    # plot data
    folium.GeoJson(routes, style_function=style).add_to(map)

    drawMarkers(map, 970, 4321)

    drawNodes(map)


    # save to file
    map.save("index.html")


main()

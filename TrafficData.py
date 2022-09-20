from random import Random

#TODO this is just a dummy filler function for now need to create proper implementation for final
# get traffic data

def get_traffic_volume(scats_number: int, time: float) -> float:
    random = Random()
    random.seed(scats_number)
    return random.uniform(1, 100)
from random import Random

#TODO this is just a dummy filler function for now need to create proper implementation for final
# get traffic data

# returns the vehicles traveled throught the intersection over the last hour
def get_traffic_flow(scats_number: int, time: float) -> float:
    random = Random()
    random.seed(scats_number + time)
    return random.uniform(1, 100)
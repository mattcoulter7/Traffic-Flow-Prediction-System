import csv
import os

def read_all_locations(file_name):
    reader = csv.reader(open(os.path.join(os.path.dirname(__file__),file_name), 'r'))
    locations = []
    for line in reader:
        locations.append([line[0],list(filter(lambda a: a != "",line[5].split(";")))])
    return locations[1:]

def write_errors(errors):
    writer = csv.writer(open(os.path.join(os.path.dirname(__file__),'errors.csv'), 'w', newline=''))
    for error in errors:
        writer.writerow([error])

def main():
    correct = read_all_locations('traffic_network.csv')
    calculated = read_all_locations('traffic_network2.csv')

    errors = []

    for correct,calculated in zip(correct,calculated):
        site = correct[0]
        for connection in correct[1]:
            if connection not in calculated[1]:
                errors.append(f'{site} did not find connection to {connection}')

        for connection in calculated[1]:
            if connection not in correct[1]:
                errors.append(f'{site} calculated connection {connection} but shouldnt have')

    write_errors(errors)

if __name__ == "__main__":
    main()
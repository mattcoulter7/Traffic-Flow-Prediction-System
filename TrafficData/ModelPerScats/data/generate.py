import csv
import os

# file where data is read from
src_path = "data/OriginalDataset.csv"
reader = csv.reader(open(src_path, 'r'))

# folder where data is saved to
dest_dir = "data/locations"

# locations with csv values
locations = {}

# validation data size needs to be a multiple of 96 to ensure the first line of test data is at 0:00
validation_data_size = 1440 

# csv header
header = ["5 Minutes","Lane 1 Flow (Veh/5 Minutes)","# Lane Points","% Observed","SCATS"]

# capture all of the data grouped by locations
times = []
for row in reader:
    if reader.line_num == 1:
        times = list(row[10:-3])
        print(times)
    else:
        date = row[9]
        location = row[0]
        num_col = len(row)
        for col_index in range(len(times)):
            values = []
            values.append(date + " " + times[col_index])
            values.append(row[col_index + 10])
            values.append(1)
            values.append(100)
            values.append(location)
            
            existing_values = locations.get(location,[header])
            existing_values.append(values)
            locations[location] = existing_values


# ensure output folder exists
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# save each location data to a separate csv file
for location, values in locations.items():
    splice_point = len(values) - validation_data_size

    # write the training data
    train_rows = values[:(splice_point)]
    train_output = os.path.join(dest_dir,f"{location}-train.csv")
    train_writer = csv.writer(open(train_output, 'w', newline=''))
    train_writer.writerows(train_rows)

    # write the validation data
    test_rows = values[splice_point:]
    test_rows.insert(0,train_rows[0]) # add header back in
    test_output = os.path.join(dest_dir,f"{location}-test.csv")
    test_writer = csv.writer(open(test_output, 'w', newline=''))
    test_writer.writerows(test_rows)

# save a locations file which has every location in it
csv.writer(open(os.path.join(dest_dir,"locations.txt"), 'w', newline='')).writerows(map(lambda a: [a],locations.keys()))
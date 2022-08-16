import csv
import datetime
import numpy as np

DATA_PATH = 'Scats Data October 2006.csv'

SCATS_NUMBER_INDEX = 0
GEO_X_INDEX = 3
GEO_Y_INDEX = 4
DATE_INDEX = 9
TIMES_INDEX_START = 10
TIMES_INDEX_END = 106

def get_date(date_string):
    [day, month, year] = date_string.split('/')
    return datetime.date(int(year), int(month), int(day))

def traffic_data():
    X_val = []
    y_val = []
    with open(DATA_PATH, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_index = -1
        for row in spamreader:
            row_index += 1
            if (row_index < 2):
                continue
            scats_number = row[SCATS_NUMBER_INDEX]
            geo_x = float(row[GEO_X_INDEX])
            geo_y = float(row[GEO_Y_INDEX])
            date = get_date(row[DATE_INDEX])
            day = date.weekday()
            densities = row[TIMES_INDEX_START:TIMES_INDEX_END]

            for i in range(len(densities)):
                density = int(densities[i])
                time = i * 15
                # X_val.append([scats_number,day,time])
                X_val.append([geo_x/100, geo_y/100, (day-3.5) / 3.5, (time - 720) / 720])
                y_val.append([(density - 347.5)/347.5])

    return np.array(X_val), np.array(y_val)

# [
#   [scats_number,day_index,time] || [geo_x,geo_y,day_index,time]
#   [density]
# ]


# generate the data
X, y = traffic_data()

# shuffle the data in preparation for extracting validation data
np.random.shuffle(X)
np.random.shuffle(y)

# extract the validation data
validation_data_size = 20000
X_test = X[0:validation_data_size]
y_test = y[0:validation_data_size]
X = X[validation_data_size:]
y = y[validation_data_size:]
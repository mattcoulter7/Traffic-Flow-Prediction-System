import csv

data = "data/TrainingDataAdapted.csv"

output = "data/TrainingDataAdaptedOutput.csv"

fields = ['Time', 'Flow', 'Points', 'Observed']


reader = csv.reader(open(data, 'r'))
writer = csv.writer(open(output, 'w', newline=''))



# row_count = sum(1 for row in reader)
times = []

for row in reader:
    if reader.line_num == 1:
        times = list(row)
        times.pop(0)
        print(times)
    else:
        values = []
        date = row[0]
        col_index = 0
        num_col = len(row)
        for col in row:
            col_index += 1
            if col_index >= num_col:
                break
            values.append(date + " " + times[col_index-1])
            values.append(row[col_index])

            writer.writerow(values)
            
            values = []
            # if col_index >= num_col-3:
            #     col_index = 0


#         for col in r:
#             print(col)

#    for row in reader:

#        if reader.line_num != 1:





           
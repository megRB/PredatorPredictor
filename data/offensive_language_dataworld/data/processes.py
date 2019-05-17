import csv

with open('labeled_data.csv', 'r') as csvSrc:
    csvDest = open('labeled_data_squashed.csv', 'w')
    reader = csv.reader(csvSrc)
    writer = csv.writer(csvDest)
    for row in reader:
        print(len(row))
        if(row[5].isdigit() == False):
        	continue
        print(row[5])
        if (int(row[5])<2):
        	row.append('1')
        elif (int(row[5]) == 2):
        	row.append('0')
        print(len(row))
        writer.writerow(row)
csvDest.close()

# import csv

# row = ['2', ' Marie', ' California']

# with open('labeled_data.csv', 'r') as readFile:
#     reader = csv.reader(readFile)
#     lines = list(reader)

# for i in range(len(lines)):
# 	continue
# 	print(lines[i][5])
# 	if (int(lines[i][5])<2):
# 		lines[i].append('1')
# 	elif (int(lines[i][5]) == 2):
# 		lines[i].append('0')

# with open('labeled_data_squashed.csv', 'w') as writeFile:
#     writer = csv.writer(writeFile)
#     writer.writerows(lines)

# readFile.close()
# writeFile.close()
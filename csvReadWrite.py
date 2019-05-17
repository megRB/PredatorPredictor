import preprocess

import csv
with open('./data/detecting_insults_kaggler/train.csv','r') as readFile:
	writeFile = open('./data/detecting_insults_kaggler/train_processed.csv', 'w+')
	writeFile = csv.writer(writeFile)
	data = csv.reader(readFile)
	print("Cleaning data now. Might take a while...")
	for row in data:
		print(row[2])
		# print(row[2])
		new_text = preprocess.cleanup(row[2])
		print(new_text)
		row[2] = new_text
		print(row)
		writeFile.writerow(row)
print("Cleaned file kaggle/train.csv")
print("Cleaned file in kaggle/train_processed.csv")

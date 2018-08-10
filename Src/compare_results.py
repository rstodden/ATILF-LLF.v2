import csv
with open("../Results/official_results.csv") as f:
	content = f.readlines()
	official = list()
	for line in content:
		split_line = line.split(";")
		official.append([])
		for element in split_line:
			official[-1].append(element.strip())
print(official)
#with open("../Results_epo_5/CNN/summary.txt") as f:
with open("../Results/MWEFiles/testSet/loop_2/499/CNN/summary.txt") as f:
	content = f.readlines()
	summary = list()
	for line in content [1:]:
		split_line = line.split(';')

		summary.append([split_line[0], round(float(split_line[5].strip())*100,2), round(float(split_line[6].strip())*100,2)])
		print(float(split_line[5].strip())*100, float(split_line[6].strip())*100)
print(summary)

comparison = [official[0]+["new MWE-F1", "new Tok-F1", "div MWE-F1", "div Tok-F1"]]
for line in official:
	new_line = line
	for s_line in summary:
		if line[0].strip() == s_line[0]:
			div_mwe = round(float(line[1]) - s_line[1],2)
			div_tok = round(float(line[2]) - s_line[2],2)
			new_line.extend(s_line[1:])
			new_line.extend([div_mwe, div_tok])
			comparison.append(new_line)

print(comparison)

with open("../Results/MWEFiles/testSet/loop_2/499/CNN/new_table.txt", "w") as f:
	writer = csv.writer(f, delimiter=";")
    	writer.writerows(comparison)

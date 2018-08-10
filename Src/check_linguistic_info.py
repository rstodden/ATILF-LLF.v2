import os, json
from collections import Counter

# test if exist and mkdir
# ../Results/features/corpus_info/
# ../Results/labels

for col in ["FEATS", "PARSEME:MWE", "UPOS", "XPOS", "DEPREL", "DEPS", "LEMMA"]:
	for file_type in ["train.cupt", "dev.cupt", "test.blind.cupt"]:
		counter_dict = dict()
		count_all = Counter()
		for lang_dir in os.listdir("../sharedtask_11/"):
			for filename in os.listdir("../sharedtask_11/"+lang_dir):
				if filename.endswith(file_type):
					lang_counter = Counter()
					with open("../sharedtask_11/"+lang_dir+'/'+filename) as f:
						content = f.readlines()
					for line in content:
						if line.startswith('# global.columns = '):
							header = line.split('# global.columns = ')[1].strip()
							nr_col = header.split(' ').index(col)
							print(nr_col)
						if not line.startswith("#") and line != '\n':
							col_value = line.strip().split('\t')[nr_col]
							if col_value == "_" or col_value == "*":
								lang_counter[col_value] += 1
								count_all[col_value] +=1
								continue
							if col == "FEATS":
								splitted_morpho_info = col_value.split("|")
								if len(splitted_morpho_info) > 1:
									for e in splitted_morpho_info:
										if "=" in e:
											e = e.split("=")[0]
										lang_counter[e] += 1
										count_all[e] += 1
								else:
									if "=" in splitted_morpho_info[0]:
										splitted_morpho_info = splitted_morpho_info[0].split("=")[0]
									else:
										splitted_morpho_info = splitted_morpho_info[0]
									lang_counter[splitted_morpho_info] += 1
									count_all[splitted_morpho_info] +=1
							elif col == "PARSEME:MWE":
								splitted_mwe_type = col_value.split(";")
								if len(splitted_mwe_type) > 1:
									for e in splitted_mwe_type:
										if ":" in e:
											e = e.split(":")[1]
										if e.isdigit():
											continue
										lang_counter[e] += 1
										count_all[e] += 1
								else:
									if ":" in col_value:
										col_value = col_value.split(":")[1]
									if col_value.isdigit():
										continue
									lang_counter[col_value] += 1
									count_all[col_value] +=1
							else:
								lang_counter[col_value] += 1
								count_all[col_value] +=1
							
					counter_dict[lang_dir] = lang_counter

		if col == "PARSEME:MWE":
			with open("../Results/labels/number_"+col+'_'+file_type.split(".")[0]+".json", "w") as f:
				json.dump(counter_dict, f)
			with open("../Results/labels/number_"+col+'_'+file_type.split(".")[0]+"_all.json", "w") as f:
				json.dump(count_all, f)
		else:
			with open("../Results/features/corpus_info/number_"+col+'_'+file_type.split(".")[0]+".json", "w") as f:
				json.dump(counter_dict, f)
			with open("../Results/features/corpus_info/number_"+col+'_'+file_type.split(".")[0]+"_all.json", "w") as f:
				json.dump(count_all, f)

from __future__ import division
import os, json
from collections import Counter
dict_number_mwes = {'train.cupt': {'EN': 331, 'BG': 5364, 'HE': 1236, 'PL': 4122, 'EU': 2823, 'HU': 6205, 'DE': 2820, 'RO': 4713, 'IT': 3254, 'EL': 1404, 'HR': 1450, 'LT': 312, 'HI': 534, 'FA': 2451, 'SL': 2378, 'PT': 4430, 'FR': 4550, 'ES': 1739, 'TR': 6125}, 'dev.cupt': {'EN': 0, 'BG': 670, 'HE': 501, 'PL': 515, 'EU': 500, 'HU': 779, 'DE': 503, 'RO': 589, 'IT': 500, 'EL': 500, 'HR': 500, 'LT': 0, 'HI': 0, 'FA': 501, 'SL': 500, 'PT': 553, 'FR': 629, 'ES': 500, 'TR': 510}}
#dict_number_mwes = dict()
file_type = "train.cupt" # "test.cupt" "dev.cupt"
for file_type in ["train.cupt", "dev.cupt"]:
	counter_dict = Counter()
	#dict_number_mwes[file_type] = dict()
	for lang_dir in os.listdir("../sharedtask_11/"):
		for filename in os.listdir("../sharedtask_11/"+lang_dir):
			if filename.endswith(file_type):
				counter_mwe_sent = Counter()
				lang_counter = Counter()
				with open("../sharedtask_11/"+lang_dir+'/'+filename) as f:
					content = f.readlines()
				for line in content:
					if line.startswith("# source_sent_id"):
						for mwe_id in counter_mwe_sent:
							# counts the number of mwe length per language
							lang_counter[counter_mwe_sent[mwe_id]] += 1
						counter_mwe_sent = Counter()
					if not line.startswith("#") and line != '\n':
						mwe_type = line.strip().split('\t')[-1]
						splitted_mwe_type = mwe_type.split(";")
						if len(splitted_mwe_type) > 1:
							for e in splitted_mwe_type:
								if ":" in e:
									e = e.split(":")[0]
								if e.isdigit():
									# sums element per each id of vme in a sentence
									counter_mwe_sent[e] += 1
								
						else:
							if ":" in mwe_type:
								mwe_type = mwe_type.split(":")[0]
							if mwe_type.isdigit():
								counter_mwe_sent[mwe_type] += 1
				if len(counter_mwe_sent) != 0:
					# counts of last sentence
					for mwe_id in counter_mwe_sent:
						lang_counter[counter_mwe_sent[mwe_id]] += 1
				counter_dict[lang_dir] = lang_counter
				#dict_number_mwes[file_type][lang_dir] = sum(lang_counter.values())
	#print(counter_dict)
	relative_counter_dict = dict()
	for lang in counter_dict:
		relative_counter_dict[lang] = dict()
		for count in counter_dict[lang]:
			relative_counter_dict[lang][count] = round((counter_dict[lang][count]/dict_number_mwes[file_type][lang])*100,2)
	with open("../Results/labels/number_tags_"+file_type.split(".")[0]+"_1.json", "w") as f:
		json.dump(relative_counter_dict, f)

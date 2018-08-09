for lang in ["FA", "RO", "EL",  "BG", "DE", "EL", "EN", "ES", "EU", "FR", "HE", "HI", "HR", "HU", "IT", "LT", "PL", "PT", "RO", "SL", "TR"]:
    for filetype in ["train.dev", "test"]:
        with open("../sharedtask_11/"+lang+"/"+filetype+".cupt") as corpusFile:
            # Read the corpus file
            print("filename", "../sharedtask_11/"+lang+"/"+filetype+".cupt")
            lines = corpusFile.readlines()


        parseme_content = ""
        conllu_content = ""
        for line in lines:
            if not line.startswith('#') and line != "\n":
                split_line = line.strip().split("\t")
                parseme_content += split_line[0]+"\t"+split_line[1]
                if split_line[-2] != '_':
                    parseme_content += "\t"+"nsp"
                else:
                    parseme_content += "\t" + "_"
                if split_line[-1] == "*" or split_line[-1] == "_":
                    parseme_content += "\t_\n"
                else:
                    parseme_content += "\t"+split_line[-1]+"\n"
                conllu_content += "\t".join(split_line[:-1])+"\n"

            elif line == "\n":
                parseme_content += "\n"
                conllu_content += "\n"

        with open("../sharedtask_11/"+lang+"/"+filetype+".parsemetsv", "w") as f:
            f.write(parseme_content)
        with open("../sharedtask_11/"+lang+"/"+filetype+".conllu", "w") as f:
            f.write(conllu_content)

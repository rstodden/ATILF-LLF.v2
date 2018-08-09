# run system
python ../Src/identifier.py

#evaluate system
dir=../Results/MWEFiles/testSet
for language in "PT" "FA" "BG" "DE" "EL" "EN" "ES" "EU" "FR" "HE" "HI" "HR" "HU" "IT" "LT" "PL" "PT" "RO" "SL" "TR"; do
	mkdir -p $dir/$language
	../bin/parsemetsv2cupt.py --input ../sharedtask_11/$language/test.parsemetsv >> $dir/$language/test.system.cupt
	../bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred $dir/$language/test.system.cupt >> $dir/$language.eval.txt

	tok=$(grep -E "^\* Tok-based:.*" $dir/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	mwe=$(grep -E "^\* MWE-based:.*" $dir/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	echo $language";"$mwe";"$tok >> $dir/summary.txt
done

../bin/average_of_evaluations.py $dir/*.eval.txt >> $dir/avg.txt
	tok=$(grep -E "^\* Tok-based:.*" $dir/avg.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	mwe=$(grep -E "^\* MWE-based:.*" $dir/avg.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
echo avg";"$mwe";"$tok >> $dir/summary.txt
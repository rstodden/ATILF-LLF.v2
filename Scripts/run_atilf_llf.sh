# run system
#python ../Src/identifier.py

#evaluate system
dir=../Results/MWEFiles/testSet/outputcodeclassifier
for language in "BG" "DE" "EL" "EN" "ES" "EU" "FA" "FR" "HE" "HI" "HR" "HU" "IT" "LT" "PL" "PT" "RO" "SL" "TR"; do
	mkdir -p $dir/$language
	../bin/parsemetsv2cupt.py --input $dir/$language.txt >> $dir/$language/test.system.cupt
	if [ $language = "EN" ]; then
	    ../bin/evaluate.py --gold ../sharedtask_11/$language/test_fixed.cupt --pred $dir/$language/test.system.cupt >> $dir/$language.eval.txt
	else
	    ../bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred $dir/$language/test.system.cupt >> $dir/$language.eval.txt
	fi


	tok=$(grep -E "^\* Tok-based:.*" $dir/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	mwe=$(grep -E "^\* MWE-based:.*" $dir/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	echo $language";"$mwe";"$tok >> $dir/summary.txt
done

../bin/average_of_evaluations.py $dir/*.eval.txt >> $dir/avg.txt
	tok=$(grep -E "^\* Tok-based:.*" $dir/avg.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	mwe=$(grep -E "^\* MWE-based:.*" $dir/avg.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
echo avg";"$mwe";"$tok >> $dir/summary.txt
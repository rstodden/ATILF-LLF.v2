#dir=../Results/MWEFiles/testSet/only_SVM
dir=../Results/MWEFiles/testSet/official_run/CNN/
echo "language;nrrun;epoches;modulo;duration;MWE_F-Score;Tok_F-Score" >> $dir/summary.txt
for language in "FA" "BG" "DE" "EL" "EN" "ES" "EU" "FR" "HE" "HI" "HR" "HU" "IT" "LT" "PL" "PT" "RO" "SL" "TR"; do
        epoches=15
        modulo=499
        nrrun=0
        cnn=1
        #for allFilters in 1 0; do
        #descr="all_feats"
	#mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo
	#mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM
	#mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language
	#mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN
	#mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language
        #prediction=0
        #python ../Src/identifier.py --svm $svm --cnn 1 --epoches $epoches --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 0 $language".json" "train.dev.cupt" "test.blind.cupt"
	#../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred $dir/$language/test.system.cupt
	../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred $dir/$language/test.system.cupt >> $dir/$language.eval.txt
	
	tok=$(grep -E "^\* Tok-based:.*" $dir/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	mwe=$(grep -E "^\* MWE-based:.*" $dir/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	echo $language";"$nrrun";"$epoches";"$modulo";"$time";"$mwe";"$tok >> $dir/summary.txt
done
../11/bin/average_of_evaluations.py $dir/*.eval.txt >> $dir/avg.txt
	tok=$(grep -E "^\* Tok-based:.*" $dir/avg.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
	mwe=$(grep -E "^\* MWE-based:.*" $dir/avg.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
echo avg";"$nrrun";"$epoches";"$modulo";"$time";"$mwe";"$tok >> $dir/summary.txt

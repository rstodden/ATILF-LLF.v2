nrrun=0
descr="pmi_only_devset" #$nrrun
epoches=15
modulo=499
cnn=1
for svm in 0; do
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language

            echo "language;nrrun;epoches;modulo;duration;MWE_F-Score;Tok_F-Score" >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN/summary_$descr.txt
            echo "language;nrrun;epoches;modulo;duration;MWE_F-Score;Tok_F-Score" >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/summary_$descr.txt
	    for language in "FA"; do # "BG" "DE" "EL" "EN" "ES" "EU" "FR" "HE" "HI" "HR" "HU" "IT" "LT" "PL" "PT" "RO" "SL" "TR"; do 
		start="$(date -u +%s)"
		
		#for allFilters in 1 0; do
		#descr="loop_"$nrrun
		prediction=0
		python ../Src/identifier.py --svm $svm --cnn 1 --epoches $epoches --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 0 $language".json" "train.dev.cupt" "test.blind.cupt"
		end="$(date -u +%s)"

		if [ $cnn = 1 ] && [ $svm = 1 ]; then
		    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language/test.system.cupt
		    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language/test.system.cupt >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language.eval.txt
		    time=$(( $end - $start ))
		    tok=$(grep -E "^\* Tok-based:.*" ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
		    mwe=$(grep -E "^\* MWE-based:.*" ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
		    echo $language";"$nrrun";"$epoches";"$modulo";"$time";"$mwe";"$tok >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/summary_$descr.txt


		elif [ $cnn = 1 ] && [ $svm = 0 ]; then

		    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language/test.system.cupt
		    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language/test.system.cupt >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval.txt
		    time=$(( $end - $start ))
		    tok=$(grep -E "^\* Tok-based:.*" ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
		    mwe=$(grep -E "^\* MWE-based:.*" ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
		    echo $language";"$nrrun";"$epoches";"$modulo";"$time";"$mwe";"$tok >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN/summary_$descr.txt

		fi
	    done
	#done
done

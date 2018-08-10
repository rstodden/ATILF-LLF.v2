
mkdir -p ../Results/model
mkdir -p ../Results/model/$modulo
mkdir -p ../Results/labels
mkdir -p ../Results/labels/$modulo
mkdir -p ../Results/features
mkdir -p ../Results/features/$modulo





for nrrun in $(seq 1 2); do
    for language in  "FA" "SL" "TR" "EN"; do # "EN" "DE" "HI" "HR" "SL" "TR" "FR" "IT" "PT"   "RO" "BG" "EL""EU" "HE" "HU" "LT" "PL" "PT"; do #"TR" "FR" "SL" "IT" "PT" "HR"; do # "FA" "RO" "BG" "DE" "EL" "EN" "ES" "EU" "FR" "HE" "HI" "HR" "HU" "IT" "LT" "PL" "PT" "SL"
        epoches=5
        modulo=499 #73
        svm=0
        cnn=1
        #for allFilters in 1 0; do
        descr="buxfixing_transitions_"$nrrun
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language
        prediction=0
        python ../Src/identifier.py --svm $svm --cnn 1 --epoches $epoches --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 0 $language".json" "train.dev.cupt" "test.blind.cupt"
	../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language/test.system.cupt
	../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language/test.system.cupt >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval
   done
done
# IOError: 3,453,974,208 requested and 2,545,697,776 written

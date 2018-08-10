mkdir -p ../Results/model
mkdir -p ../Results/model/$modulo
mkdir -p ../Results/labels
mkdir -p ../Results/labels/$modulo
mkdir -p ../Results/features
mkdir -p ../Results/features/$modulo

svm=0
for language in  "FA"; do # "EN" "DE" "HI" "HR" "SL" "TR" "FR" "IT" "PT"   "RO" "BG" "EL""EU" "HE" "HU" "LT" "PL" "PT"; do #"TR" "FR" "SL" "IT" "PT" "HR"; do # "FA" "RO" "BG" "DE" "EL" "EN" "ES" "EU" "FR" "HE" "HI" "HR" "HU" "IT" "LT" "PL" "PT" "SL"
    epoches=5
    modulo=499 #73
    nrrun=0
    cnn=1
    #for allFilters in 1 0; do
    descr="extern_feature"
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN
	mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language
    prediction=1
    python2 ../Src/identifier.py --svm $svm --cnn 1 --epoches $epoches --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 0 --use_extern_labels 1 --label_file "Results/result.reg.farsi_201806101815.txt" $language".json" "train.dev.cupt" "test.blind.cupt"
	../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/$nrrun/CNN/$language/test.system.cupt
	../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/$nrrun/CNN/$language/test.system.cupt >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval
done

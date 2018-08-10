nrrun=0
prediction=0
usebatch=0
descr="best_results"
echo $descr
mkdir -p ../Results/model
mkdir -p ../Results/model/$modulo
mkdir -p ../Results/labels
mkdir -p ../Results/labels/$modulo
mkdir -p ../Results/features
mkdir -p ../Results/features/$modulo

mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo
mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM
mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language
mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN
mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language

#for svm in 0; do
#    for epoches in 15; do
#        for modulo in 499 1000 2000 5000 10000; do
#            for language in "FA" "RO"  "EL"   "BG" "DE" "EL" "EN" "ES" "EU" "FR" "HE" "HI" "HR" "HU" "IT" "LT" "PL" "PT" "RO" "SL" "TR"; do
#                #for allFilters in 1 0; do
#                    descr="ppmi_mod"$modulo"_epoches"$epoches"_svm"$svm
#                    #python ../Src/identifier.py --svm 0 --cnn 1 --epoches 10 --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 1 --useMI $mi --useCHI2 $chi --numPerc $num_perc --usePercentile $percent $language".json" "a" "b"
#                    python ../Src/identifier.py --svm $svm --cnn 1 --epoches $epoches --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 0 $language".json" "train.dev.cupt" "dev.cupt"
#
#
#                    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/dev.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language/test.system.cupt
#                    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/dev.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language/test.system.cupt >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval.txt
#                #done
#            done
#        done
#    done
#done

for svm in 1; do
    for epoches in 15; do
        for modulo in 499; do
            for language in "BG" "EL" "ES" "FR" "HE" "HR" "IT" "LT" "PL" "PT" "SL" "TR"; do
                #for allFilters in 1 0; do
                    #descr="ppmi_mod"$modulo"_epoches"$epoches"_svm"$svm
                    #python ../Src/identifier.py --svm 0 --cnn 1 --epoches 10 --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 1 --useMI $mi --useCHI2 $chi --numPerc $num_perc --usePercentile $percent $language".json" "a" "b"
                    #python ../Src/identifier.py --svm $svm --cnn 1 --epoches $epoches --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 0 $language".json" "a" "b"
                    python ../Src/identifier.py --svm $svm --cnn 1 --epoches $epoches --number_modulo $modulo --nrrun $nrrun --predictonly 0 --descriptionPath $descr --useBatchGenerator 0 $language".json" "train.dev.cupt" "test.blind.cupt"


                    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language/test.system.cupt
                    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language/test.system.cupt >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language.eval.txt
                #done
            done
        done
    done
done


#../11/bin/evaluate.py --gold ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language/$language.gold.txt --pred ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language/test.system.cupt
    #../11/bin/evaluate.py --gold ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language/$language.gold.txt --pred ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language/test.system.cupt --debug | less -RS >> ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language"_epo"$epoches"_mod"$mod"_"$i.eval.txt

    #../11/bin/evaluate.py --gold ../sharedtask_11/$language/dev.cupt --pred ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language/test.system.cupt
    #../11/bin/evaluate.py --gold ../sharedtask_11/$language/dev.cupt --pred ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language/test.system.cupt --debug | less -RS >> ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language"_epo"5"_mod"300"_".eval.txt

    #python ../Src/identifier.py --svm 0 --cnn 1 --epoches 5 --number_modulo $modulo $language".json" "a" "b"


#!/bin/bash
testfile="a"
trainingfile="b"
corpora_version="1.1"


mkdir -p ../Results
mkdir -p ../Results/features
mkdir -p ../Results/labels
mkdir -p ../Results/model
mkdir -p ../Results/MWEFiles
mkdir -p ../Results/MWEFiles/testSet




cnn=1
epoches=10
for i in $(seq 1 5);  do
    #configfile=$language'.json'
    for mod in 300 500 750; do
        mkdir -p ../Results/MWEFiles/testSet/$mod
        mkdir -p ../Results/MWEFiles/testSet/$mod/CNN-SVM
        mkdir -p ../Results/MWEFiles/testSet/$mod/CNN
        mkdir -p ../Results/model/$mod
        mkdir -p ../Results/labels/$mod
        mkdir -p ../Results/features/$mod
        for dir in ../sharedtask_11/**; do
            language=${dir##*/}
            configfile=$language'.json'
            echo $language
            mkdir -p ../Results/MWEFiles/testSet/$mod/CNN/$language
            mkdir -p ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language
            for svm in 1 0; do
                python ../Src/identifier.py --svm $svm --cnn $cnn --epoches $epoches --number_modulo $mod $configfile $trainingfile $testfile
                if [ $cnn = 1 ] && [ $svm = 1 ]; then
                    echo $svm $cnn
                    mkdir -p ..Results/MWEFiles/testSet/CNN-SVM
                    mkdir -p ..Results/MWEFiles/testSet/CNN-SVM/$language
                    if [ $language = "EN" ] || [ $language = "HI" ] || [ $language = "LT" ]; then
                        ../11/bin/evaluate.py --gold ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language/$language.gold.txt --pred ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language/test.system.cupt
                        ../11/bin/evaluate.py --gold ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language/$language.gold.txt --pred ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language/test.system.cupt --debug | less -RS >> ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language"_epo"$epoches"_mod"$mod"_"$i.eval.txt
                    else
                        ../11/bin/evaluate.py --gold ../sharedtask_11/$language/dev.cupt --pred ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language/test.system.cupt
                        ../11/bin/evaluate.py --gold ../sharedtask_11/$language/dev.cupt --pred ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language/test.system.cupt --debug | less -RS >> ../Results/MWEFiles/testSet/$mod/CNN-SVM/$language"_epo"$epoches"_mod"$mod"_"$i.eval.txt
                    fi
                else
                    mkdir -p ..Results/MWEFiles/testSet/CNN
                    mkdir -p ..Results/MWEFiles/testSet/CNN/$language
                    if [ $language = "EN" ] || [ $language = "HI" ] || [ $language = "LT" ]; then
                        ../11/bin/evaluate.py --gold ../Results/MWEFiles/testSet/$mod/CNN/$language/$language.gold.txt --pred ../Results/MWEFiles/testSet/$mod/CNN/$language/test.system.cupt
                        ../11/bin/evaluate.py --gold ../Results/MWEFiles/testSet/$mod/CNN/$language/$language.gold.txt --pred ../Results/MWEFiles/testSet/$mod/CNN/$language/test.system.cupt --debug | less -RS >> ../Results/MWEFiles/testSet/$mod/CNN/$language"_epo"$epoches"_mod"$mod"_"$i.eval.txt
                    else
                        ../11/bin/evaluate.py --gold ../sharedtask_11/$language/dev.cupt --pred ../Results/MWEFiles/testSet/$mod/CNN/$language/test.system.cupt
                        ../11/bin/evaluate.py --gold ../sharedtask_11/$language/dev.cupt --pred ../Results/MWEFiles/testSet/$mod/CNN/$language/test.system.cupt --debug | less -RS >> ../Results/MWEFiles/testSet/$mod/CNN/$language"_epo"$epoches"_mod"$mod"_"$i.eval.txt
                    fi

                fi
            done
        done
    done
done


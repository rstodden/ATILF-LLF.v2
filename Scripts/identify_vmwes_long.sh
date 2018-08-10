#!/bin/bash
testfile="test.parsemetsv"
trainingfile="train.parsemetsv"
corpora_version="1.1"


mkdir -p ../Results
mkdir -p ..Results/features
mkdir -p ..Results/labels
mkdir -p ..Results/model
mkdir -p ..Results/MWEFiles
mkdir -p ..Results/MWEFiles/testSet

#configfile="DE.json"
#python ../Src/identifier.py --svm 1 --cnn 1 --epoches 5 --number_modulo 500 --corpora_version "1.1" $configfile $trainingfile $testfile
'cnn=1
configfile="FA.json"
language="FA"
mkdir -p ..Results/MWEFiles/testSet$language
for epoches in 10 20; do
    for svm in 1 0; do
        for mod in 250 500 750 1000; do
            for i in $(seq 1 5); do
                python ../Src/identifier.py --svm $svm --cnn $cnn --epoches $epoches --number_modulo $mod --corpora_version $corpora_version $configfile $trainingfile $testfile
                ../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt >> ../Results/MWEFiles/$language"_epo"$epoches"_mod"$mod"_svm"$svm"_cnn"$cnn"_"$i.eval
                ../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt
                done
            done
        done
    done
'



'
cnn=1
configfile="RO.json"
language="RO"
mkdir -p ..Results/MWEFiles/testSet/$language
for epoches in 10 20; do
    for svm in 1 0; do
        for mod in 250 500 750 1000; do
            for i in $(seq 1 5);  do
                python ../Src/identifier.py --svm $svm --cnn $cnn --epoches $epoches --number_modulo $mod $configfile $trainingfile $testfile
                ../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt >> ../Results/MWEFiles/$language"_epo"$epoches"_mod"$mod"_svm"$svm"_cnn"$cnn"_"$i.eval
                ../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt
                done
            done
        done
    done
'


cnn=1
configfile="DE.json"
language="DE"
epoches=10
for language in "DE" "FA" "HR" "ES"; do
    mkdir -p ..Results/MWEFiles/testSet/$language
    configfile=$language+'.json'
    for svm in 1 0; do
        for mod in 500 750; do
            for i in $(seq 1 5);  do
            echo $language
                python ../Src/identifier.py --svm $svm --cnn $cnn --epoches $epoches --number_modulo $mod $configfile $trainingfile $testfile
                ../11/bin/evaluate.py --gold ../Results/MWEFiles/testSet/$language/$language.gold.txt --pred ../Results/MWEFiles/testSet/$language/$language.txt >> ../Results/MWEFiles/$language"_epo"$epoches"_mod"$mod"_svm"$svm"_cnn"$cnn"_"$i.eval
                ../11/bin/evaluate.py --gold ../Results/MWEFiles/testSet/$language/$language.gold.txt --pred ../Results/MWEFiles/testSet/$language/$language.txt
                done
            done
        done
    done

'
cnn=1
configfile="SV.json"
language="SV"
mkdir -p ..Results/MWEFiles/testSet$language
for epoches in 10 20; do
    for svm in 1 0; do
        for mod in 250 500 750 1000; do
            for i in $(seq 1 5);  do
                python ../Src/identifier.py --svm $svm --cnn $cnn --epoches $epoches --number_modulo $mod $configfile $trainingfile $testfile
                ../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt >> ../Results/MWEFiles/$language"_epo"$epoches"_mod"$mod"_svm"$svm"_cnn"$cnn"_"$i.eval
                ../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt
                done
            done
        done
    done

#for configpath in ../FeatureGroups/**;
#    do
#        configfile=${configpath##*/}
#        language=${configfile%.*}
#        python ../Src/bash_play.py --epoches $epoches $configfile $trainingfile $testfile
#    done

#python ../Src/identifier.py --eopchs $epoches --cnn 1 --svm 1 $configfile $trainingfile $testfile
#../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt >> ../Results/MWEFiles/$language_$epoches_svm_cnn_$number.eval


# store gold and result file in directory
#for dir in ../Results/MWEFiles/testSet/**;
#    do
#        configfile=${dir##*/}
#        language=${configfile%.*}
#       echo "Results of" $language;
#        ../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt >> ../Results/MWEFiles/$language_$epoches.eval
#        ../Baseline/sharedTaskEv/bin/evaluate.py ../Results/MWEFiles/testSet/$language/$language.gold.txt ../Results/MWEFiles/testSet/$language/$language.txt
#    done

#change svm / cnn and epoches

'


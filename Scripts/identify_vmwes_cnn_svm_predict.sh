descr_short="multichannel_loop"
mkdir -p ../Results/MWEFiles/testSet/$descr_short
echo "language;decription;nrrun;epoches;modulo;duration;svm;cnn;ppmi;trainset;multichannel;kernelsizes;dim_hidden;filters;batchsize;embedding_dims;dropout;compile_metric;pooling;MWE_F-Score;Tok_F-Score" >> ../Results/MWEFiles/testSet/$descr_short/summary.txt
echo "language;decription;nrrun;epoches;modulo;duration;svm;cnn;ppmi;trainset;multichannel;kernelsizes;dim_hidden;filters;batchsize;embedding_dims;dropout;compile_metric;pooling;MWE_F-Score;Tok_F-Score" >> ../Results/MWEFiles/testSet/$descr_short/summary.txt
for svm in 0 1; do
    for modulo in 499 997; do
        for kernelsizes in "4,6,8" "2,4,6,8"; do
            for embedding_dims in 90 180; do
                for dim_hidden in 50 150 300; do
                    for pooling in "max_pool" "avg_pool"; do
                        for nrrun in $(seq 1 3); do
                            epoches=15
                            cnn=1
                            ppmi=0
                            trainset="train+dev"
                            multichannel=1
                            filters=32
                            batchsize=32
                            dropout=0.01
                            compile_metric="fbeta_score"
                            mkdir -p ../Results/model
                            mkdir -p ../Results/model/$modulo
                            mkdir -p ../Results/labels
                            mkdir -p ../Results/labels/$modulo
                            mkdir -p ../Results/features
                            mkdir -p ../Results/features/$modulo
                            mkdir -p ../Results/MWEFiles/testSet/$modulo
                            mkdir -p ../Results/MWEFiles/testSet/$modulo/CNN-SVM
                            mkdir -p ../Results/MWEFiles/testSet/$modulo/CNN

                            descr=$descr_short"_"$modulo"_"$kernelsizes"_"$embedding_dims"_"$dim_hidden"_"$pooling"_"$nrrun
                            mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo
                            mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM
                            mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN


                            #for language in "FA" "BG" "DE" "EL" "EN" "ES" "EU" "FR" "HE" "HI" "HR" "HU" "IT" "LT" "PL" "PT" "RO" "SL" "TR"; do
                            for language in "FA" "SL" "TR"; do
                                mkdir -p ../Results/MWEFiles/testSet/$modulo/CNN/$language
                                mkdir -p ../Results/MWEFiles/testSet/$modulo/CNN-SVM/$language
                                mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language
                                mkdir -p ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language
                                start="$(date -u +%s)"
                                prediction=0
                                python ../Src/identifier.py --svm $svm --cnn 1 --epoches $epoches --number_modulo $modulo --nrrun $nrrun --predictonly $prediction --descriptionPath $descr --useBatchGenerator 0 --kernelsizes $kernelsizes --dimensions_hiddenlayer $dim_hidden --filters $filters --batchsize $batchsize --dimensions_embedding $embedding_dims --dropout $dropout --compile_metric $compile_metric --pooling $pooling --ppmi $ppmi  $language".json" "train.dev.cupt" "test.blind.cupt"
                                end="$(date -u +%s)"

                                if [ $cnn = 1 ] && [ $svm = 1 ]; then
                                    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language/test.system.cupt
                                    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language/test.system.cupt >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language.eval.txt
                                    time=$(( $end - $start ))
                                    tok=$(grep -E "^\* Tok-based:.*" ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
                                    mwe=$(grep -E "^\* MWE-based:.*" ../Results/MWEFiles/testSet/$descr/$modulo/CNN-SVM/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
                                    echo $language";"$descr";"$nrrun";"$epoches";"$modulo";"$time";"$svm";"$cnn";"$ppmi";"$trainset";"$multichannel";"$kernelsizes";"$dim_hidden";"$filters";"$batchsize";"$embedding_dims";"$dropout";"$compile_metric";"$mwe";"$tok >> ../Results/MWEFiles/testSet/$descr_short/summary.txt

                                elif [ $cnn = 1 ] && [ $svm = 0 ]; then

                                    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language/test.system.cupt
                                    ../11/bin/evaluate.py --gold ../sharedtask_11/$language/test.cupt --pred ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language/test.system.cupt >> ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval.txt
                                    time=$(( $end - $start ))
                                    tok=$(grep -E "^\* Tok-based:.*" ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
                                    mwe=$(grep -E "^\* MWE-based:.*" ../Results/MWEFiles/testSet/$descr/$modulo/CNN/$language.eval.txt | grep  -P -o "(?<=F=)[\d|\.]{0,6}")
                                    echo $language";"$descr";"$nrrun";"$epoches";"$modulo";"$time";"$svm";"$cnn";"$ppmi";"$trainset";"$multichannel";"$kernelsizes";"$dim_hidden";"$filters";"$batchsize";"$embedding_dims";"$dropout";"$compile_metric";"$pooling";"$mwe";"$tok >> ../Results/MWEFiles/testSet/$descr_short/summary.txt

                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done






#!/bin/bash

for dir in ../sharedtask_11/**; do
    { cat $dir"/train.cupt"; sed "1d" $dir"/dev.cupt"; } > $dir"/train.dev.cupt"
    language=${dir##*/}
done
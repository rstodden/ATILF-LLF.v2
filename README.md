This system has participated in the PARSEME MWE shared task 1.1  on automatic identification of verbal multiword expressions (VMWEs). 
See shared task results [here](http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_50_Shared_task_results)

The system is a modified arc-standard transition model for the identification of the verbal multi-word expressions. 
It's a modified version of [ATILF-LLF System of Saied, H. A., Constant, M., and Candito, M. (2017)](https://github.com/hazemalsaied/ATILF-LLF.v2), we combining a data-independent dimension reduction with convolutinal neural networks.
  

**Corpora**: Data sets were provided by [PARSEME MWE Shared task 1.1](https://gitlab.com/parseme/sharedtask-data).

**Requirements** 
* python 2.7
* keras with a tensorflow backend
* scikit-learn 0.19 or higher
* numpy 

**SourceCode**: The source code is in the Src folder

**Run**
To run the code pick one of the files in the Scripts folder.
There are multiple options to run the code, you can run the code with only SVM, only CNN, a CNN with a SVM and also changing much parameters, 
e.g. number of epoches, number of dimensions after feature reduction, ...

**Results**: results of experiments on 19 languages with multiple reports reflecting the progress of the identification process.

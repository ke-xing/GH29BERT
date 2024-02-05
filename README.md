 # GH29BERT
- This repository contains the code and testing sequence data for reproduce the prediction results for GH29BERT, a protein functional cluster prediction model for GH29 family sequences. It is trained based on a semi-supervised deep learning method with a. 34,258 unlabeled and non-redundant GH29 sequences (i.e., unlabelled data) extracted from CAZy and Interpro databases and b. 2,796 labelled sequences with 45 cluster classes based on a thorough SSN analysis.
  For clear representation and easy reproduction, we provide a Jupyter notebook show the executable code and testing results, including:  
- prediction performance of the Xx (2,796 *20\%) labelled GH29 sequences,
- prediction results for 14 known-label (characterized) sequences that was excluded from both pre-training and task-training,
- visualization of GH29 representations by UMAP.

GH29BERT model is also accessible through a friendly user-interface: https://huggingface.co/spaces/Oiliver/GH29BERT.  

 ## Prerequisites for environment preparation
  （参考，需要修改。导出.yml 或者dockerfile都行，看你方便，附上配置环境的命令行）  
% tensorflow == 2.3.1  
% keras== 2.4.3  
%  Please check all the useful packages in the file **env.yml**.  
 % With the help of **conda**, just run `conda env create --file env.yml` to create an independent environment to implement the experiments.
 

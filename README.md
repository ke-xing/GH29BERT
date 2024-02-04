- # GH29BERT
- This repository contains the code and testing sequence data for reproduce the prediction results for GH29BERT, a protein functional cluster prediction model devised for GH29 family sequences. It is trained based on a semi-supervised deep learning method with:
	- a. 34,258 unlabeled and non-redundant GH29 sequences (i.e., unlabelled data) extracted from CAZy and Interpro databases and
	- b. 2,796 labelled sequences with 45 cluster classes based on a thorough SSN analysis.
- Specifically, the reproducible testing materials (code and data) on following two types of GH29 sequences used in submitted manuscript are provided, including:
	- 559 labelled GH29 testing sequences (2,796 labelled data with a random 80%-20% split for training and testing), see file `data/test.fasta`
	- 15 held-out characterized sequences that was excluded from both pre-training and task-training, see file `data/15_seq_for-test.fasta`
- ## Interactive deployment of GH29BERT for prediction testing
	- GH29BERT model is also accessible through a friendly user-interface on HuggingFace: https://huggingface.co/spaces/Oiliver/GH29BERT.
	- It is easier to test the above provided GH29 sequences or your custom GH29 sequence using this web tool.
- ## Prerequisites
	- ### Repository download
		- To get started, clone this repository, e.g., execute the following in the terminal: `git clone https://github.com/ke-xing/GH29BERT.git`
	- ### Environment preparation
		- Please check all the useful packages in the file **environment.yml**.   
		  With the help of [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html), run `conda env create --file environment.yml` to create an independent environment for implementing the testing  
	- ### Model parameter download
		- Due to the limit of single file size of GitHub repository, we upload the model parameter files at another open repository
			- GH29BERT
				-
				  ```python
				  # Load GH29BERT pre-trained model
				  GH29BERT=torch.load('transformer1500_95p_500.pt')
				  GH29BERT=GH29BERT.module
				  GH29BERT=GH29BERT.to('cuda:0')
				  # Load GH29BERT task model
				  downstream_GH29BERT=torch.load('down_model_500_kfold1.pt').to('cuda:0')
				  ```
			- ProtT5-XL
				- Reproducing prediction testing based on [pre-trained ProtT5-XL](https://ieeexplore.ieee.org/document/9477085) requires installing extra dependency libraries:
					-
					  ```
					  pip install torch
					  pip install transformers
					  pip install sentencepiece
					  ```
					- For more details, please follow the instructions of ProtTrans repository from [github](https://github.com/agemagician/ProtTrans/?tab=readme-ov-file) and [huggingface](https://huggingface.co/docs/transformers/installation).
				-
				  ```python
				  from transformers import T5Tokenizer, T5EncoderModel
				  
				  # Load ProtT5_XL pre-trained model
				  ProtT5_XL=T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc",cache_dir='./').to('cuda:0')
				  # Load ProtT5_XL task model
				  downstream_ProtT5_XL=torch.load('down_model_500_kfold1.pt').to('cuda:0')
				  ```
- ## Cluster prediction
	- Run `python python test.py` for predicting the fasta data. Model and data loading directory should be adjusted if need.
- ## Representation visualization
	- The visualization of GH29 representations with GH29BERT or other pre-training models can be implemented through `python visualization by UMAP.py` for obtaining the dimension-reduced intermediate representations and run `python figure1.py figure2.py` to get the visualization map.
- ## Code for model training
	- We also provide the model training code for pre-training  and downstream task-training. Run `python Pretrain/transformer/transformer_train.py` for GH29BERT model pre-training. Run`python classification/downstream_embedding.py` for loading the pre-trained model parameters and the embedding data(.npz) preparing for the task-training, and then run `python classification/downstream_train.py` for cluster prediction for task-training.
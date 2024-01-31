# GH29BERT

- This repository contains the code and testing sequence data for reproduce the prediction results for GH29BERT, a protein functional cluster prediction model for GH29 family sequences. It is trained based on a semi-supervised deep learning method with a. 34,258 unlabeled and non-redundant GH29 sequences (i.e., unlabelled data) extracted from CAZy and Interpro databases and b. 2,796 labelled sequences with 45 cluster classes based on a thorough SSN analysis.
  For clear representation and easy reproduction, we provide a Jupyter notebook show the executable code and testing results, including:  
- prediction performance of the Xx (2,796 *20\%) labelled GH29 sequences,
- prediction results for 14 known-label (characterized) sequences that was excluded from both pre-training and task-training,
- visualization of GH29 representations by UMAP.

GH29BERT model is also accessible through a friendly user-interface: https://huggingface.co/spaces/Oiliver/GH29BERT.  

---



## Pretraining

run:

```python
python Pretrain/transformer/transformer_train.py
```

to pretrain the bert-based model

---



## Task-training

run:

```python
python classification/downstream_embedding.py
```

to load the pretrained model and get the embedding data(.npz) preparing for the task-training

then run:

```python
python classification/downstream_train.py
```

to train the classify-task model

---



## Prediction

run:

```python
python test.py
```

to preditiction your own fasta data

---



## visualization

run:

```python
python visualization by UMAP.py
```

to get the reduction data by UMAP

then run:

```python
python figure1.py figure2.py
```

to get the visualization map





---



## Prerequisites for environment preparation

> environment.yml
> name: torch  
> channels:  
> 
> - defaults  
>   dependencies:  
> - bzip2=1.0.8=he774522_0  
> - ca-certificates=2023.12.12=haa95532_0  
> - libffi=3.4.4=hd77b12b_0  
> - openssl=3.0.12=h2bbff1b_0  
> - pip=23.3.1=py311haa95532_0  
> - python=3.11.7=he1021f5_0  
> - setuptools=68.2.2=py311haa95532_0  
> - sqlite=3.41.2=h2bbff1b_0  
> - tk=8.6.12=h2bbff1b_0  
> - tzdata=2023d=h04d1e81_0  
> - vc=14.2=h21ff451_1  
> - vs2015_runtime=14.27.29016=h5e58377_2  
> - wheel=0.41.2=py311haa95532_0  
> - xz=5.4.5=h8cc25b3_0  
> - zlib=1.2.13=h8cc25b3_0  
> - pip:  
>   - biopython==1.83  
>   - boto3==1.34.22  
>   - botocore==1.34.22  
>   - certifi==2023.11.17  
>   - charset-normalizer==3.3.2  
>   - colorama==0.4.6  
>   - distlib==0.3.8  
>   - filelock==3.13.1  
>   - fsspec==2023.12.2  
>   - idna==3.6  
>   - jinja2==3.1.3  
>   - jmespath==1.0.1  
>   - joblib==1.3.2  
>   - lmdb==1.4.1  
>   - markupsafe==2.1.3  
>   - mpmath==1.3.0  
>   - networkx==3.2.1  
>   - numpy==1.26.3  
>   - packaging==23.2  
>   - pillow==10.2.0  
>   - pipenv==2023.11.17  
>   - platformdirs==4.2.0  
>   - protobuf==4.25.2  
>   - python-dateutil==2.8.2  
>   - requests==2.31.0  
>   - s3transfer==0.10.0  
>   - scikit-learn==1.4.0  
>   - scipy==1.11.4  
>   - six==1.16.0  
>   - sympy==1.12  
>   - tape-proteins==0.5  
>   - tensorboardx==2.6.2.2  
>   - threadpoolctl==3.2.0  
>   - torch==2.1.2  
>   - torchaudio==2.1.2  
>   - torchvision==0.16.2  
>   - tqdm==4.66.1  
>   - typing-extensions==4.9.0  
>   - urllib3==2.0.7  
>   - virtualenv==20.25.0  
>     prefix: C:\Users\15275\anaconda3\envs\torch
>   - Please check all the useful packages in the file **environment.yml**.  
>   - With the help of **conda**, just run `conda env create --file environment.yml` to create an independent environment to implement the experiments.

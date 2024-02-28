# DeepLION2<img  align="left" src="Figures/Lion.png" width="45" height="45" > 

Deep Multi-Instance Contrastive Learning Framework Enhancing the Prediction of Cancer-Associated T Cell Receptors by Attention Strategy on Motifs
------------------------------

DeepLION2 is an innovative deep multi-instance contrastive learning framework specifically designed to enhance cancer-associated T cell receptor (TCR) prediction. It setups a three-component workflow, comprising data preprocessing, TCR antigen-specificity extraction, and multi-instance learning. In the third part, DeepLION2 introduced a content-based sparse self-attention mechanism in conjunction with contrastive learning to effectively aggregate TCR features and embed the TCR repertoire. By considering the relationships among TCRs within the repertoire and the sparsity of caTCRs, it significantly enhanced the aggregation process, enabling accurate prediction of whether the TCR repertoire was cancerous or non-cancerous. For more details, please read our paper [`DeepLION2: deep multi-instance contrastive learning framework enhancing the prediction of cancer-associated T cell receptors by attention strategy on motifs`](https://doi.org/10.3389/fimmu.2024.1345586).

<p float="left">
  <img src="Figures/DeepLION2_workflow.png" width="781" height="646"/>
</p>


## Content

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Project Organization](#project-organization)
- [Usage](#usage)
  - [Python package versions](#python-package-versions)
  - [Making predictions using the pre-trained model](#making-predictions-using-the-pre-trained-model)
  - [Training DeepLION2 models](#training-deeplion2-models)
  - [Identifying TCR motifs](#identifying-tcr-motifs)
- [Citation](#citation)
- [Contacts](#contacts)


<!-- /code_chunk_output -->



## Project Organization

    ├── LICENSE                         <- Non-commercial license.
    │     
    ├── README.md                       <- The top-level README for users using DeepLION2.
    │ 
    ├── Codes                           <- Python scripts of DeepLION2. See README for their usages.
    │   ├── caRepertoire_prediction.py  <- Making predictions, training models and identifying motifs on the samples.
    │   ├── loss.py                     <- Containing loss functions used in networks.
    │   ├── network.py                  <- All networks used in DeepLION2.
    │   ├── preprocess.py               <- Processing raw TCR-sequencing data files.
    │   └── utils.py                    <- Containing the commonly used function components.
    │ 
    ├── Data                            <- Data used in DeepLION2.
    │   ├── Geneplus
    │   │   ├── GICA
    │   │   ├── LUCA
    │   │   └── THCA
    │   │ 
    │   ├── RawData
    │   │   └── Example_raw_file.tsv
    │   │     
    │   └──AAidx_PCA.txt
    │
    ├── Figures                         <- Figures used in README.
    │   ├── DeepLION2_workflow.png
    │   └── Lion.png
    │  
    ├── Models                          <- Pre-trained DeepLION2 models for users making predictions directly.                             
    │   ├── Geneplus 
    │   │   └── THCA
    │   │      
    │   └── Pretrained_TCRD.pth
    │      
    └── Results                         <- Some results of using DeepLION2.
        ├── ProcessedData              
        ├── ProcessedDataPredicted    
        ├── THCA_motifs  
        ├── DeepLION2_THCA_test0.pth
        └── DeepLION2_THCA_test0.tsv     

## Usage

### Python package versions

DeepLION2 works perfectly in the following versions of the Python packages:

```
Python          3.7.2
matplotlib      3.0.2
numpy           1.21.6
scikit-learn    0.24.2
sparsemax       0.1.9
torch           1.10.1
torchaudio      0.10.1
torchvision     0.4.2+cpu
```

### Making predictions using the pre-trained model

Users can use the pre-trained models we provided in `./Models/` to make predictions directly.

First, we need to collect the raw TCR-sequencing data files in one directory, such as `./Data/RawData/Example_raw_file.tsv`, and use the Python script `./Codes/preprocess.py` to process them by this command:

```
python ./Codes/preprocess.py --sample_dir ./Data/RawData/ --info_index [-3,5,2] --aa_file ./Data/AAidx_PCA.txt --save_dir ./Results/ProcessedData/
```

After processing, TCRs with their information are saved in `./Results/ProcessedData/RawData/0/Example_raw_file.tsv_processed.tsv`:

```
amino_acid  v_gene  frequency   target_seq
CASSLTRLGVYGYTF TRBV6-6*05  0.06351 CASSLTRLGVYGYTF
CASSKREIHPTQYF  TRBV28*01(179.7)    0.043778    CASSKREIHPTQYF
CASSLEGGAAMGEKLFF   TRBV28*01(194.7)    0.039882    CASSLEGGAAMGEKLFF
CASSPPDRGAFF    TRBV28*01(179.5)    0.034422    CASSPPDRGAFF
CASSTGTAQYF TRBV19*03   0.028211    CASSTGTAQYF
CASSEALQNYGYTF  TRBV2*01(255.6) 0.027918    CASSEALQNYGYTF
CSARADRGQGYEQYF TRBV20-1*01 0.027427    CSARADRGQGYEQYF
CASSPWAATNEKLFF TRBV28*01(179.7)    0.023224    CASSPWAATNEKLFF
CAWGWTGGTYEQYF  TRBV30*05   0.019363    CAWGWTGGTYEQYF
······
```

If users get raw files in different format, they can also apply this script by setting the argument `--info_index` (`default: [-3,5,2]`) to the indexes of CDR3 sequences, their V gene and clone fraction information in their files. The users can also use the argument `--ration` to split the samples. For example, if they want to randomly divide the samples into five equal parts, set this argument to `[0.2, 0.2, 0.2, 0.2, 0.2]`. Besides, a pretrained model TCRD from DeepLION is provided for predicting TCRs in the preprocess. According to the predictedTCR scores, users can screen key TCRs for downstream analysis. The specific command is as follows:

```
python ./Codes/preprocess.py --sample_dir ./Data/RawData/ --info_index [-3,5,2] --aa_file ./Data/AAidx_PCA.txt --save_dir ./Results/ProcessedDataPredicted/ --model_file ./Models/Pretrained_TCRD.pth 
```

Then, the preprocessed samples of the Geneplus dataset are included and the samples of each cancer type are randomly divided into five equal parts. For example, the samples of the thyroid cancer are in `./Data/Geneplus/THCA/`. We pre-trained the DeepLION2 model on the sample sets 1, 2, 3 and 4, and obtained the model `./Models/Geneplus/THCA/DeepLION2_THCA_test0.pth`. Then, we can use the Python script `./Codes/caRepertoire_prediction.py` to make predictions on the sample set 0 `./Data/Geneplus/THCA/0/` using the pre-trained model by this command:

```
python ./Codes/caRepertoire_prediction.py --network DeepLION2 --mode 0 --sample_dir ./Data/Geneplus/THCA/0/ --aa_file ./Data/AAidx_PCA.txt --model_file ./Models/Geneplus/THCA/DeepLION2_THCA_test0.pth --record_file ./Results/DeepLION2_THCA_test0.tsv
```

The prediction results, including sample filenames and probabilities of being cancer-associated, are saved in `./Results/DeepLION2_THCA_test0.tsv`:

```
negative_HUMRAEJDZPEI-113_random.tsv_processed_TCRD.tsv 0.43275627780195675
negative_HUMRNEQKBPEI-122_random.tsv_processed_TCRD.tsv 0.224548884798489
negative_HUMRNEQKEPEI-149_random.tsv_processed_TCRD.tsv 0.35346354466520075
negative_HUMRNEQKFPEI-124_random.tsv_processed_TCRD.tsv 0.34982483125127034
negative_HUMRNEUSGPEI-105_random.tsv_processed_TCRD.tsv 0.4477410045421965
negative_HUMRNEUSHPEI-130_random.tsv_processed_TCRD.tsv 0.21079997918044313
negative_HUMRNEUSVPEI-116_random.tsv_processed_TCRD.tsv 0.1893240348761181
······
```

And the metrics, accuracy, sensitivity, specificity, Matthews correlation coefficient (MCC), and area under the receiver operating characteristic (ROC) curve (AUC), are calculated and printed as: 

```
Accuracy =  0.8604651162790697
Sensitivity =  0.7027027027027027
Specificity =  0.9795918367346939
MCC =  0.727885234638544
AUC =  0.960838389409818
```

### Training DeepLION2 models

Users can use the Python script `./Codes/caRepertoire_prediction.py` to train their own DeepLION2 models on their TCR-sequencing data samples for a better prediction performance. For example, we can train the DeepLION2 model on the THCA sample sets 1, 2, 3 and 4, by this command:

```
python ./Codes/caRepertoire_prediction.py --network DeepLION2 --mode 1 --sample_dir ['./Data/Geneplus/THCA/1/','./Data/Geneplus/THCA/2/','./Data/Geneplus/THCA/3/','./Data/Geneplus/THCA/4/'] --aa_file ./Data/AAidx_PCA.txt --model_file ./Results/DeepLION2_THCA_test0.pth
```

After the training process, the final model can be found in `./Results/Geneplus/THCA/DeepLION2_THCA_test0.pth`

### Identifying TCR motifs

Users can use the Python script `./Codes/caRepertoire_prediction.py` to identify the key motifs on their TCR-sequencing data samples for further analysis. For example, we can use the pre-trained DeepLION2 model `./Models/Geneplus/THCA/DeepLION2_THCA_test0.pth` to identify the motifs of TCRs in the THCA sample set 0 `./Data/Geneplus/THCA/0/` by this command:

```
python ./Codes/caRepertoire_prediction.py --network DeepLION2 --mode 2 --sample_dir ./Data/Geneplus/THCA/0/ --aa_file ./Data/AAidx_PCA.txt --model_file ./Models/Geneplus/THCA/DeepLION2_THCA_test0.pth --record_file ./Results/THCA_motifs/
```

The prediction results are saved in `./Results/THCA_motifs/`

## Citation

When using our results or modelling approach in a publication, please cite our papers (DeepLION2: [https://doi.org/10.3389/fimmu.2024.1345586](https://doi.org/10.3389/fimmu.2024.1345586); DeepLION: [https://doi.org/10.3389/fgene.2022.860510](https://doi.org/10.3389/fgene.2022.860510)):

>Qian X, Yang G, Li F, Zhang X, Zhu X, Lai X, Xiao X, Wang T and Wang J (2024) DeepLION2: deep multi-instance contrastive learning framework enhancing the prediction of cancer-associated T cell receptors by attention strategy on motifs. *Front. Immunol.* 15:1345586. doi: 10.3389/fimmu.2024.1345586

>Xu Y, Qian X, Zhang X, Lai X, Liu Y and Wang J (2022) DeepLION: Deep Multi-Instance Learning Improves the Prediction of Cancer-Associated T Cell Receptors for Accurate Cancer Detection. *Front. Genet.* 13:860510. doi: 10.3389/fgene.2022.860510

## Contacts

DeepLION2 is actively maintained by Xinyang Qian, currently a Ph.D student at Xi'an Jiaotong University in the research group of Prof. Jiayin Wang.

If you have any questions, please contact us by e-mail: qianxy@stu.xjtu.edu.cn.

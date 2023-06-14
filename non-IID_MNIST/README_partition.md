# NONIID DATASET

`partition.py` is taken from [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench  and related to the paper [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/pdf/2102.02079.pdf).


It can be used to divide tabular datasets (csv format) into multiple files using our non-IID partitioning strategies. Column `Class` in the header is recognized as label. 

In our case, we have taken as example this [dataset](https://www.kaggle.com/datasets/aryarishabh/of-genomes-and-genetics-hackerearth-ml-challenge), imported as `train.csv` and `test.csv` in this directory. This dataset should not be used as is in our work, because it does not give good results even in the case of a centralized learning algorithm like SVM.  



## Label Distribution Skew

-   **Quantity-based label imbalance**: each party owns data samples of a fixed number of labels.
-   **Distribution-based label imbalance**: each party is allocated a proportion of the samples of each label according to Dirichlet distribution. 

### Feature Distribution Skew

-   **Noise-based feature imbalance**: We first divide the whole dataset into multiple parties randomly and equally. For each party, we add different levels of Gaussian noises.
-   **Synthetic feature imbalance**: For generated 3D data set, we allocate two parts which are symmetric of(0,0,0) to a subset for each party.
-   **Real-world feature imbalance**: For FEMNIST, we divide and assign the writers (and their characters) into each party randomly and equally. 

### Quantity Skew

-   While the data distribution may still be consistent amongthe parties, the size of local dataset varies according to Dirichlet distribution.

## Data Partition Map

The default value of `noise` is 0 unless stated. We list the way to get our data partition as follow. 

- **Quantity-based label imbalance**: `partition`=`noniid-#label1`, `noniid-#label2` or `noniid-#label3` 
- **Distribution-based label imbalance**: `partition`=`noniid-labeldir`, `beta`=`0.5` or `0.1` 
- **Noise-based feature imbalance**: `partition`=`homo`, `noise`=`0.1` (actually noise does not affect `net_dataidx_map`) 
- **Synthetic feature imbalance & Real-world feature imbalance**: `partition`=`real` 
- **Quantity Skew**: `partition`=`iid-diff-quantity`, `beta`=`0.5` or `0.1` 
- **IID Setting**: `partition`=`homo` 
- **Mixed skew**: `partition` = `mixed` for mixture of distribution-based label imbalance and quantity skew; `partition` = `noniid-labeldir` and `noise` = `0.1` for mixture of distribution-based label imbalance and noise-based feature imbalance.

## Usage

Here is one example to run this code (see `partition_to_file.sh`):

```         
python partition.py --n_parties=10 \
  --partition=homo \
  --beta=0.5\
  --datadir="train.csv" \
  --outputdir="genetic" \
  --init_seed=0

```

We can check labels distribution in each parition by using `check_parition_ipynb` script.

## Citation

If you find this repository useful, please cite our paper:

```         
@inproceedings{li2022federated,
      title={Federated Learning on Non-IID Data Silos: An Experimental Study},
      author={Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng},
      booktitle={IEEE International Conference on Data Engineering},
      year={2022}
}
```


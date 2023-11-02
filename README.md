# TAGNet

Implementation of the paper ["Distance-Based Propagation for Efficient Knowledge Graph Reasoning"]() (EMNLP 2023).

## Abstract

Knowledge graph completion (KGC) aims to predict unseen edges in knowledge graphs (KGs), resulting in the discovery of new facts. A new class of methods have been proposed to tackle this problem by aggregating path information. These methods have shown tremendous ability in the task of KGC. However they are plagued by efficiency issues. Though there are a few recent attempts to address this through learnable path pruning, they often sacrifice the performance to gain efficiency. In this work, we identify two intrinsic limitations of these methods that affect the efficiency and representation quality. To address the limitations, we introduce a new method, TAGNet, which is able to efficiently propagate information. This is achieved by only aggregating paths in a fixed window for each source-target pair. We demonstrate that the complexity of TAGNet is independent of the number of layers. Extensive experiments demonstrate that TAGNet can cut down on the number of propagated messages by as much as 90\% while achieving competitive performance on multiple KG datasets.

## Requirements

All experiments were conducted using python 3.9.13. 

For the required python packages, please see `requirements.txt`.

## Reproduce Results

To reproduce the transductive results in table 1, please run the following scripts:
```
cd scripts

# TAGNet
bash fb15k237_transductive.sh
bash wn18rr_transductive.sh

# A*Net + TAGNet
bash tagnet_astar_transductive.sh
```

To reproduce the inductive results in table 2, please run the following scripts:
```
cd scripts

# TAGNet
bash fb15k237_inductive.sh
bash wn18rr_inductive.sh

# A*Net + TAGNet
bash tagnet_astar_inductive.sh
```

## Acknowledgements

The code is modified from the NBFNet-PyG codebase (https://github.com/KiddoZhu/NBFNet-PyG). We sincerely thank them for their contributions.

## Cite
```
@inproceedings{shomer23distance,
  title={Distance-Based Propagation for Efficient Knowledge Graph Reasoning},
  author={Shomer, Harry and Ma, Yao and Li, Juanhui and Wu, Bo and Aggarwal, Charu C. and Tang, Jiliang},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}
```


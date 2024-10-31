# SST-DPN
This is the official repository to the paper "[A Spatial-Spectral and Temporal Dual Prototype Network for Motor Imagery Brain-Computer Interface](https://arxiv.org/pdf/2407.03177)".

## Abstract
![image](https://github.com/hancan16/EDPNet/blob/main/figs/framework.png)
- To extract powerful spatial-spectral features, we design a lightweight attention mechanism that explicitly models the relationships among multiple channels in the spatial-spectral dimension. This method enables finer-grained spatial feature modeling, highlighting key spatial-spectral channels for the current MI task.
- To capture long-term temporal features from high temporal resolution EEG signals, we develop a Multi-scale Variance Pooling (MVP) module with large kernels. Compared to commonly used transformers, the MVP module is parameter-free and computationally efficient. Extensive experiments show that MVP outperforms transformers, indicating its potential as an alternative for capturing long-term temporal features in EEG signal decoding and real-time BCI applications.
- To overcome the small-sample issue, we propose a novel Dual Prototype Learning (DPL) method to optimize feature space distribution, making same-class features more compact and different-class features more separated. The DPL acts as a regularization technique, enhancing the model’s generalization ability and classification performance. Furthermore, the DPL can be easily integrated with existing advanced methods, serving as a general approach to enhance model performance. To the best of our knowledge, this paper is the first to apply the prototype learning to EEG-MI decoding. We believe it offers valuable insights that will inspire further advancements in the field.
- We conduct experiments on three benchmark public datasets to evaluate the superiority of the proposed SST-DPN against state-of-the-art (SOTA) MI decoding methods Additionally, comprehensive ablation experiments and visual analysis demonstrate the effectiveness and interpretability of each module in the proposed method.

## Requirements:
- Python 3.10
- Pytorch 2.12

## Rusults and Visualization
In the following datasets we have used the official criteria for dividing the training and test sets:
- [BCI_competition_IV 2a](https://www.bbci.de/competition/iv/) -acc 84.11%
- [BCI_competition_IV 2b](https://www.bbci.de/competition/iv/) -acc 86.65%
- [BCI_competition_III IVa](https://bbci.de/competition/iii/desc_IVa.html) -acc 82.03%

![image](https://github.com/hancan16/EDPNet/blob/main/figs/tsne_DPL.png)

## Acknowledgments
We are deeply grateful to Martin for providing clear and easily executable code in the [channel-attention](https://github.com/martinwimpff/channel-attention) repository. In our paper, we referenced the code and results from [channel-attention](https://github.com/martinwimpff/channel-attention) to ensure the reliability of our reproductions of the baseline methods.


## Contact
If you have any questions, please feel free to email hancan@sjtu.edu.cn.
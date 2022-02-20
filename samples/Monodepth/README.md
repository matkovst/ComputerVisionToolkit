# Monocular depth estimation
This sample contains LibTorch C++ implementation of DPT model [[1]](#1) which solves the depth problem.

## Preliminaries
In order to load the model with LibTorch, it must be first converted to .torchscript format. This [issue](https://github.com/isl-org/DPT/issues/42#issuecomment-893542411) resolves converting troubles.

## Depth interpretation
The model yields only scale and shift invariant absolute depth values which are not linked to any real depth values. To convert absolute values to real values, the model output must be re-scaled and re-shifted. More information can be found [here](https://github.com/isl-org/DPT/issues/63#issuecomment-1040260597).

## References
<a id="1">[1]</a> 
René Ranftl and Alexey Bochkovskiy and Vladlen Koltun. (2021). Vision Transformers for Dense Prediction. ArXiv preprint.

<a id="2">[2]</a> 
René Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun. (2020). Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

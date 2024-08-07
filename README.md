# Brain_Network_Modelling
用于模拟脑网络模型（Brain Network Model，BNM）的matlab代码 

matlab codes to simulate and analyze Brain Network Models

## Introduction to Brain Network Model
脑网络模型是用于构建数字孪生脑、模拟全脑大尺度神经动力学的多种技术路径中的一个。模型使用从弥散张量成像(DTI)数据中得到结构连接(Structural Connectivity, SC)，将描述神经元群体平均活动的“神经群体模型”(Neural Mass Model, NMM)耦合起来，从而模拟全脑各个脑区的神经活动。模型模拟的大脑活动可以通过 forward models 转化为模拟的功能磁共振(fMRI)或者脑电(EEG)数据，通过比较模拟数据和实测数据的相似性，可以优化模型的参数。

The Brain Network Model is one of the various approaches for constructing a digital twin brain and simulating large-scale neural dynamics of the whole brain. The model uses structural connectivity (SC) obtained from diffusion tensor imaging (DTI) data to couple the “neural mass model” (NMM), which describes the average activity of neuronal populations, thereby simulating the neural activity of each brain region of the whole brain. The brain activity simulated by the model can be transformed into simulated functional magnetic resonance imaging (fMRI) or electroencephalogram (EEG) data through forward models. By comparing the similarity between the simulated data and the empirical data, the parameters of the model can be optimized.

脑网络模型的介绍可以参见综述文章：[Dynamic models of large-scale brain activity | Nature Neuroscience](https://www.nature.com/articles/nn.4497) 和我的知乎翻译：[大尺度脑活动的动力学模型 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/496492439)

The introduction to the brain network model can be found in review article: [Dynamic models of large-scale brain activity | Nature Neuroscience](https://www.nature.com/articles/nn.4497)

## Introduction to Codes

### demo

[demo](demo) 包含dMFM、EI_dMFM和Wilson-Cowan模型的模拟案例和对EI_dMFM进行二维参数搜索的案例。[demo](demo)中使用的数据来自以下两篇文章：

The demo includes simulation cases for the dMFM, EI_dMFM, and Wilson-Cowan models, as well as cases for two-dimensional parameter search for EI_dMFM. The data used in demo comes from the following two articles: 

Kong, X., Kong, R., Orban, C. *et al.* Sensory-motor cortices shape functional connectivity dynamics in the human brain. *Nat Commun* **12**, 6373 (2021). https://doi.org/10.1038/s41467-021-26704-y

Muldoon, Sarah Feldt et al. (2017). Data from: Stimulation-based control of dynamic brain networks [Dataset]. Dryad. https://doi.org/10.5061/dryad.8g4vp

### forward_models

[forward_models](forward_models) 包含forward models的代码，用于将模拟得神经活动转化为测量数据。目前有用于生成fMRI数据的 [Balloon_Windkessel_model.m](forward_models/hemodynamics/Balloon_Windkessel_model.m) 

forward_models are used to convert simulated neural activity into measured data. Currently, there is  [Balloon_Windkessel_model.m](forward_models/hemodynamics/Balloon_Windkessel_model.m) for generating fMRI data.

### model_metrics

[model_metrics](model_metrics) 包含一些用于评价模型模拟优劣的度量，仅包含本人在工作中用到的自己写的度量函数，并不是一个完整的模型评估的代码库，用何种指标评估模型的模拟结果取决于具体研究问题的需要

### neural_mass_models

[neural_mass_models](neural_mass_models) 包含多种类型的模型：

- dMFM: refer to [Resting-State Functional Connectivity Emerges from Structurally and Dynamically Shaped Slow Linear Fluctuations | Journal of Neuroscience (jneurosci.org)](https://www.jneurosci.org/content/33/27/11239)
- EI_dMFM: refer to [How Local Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics | Journal of Neuroscience (jneurosci.org)](https://www.jneurosci.org/content/34/23/7886.short)
- SAR: refer to [Predicting functional connectivity from structural connectivity via computational models using MRI: An extensive comparison study - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1053811915000932)
- Wilson-Cowan: refer to [Stimulation-Based Control of Dynamic Brain Networks | PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005076)

### other_methods


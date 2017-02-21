# Mixture Density Networks implementation for distribution and uncertainty estimation
A generic Mixture Density Networks implementation for distribution and uncertainty estimation by using Keras (TensorFlow)

This repository is a collection of [Jupyter](https://jupyter.org/) notebooks intended to solve a lot of problems in which we want to predict a probability distribution by using Mixture Density Network avoiding a NaN problem and other derived problems of the model proposed by [Bishop, C. M. (1994)](http://eprints.aston.ac.uk/373/). The second major objective of this repository is to look for ways to predict uncertainty by using artificial neural networks.

The whole code, until 20.1.2017, is the result of a final Master's Thesis of the [Master's Degree in Artificial Intelligence](http://www.upc.edu/master/fitxa_master.php?id_estudi=50&lang=esp) supervised by Jordi Vitrià, PhD. The [Master's Thesis report](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/ABrando-MDN-MasterThesis.pdf) is published in this repository in a PDF format but my idea is to realize a web view of the final master's work in the coming days. To summary all the contents I explained in the report, it is possible to consult the [slides of the presentation](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/ABrando-MDN-Slides.pdf). Any contribution or idea to continue the lines of the proposed work will be very welcome.

<p align="center"><img src="https://cdn.rawgit.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/cd4d8e9c/svgs/f442dfcf42c5ca5d6c9b96753cde8768.svg" align=middle width=645.87435pt height=348.58725pt/>
</p>
<p align="center">
<em>Representation of the Mixture Density Network model. The output of the feed-forward neural network determine the parameters in a mixture density model. Therefore, the mixture density model represents the conditional probability density function of the target variables conditioned on the input vector of the neural network.</em>
</p>

## Implemented tricks and techniques

> - Log-sum-exp trick.
> - ELU+1 representation function for variance scale parameter proposed by us in the Master's Thesis that I will link when it is published.
> - Clipping of the mixing coefficient parameter value.
> - Mean log Gaussian likelihood proposed by [Bishop](http://eprints.aston.ac.uk/373/).
> - Mean log Laplace likelihood proposed by us in the Master's Thesis that I will link when it is published.
> - Fast Gradient Sign Method to produce Adversarial Training proposed [by Goodfellow et al](https://arxiv.org/abs/1412.6572).
> - Modified version of Adversarial Training proposed by [Nokland](https://arxiv.org/abs/1510.04189).
> - Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles implementation proposed by [Lakshminarayanan et a](https://arxiv.org/abs/1612.01474).

## Some Keras algorithms used

> - RMSProp optimisation algorithm.
> - Adam optimisation algorithm.
> - Gradient Clipping
> - Batch normalisation

## Implemented visualisation functionalities

> - Generic implementation to visualise mean and variance (as errorbar) of the distribution with maximum mixing coefficient  of of the MDN.
> - Generic implementation to visualise mean and variance (as errorbar) of all the distributions of of the MDN.
> - Generic implementation to visualise all the probability density function as a *heat graphic* for 2D problems.
> - Generic implementation to visualise the original 3D surface and visualise the mean of the distribution of the mixture through a sampling process.
> - Adversarial data set visualisation proposed by us in the Master's Thesis that I will link when it is published.



## Notebooks
(Currently tested on Keras (1.1.0) and TensorFlow (0.11.0rc2)

#### [Introduction to MDN models and generic implementation of MDN](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-Introduction.ipynb)

#### [MDN applied to a 2D regression problem](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-2D-Regression.ipynb)

#### [MDN applied to a 3D regression problem](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-3D-Regression.ipynb)

#### [MDN with LSTM neural network for a time series regression problem](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-LSTM-Regression.ipynb) 

#### [MDN with completely dense neural network for a time series regression problem by using Adversarial Training](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Regression.ipynb) 

#### [Ensemble of MDNs with completely dense neural network for a simple regression problem for Predictive Uncertainty Estimation](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Simple-Ensemble-Uncertainty.ipynb) 

#### [Ensemble of MDNs with completely dense neural network for a complex regression problem for Predictive Uncertainty Estimation and Adversarial Data set test](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Complex-Ensemble-Uncertainty.ipynb) 




## Contributions

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/issues).

## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [axelbrando](https://github.com/axelbrando)
* Website: [axelbrando.github.io](http://axelbrando.github.io)

### BibTex reference format for citation for the Code
```
@misc{MDNABrando,
title={Mixture Density Networks (MDN) for distribution and uncertainty estimation},
url={https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/},
note={GitHub repository with a collection of Jupyter notebooks intended to solve a lot of problems related to MDN.},
author={Axel Brando},
  year={2017}
}
```
### BibTex reference format for citation for the report of the Master's Thesis

```
@misc{MDNABrandoMasterThesis,
title={Mixture Density Networks (MDN) for distribution and uncertainty estimation},
url={https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/ABrando-MDN-MasterThesis.pdf},
note={Report of the Master's Thesis: Mixture Density Networks for distribution and uncertainty estimation.},
author={Axel Brando},
  year={2017}
}
```

## License

The content developed by Axel Brando is distributed under the following license:

    Copyright 2016 Axel Brando

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


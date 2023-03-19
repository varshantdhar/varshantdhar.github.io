---
title:  "Intro: Self-supervised Learning"
date:   2023-03-19
mathjax: true
categories:
    - blog
tags: 
    - self_supervised
    - joint_embedding
    - siamese
    - energy_based_models
    - contrastive
---

Self-supervised learning obtains supervisory signals from the data itself, often leveraging the underlying structure in the data. The general technique of self-supervised learning is to predict any unobserved or hidden part (or property) of the input from any observed or unhidden part of the input. Since self-supervised learning uses the structure of the data itself, it can make use of a variety of supervisory signals across co-occurring modalities (e.g., video and audio) and across large data sets — all without relying on labels.

### Self-supervised Learning in NLP

Models are pretrained in a self-supervised phase and then fine-tuned for a particular task, such as classifying the topic of a text. In the self-supervised pretraining phase, the system is shown a short text (typically 1,000 words) in which some of the words have been masked or replaced. The system is trained to predict the words that were masked or replaced. In doing so, the system learns to represent the meaning of the text so that it can do a good job at filling in “correct” words, or those that make sense in the context.

In NLP, predicting the missing words involves computing a prediction score for every possible word in the vocabulary. While the vocabulary itself is large and predicting a missing word involves some uncertainty, it’s possible to produce a list of all the possible words in the vocabulary together with a probability estimate of the words’ appearance at that location.

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/self-supervised-intro/self-supervised.png)

### Unified view of self-supervised methods

There is a way to think about SSL within the unified framework of an energy-based model (EBM). An EBM is a trainable system that, given two inputs, $x$ and y, tells us how incompatible they are with each other. To indicate the incompatibility between $x$ and y, the machine produces a single number, called an energy. If the energy is low, $x$ and $y$ are deemed compatible; if it is high, they are deemed incompatible.

Training an EBM consists of two parts: (1) showing it examples of $x$ and $y$ that are compatible and training it to produce a low energy, and (2) finding a way to ensure that for a particular $x$, the $y$ values that are incompatible with $x$ produce a higher energy than the $y$ values that are compatible with $x$. 

### Joint embedding, Siamese networks

A joint embedding architecture is composed of two identical (or almost identical) copies of the same network. One network is fed with $x$ and the other with $y$. The networks produce output vectors called embeddings, which represent $x$ and $y$. A third module, joining the networks at the head, computes the energy as the distance between the two embedding vectors. When the model is shown distorted versions of the same image, the parameters of the networks can easily be adjusted so that their outputs move closer together. This will ensure that the network will produce nearly identical representations (or embedding) of an object, regardless of the particular view of that object.

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/self-supervised-intro/siamese-networks.jpeg)

The difficulty is to make sure that the networks produce high energy, i.e. different embedding vectors, when $x$ and $y$ are different images. Without a specific way to do so, the two networks could happily ignore their inputs and always produce identical output embeddings. This phenomenon is called a collapse. When a collapse occurs, the energy is not higher for nonmatching $x$ and $y$ than it is for matching $x$ and $y$.

There are two categories of techniques to avoid collapse: $\textit{contrastive}$ methods and $\textit{regularization}$ methods.

### Contrastive energy-based SSL

Contrastive methods are based on the simple idea of constructing pairs of $x$ and $y$ that are not compatible, and adjusting the parameters of the model so that the corresponding output energy is large.

One starts for a complete segment of text $y$, then corrupts it, e.g., by masking some words to produce the observation $x$. The corrupted input is fed to a large neural network that is trained to reproduce the original text $y$. An uncorrupted text will be reconstructed as itself (low reconstruction error), while a corrupted text will be reconstructed as an uncorrupted version of itself (large reconstruction error). If one interprets the reconstruction error as an energy, it will have the desired property: low energy for “clean” text and higher energy for “corrupted” text.

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/self-supervised-intro/unified-view.png)

A predictive architecture of this type can produce only a single prediction for a given input. Since the model must be able to predict multiple possible outcomes, the prediction is not a single set of words but a series of scores for every word in the vocabulary for each missing word location. However, we cannot enumerate all possible images so we must try other avenues for CV -- like latent-variable predictive architectures. 

Latent-variable predictive models contain an extra input variable ($z$). With a properly trained model, as the latent variable varies over a given set, the output prediction varies over the set of plausible predictions compatible with the input $x$. 

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/self-supervised-intro/latent-variable.jpeg)


These models can be trained with contrastive methods. For e.g. the generative adversarial network has a generator and discriminator network where the generator network is trained to produce contrastive samples to which the critic is trained to associate high energy. However, this is very inefficient to train and finding a set of contrastive images that cover all the ways they can differ from a given image is a nearly impossible task.

### Non-contrastive energy-based SSL

Non-contrastive methods for joint-embedding use various tricks, such as computing virtual target embeddings for groups of similar images ([DeeperCluster](https://openaccess.thecvf.com/content_ICCV_2019/html/Caron_Unsupervised_Pre-Training_of_Image_Features_on_Non-Curated_Data_ICCV_2019_paper.html), [SwAV](https://arxiv.org/pdf/2006.09882.pdf), [SimSiam](https://arxiv.org/pdf/2011.10566.pdf)) or making the two joint embedding architectures slightly different through the architecture or the parameter vector ([BYOL](https://arxiv.org/pdf/2006.07733.pdf), [MoCo](https://arxiv.org/pdf/2003.04297.pdf)). 


Perhaps a better alternative in the long run will be to devise non-contrastive methods with latent-variable predictive models. The main obstacle is that they require a way to minimize the capacity of the latent variable. The volume of the set over which the latent variable can vary limits the volume of outputs that take low energy. By minimizing this volume, one automatically shapes the energy in the right way.













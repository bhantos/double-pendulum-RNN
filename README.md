# Solving differential equations: double pendulum and deep learning

## Introduction
The aim of my project is to investigate how neural networks perform at solving ordinary differential 
equations. Currently, machine learning solutions are trending in the IT industry. From image and 
sound  processing methods through recommender  systems to  time  series analysis it is getting 
widespread to use neural networks in fields where large amounts of data is available.

## Motivation
The  motivation  behind  this  project  stems  from  the  strong  intertwining  between  differential 
equation solving and the nature of neural networks. It is very important to mention, that there is 
a lot of related work in this area. Notably, recently solvers were used to suggest some new neural 
network architectures [1,2] as well as new training methods [3]. 

The usage of neural networks to tackle physical problems regarding ODE solving isn’t very new. 
My personal motivation of choosing this project conceived upon reading an article about using the 
MLP-model  to  solve  the  chaotic  three-body  problem  [4].  Firstly,  I  thought  that  the  model 
presented in the article was poorly developed, due to the fact that the MLP-model can’t take 
successivity into account. 

On the other hand, recurrent neural networks, as a general tool of time series analysis, can be of 
a good use, to tackle the notion of events being sequential. I propose a model in this project, which 
might be capable of solving the equation of motion for the chaotic double pendulum.

<b>See the report for my results. You may have to download it, as Github doesn't really like pdf format.</b>

[1]  Haber, Eldad, and Lars Ruthotto. "Stable architectures for deep neural networks." Inverse 
Problems 34.1 (2017): 014004. 

[2]  Chen,  Ricky  TQ,  et  al.  "Neural  ordinary  differential  equations."  Advances  in  neural 
information processing systems. 2018. 

[3]  Bo  Chang,  Lili  Meng,  Eldad  Haber,  Lars  Ruthotto,  David  Begert,  and  Elliot  Holtham. 
Reversible  architectures  for  arbitrarily  deep  residual  neural  networks.  arXiv  preprint 
arXiv:1709.03698, 2017.

[4]  Breen at al.: Newton vs the machine: solving the chaotic three-body problem using deep neural 
networks 

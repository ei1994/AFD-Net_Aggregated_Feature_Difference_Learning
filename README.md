## AFD-Net_Aggregated_Feature_Difference_Learning
**（1）paper：Element-wise Feature Relation Learning Network for Cross-Spectral Image Patch Matching（IEEE Transactions on Neural Networks and Learning Systems  2021已发表（2.05））**  [中科院一区期刊论文地址](https://ieeexplore.ieee.org/abstract/document/9349201)  
**（2）该期刊论文的核心创新点是本人研究生毕业论文《*基于特征残差学习和图像转换的异源图像块匹配方法研究*》的第四章内容。**  [研究生毕业论文地址](http://kns8.zh.eastview.com/KCMS/detail/detail.aspx?filename=1020004804.nh&dbname=CMFD202001&dbcode=cdmd&uid=&v=MzE3NjBmbnM0eVJZYW16MTFQSGJrcVdBMEZyQ1VSN2lmWXVSdUZpamdWcnpQVkYyNUhyTzRHdG5NcTVFYlBJUis=)    
**（3）该创新点在2019年被ICCV（计算机视觉方向的三大顶级会议之一）录用为口头报告(Oral)论文，而ICCV 2019的Oral录用比例仅为4.62%。**   [ICCV录用论文地址](https://openaccess.thecvf.com/content_ICCV_2019/html/Quan_AFD-Net_Aggregated_Feature_Difference_Learning_for_Cross-Spectral_Image_Patch_Matching_ICCV_2019_paper.html)  

## 论文创新点：  
  （1）基于特征图差值融合的异源图像块匹配方法。针对异源图像提取的特征信息表达能力不足的问题，通过特征图融合的方式将**空间邻域信息引入到图像特征描述子中**，充分提取利用图像的有效信息。为解决特征描述子中存在的鉴别性差异信息容易被忽视的问题，本文在特征图融合方法的基础上进行改进，利用特征图组求差值的方式得到残差特征图，之后进行匹配度量。该策略不仅提高了网络对重要匹配特征的关注力度，而且增加了网络自身的稀疏性，提高了匹配准确率和网络健壮性。  
  （2）基于特征残差学习的异源图像块匹配方法。通过对基于特征图差值融合方法的分析发现，差异特征信息在图像匹配中十分重要。针对图像的高层残差特征中会缺失重要细节信息的问题，通过增加副分支卷积网络的方式，对主干网络中不同卷积层得到的多个尺度残差特征图组进行特征学习。该策略将**底层的细节残差特征与高层的语义残差特征进行结合，充分挖掘图像中有利于匹配的重要特征信息**。在此基础上，本文进一步将主干网络与副分支网络得到的高层残差特征进行组合学习，利用三次度量的方式来提高匹配预测的准确率和泛化性能。  
  （3）基于图像转换的异源图像块匹配方法。针对异源图像之间存在的表征差异性问题，利用生成对抗网络中的**生成器将异源图像转换为同源图像，之后对同源图像进行相似性度量**，以此降低匹配难度。为改善生成的转换图像质量及多样性，通过预训练的**卷积自编码器来正则约束判别器网络的训练**，从而间接强化生成器的稳定训练。图像转换策略克服了图像之间由于表征差异性大导致的匹配困难问题，为异源图像匹配提供了全新的解决思路。

## 概述：  
  **1、** *背景：* 
  图像匹配一直是计算机视觉领域中的研究重点，在图像检索、图像配准、目标跟踪等诸多领域有着广泛的应用。异源图像包含的丰富互补信息，对于目标的全面、准确认识十分重要。但是**由于不同的成像机理和条件导致异源图像之间存在明显的表征差异性，使得同源图像的匹配方法不能适用于异源图像**，因此针对异源图像的匹配方法研究具有非常重要的意义。随着深度学习的飞速发展，基于卷积神经网络的图像匹配方法不断涌现出来。论文在总结现有图像匹配方法的基础上，将**深度卷积网络和生成对抗网络**应用于异源图像的匹配中。  
    
  **2、**  *主要工作总结：*   
  （1）基于特征图差值融合的异源图像块匹配方法。基于特征图融合的方法是在特征提取阶段，通过特征图融合的方式有效保留了图像对的空间特征信息，显著提高了匹配预测的准确率。之后，在基于特征图融合方法的基础上，利用图像对的残差特征图作为匹配度量网络的输入数据，并以此判断图像对的匹配程度，提出了基于特征图差值融合的方法。该方法最大的创新点是，使用的**特征图差值融合策略不仅保留了图像的空间特征信息，而且提高了匹配网络对重要差异特征的关注力度**。最后，通过多组对比实验证明了，改进方法在增加网络自身稀疏性的同时，提高了匹配的准确率和网络的泛化性。  
  （2）基于特征残差学习的异源图像块匹配方法。该创新点是从统计学的梯度提升树和深度学习的残差网络中获得灵感，将**残差学习**的思想引入到之前提出的基于特征图差值融合的异源图像匹配方法中，通过**增加一路副分支网络来专门学习主干网络中不同卷积层得到的多个尺度残差特征图的信息**，目的是将底层的细节残差特征与高层的语义残差特征进行结合，重点关注图像中具有鉴别性的差异特征信息，从而提出了基于特征残差学习的异源图像匹配方法。另外，论文在改进方法的基础上，将主干网络提取的残差特征信息与副分支网络经过多层学习得到的特征图残差信息进行融合学习，提出了**基于三次度量的匹配方法**，进一步提升了异源图像匹配的准确率。  
  （3）基于图像转换的异源图像块匹配方法。从不同角度出发来思考解决异源图像的匹配问题，**利用生成器将表征差异较大的异源图像转换为同源图像后进行匹配度量，图像转换思想是最大的一个创新点**。该方法利用生成对抗网络中的生成器将异源图像转换为同源图像，之后对同源图像进行匹配预测，提出了基于图像转换的匹配方法。该方法通过图像转换策略降低了匹配难度，为异源图像匹配问题提供了全新的解决思路。由于生成的转换图像质量及多样性将直接影响之后匹配网络的正确判断，因此本章通过预训练的卷积自编码器来正则约束判别器网络的训练，从而间接强化生成器的稳定训练，提出了**基于正则生成对抗网络的匹配方法**，最后通过对比实验验证了图像转换策略以及正则约束策略在异源图像匹配上的有效性。  
    
  **3、**  *主要展望：*   
  本文主要是研究基于卷积神经网络和生成对抗网络的异源图像匹配方法，在特征提取、匹配度量和图像块转换等方面进行改进，提出了创新的解决方案，并取得了一些研究成果。但是，仍然存在很多需要改进和提升的方面。  
  （1）本文的改进方法可以显著提高异源图像匹配的准确率，然而在网络的复杂度和计算量方面没有明显的改善。因此可以考虑在保证匹配准确率的前提下，简化网络结构，提升异源图像匹配预测的效率。  
  （2）本文的创新方法在不同测试集上的表现不尽相同，这主要是由测试集与训练集数据分布的一致性程度决定的。因此，如何提高匹配方法的泛化能力，在多种测试集上都取得较好的匹配效果是接下来需要研究的重要问题。  
  （3）本文初步验证了图像转换策略在异源图像匹配上的有效性。但由于转换图像的质量直接影响之后图像匹配的正确判断，因此如何设计良好的生成对抗网络结构，优化生成器的综合性能来提高生成的转换图像质量及多样性，是未来需要重点研究和解决的问题。

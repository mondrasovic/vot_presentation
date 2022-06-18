---
marp: true
paginate: true
theme: base
size: 16:9
title: Visual Object Tracking + Lessons Learned
footer: '***Siamese-based Visual Object Tracking***'
---

<!--
_header: ''
_footer: ''
_paginate: false
_class: invert
-->
# Visual Object Tracking
## ... with focus on **Siamese neural networks** 
## ... and **lessons learned** during my Ph.D. research

### ***Milan Ondrašovič***
*milan.ondrasovic@gmail.com*
***University of Žilina**, Žilina, Slovakia*
*Faculty of Management Science and Informatics*
*Department of Mathematical Methods and Operations Research*
\
Date of presentation: 22. 06. 2022

![bg contain opacity:.6](./images/title_background.png)

---
<!--
_header: ''
_footer: ''
_paginate: false
_class: invert
-->

# *Part 1*: **Similarity learning** and **Siamese networks**

---
<!--
header: '*Part 1*: Introduction to **Similarity learning** and **Siamese neural networks**'
_footer: '**[1]** - *Bromley, Jane, et al.* "[Signature verification using a 
Siamese time delay neural network.](https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html)" Advances in neural information processing systems (1993).'
-->

# Similarity learning

* **Similarity learning** started as **signature verification** [[1](https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html)].
* The **goal** is to **train** the model to discern between **similar** and **dissimilar** object.
* The notion of **similarity** is **task-specific** as well as the way to **measure** it.

![](./images/similarity_learning.jpg)

---
<!--
_footer: '**[1]** - *Zagoruyko, Sergey, and Nikos Komodakis.* "[Deep compare: A study on using convolutional neural networks to compare image patches.](https://www.sciencedirect.com/science/article/pii/S1077314217301716?casa_token=vC_LQ_Cqk0gAAAAA:gocpeGZRSaJH7HTFVdyN-GVjAPJo46R4ZCcYANYlhO4j0yyofFHcSnHuzTTJJRj8axmECChxwQ)" Computer Vision and Image Understanding 164 (2017): 38-55.
'
-->
# Siamese networks

* In general, employed for **information retrieval** tasks.
* Frequently used approach to **similarity learning**.
* They often come in a form of a "**Y-shaped**" network.
* There are **various types** from the **weight sharing** perspective [[1](https://www.sciencedirect.com/science/article/pii/S1077314217301716?casa_token=vC_LQ_Cqk0gAAAAA:gocpeGZRSaJH7HTFVdyN-GVjAPJo46R4ZCcYANYlhO4j0yyofFHcSnHuzTTJJRj8axmECChxwQ)].

![bg right:53% fit](./images/siamese_network_types.jpg)

---
<!--
_footer: '
**[1]** - *Kuma, Ratnesh, et al.* "[Vehicle re-identification: an 
efficient baseline using triplet embedding.](https://arxiv.org/abs/1901.01015)" 2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 2019.

**[2]** - *Wei, Wenyu, et al.* "[Person re-identification based on deep learning-An overview.](https://www.sciencedirect.com/science/article/pii/S1047320321002765)" Journal of Visual Communication and Image Representation (2021): 103418.

**[3]** - *Bertinetto, Luca, et al.* "[Fully-convolutional siamese networks for object tracking.](https://arxiv.org/abs/1606.09549)" European conference on computer vision. Springer, Cham, 2016.
'
-->
# Siamese networks and similarity learning

* A common use-case is to **estimate** a **degree** of **similarity** between **two images**.
* Domain of object **Re-identification** (**ReID**).
  * E.g., vehicle ReID [[1](https://arxiv.org/abs/1901.01015)], person ReID [[2](https://www.sciencedirect.com/science/article/pii/S1047320321002765)].
* Domain of **Siamese object tracking** [[3](https://arxiv.org/abs/1606.09549)].


![bg right:39% fit](./images/similarity_score_computation.jpg)

---

# Computing embedding vector similarity

Let $\mathbf{u}, \mathbf{v} \in \mathbb{R}^D$ be two arbitrary $D$-dimensional **vectors**.

## Cosine similarity

$$\cos \angle \left( \mathbf{u}, \mathbf{v} \right) = \cos \left( \theta \right) = \frac{\mathbf{u} \cdot \mathbf{v}}{\left\Vert \mathbf{u} \right\Vert_2 \left\Vert \mathbf{v} \right\Vert_2}$$

## (Squared) Euclidean distance

It's **squared form** is **proportional** to **cosine similarity** if both vectors are **normalized**.

$$\left\Vert \mathbf{u} - \mathbf{v} \right\Vert^2_2 = \sum_{i = 1}^{D} {\left( \mathbf{u}_i - \mathbf{v}_i \right)}^2 = 2 - 2 \cos \angle \left( \mathbf{u}, \mathbf{v} \right)$$ 

---

# Contrastive loss function

![height:400](./images/siamese_architecture.jpg)

---
<!--
_footer: '**[1]** - *Hadsell, Raia, Sumit Chopra, and Yann LeCun*. "[Dimensionality reduction by learning an invariant mapping.](https://ieeexplore.ieee.org/abstract/document/1640964?casa_token=s5-7Pr7LQAUAAAAA:MHfP_GruBRcXb37YixRLuvY4KkDGzZnVksgvBJBlHXsl2gtOOxqYkyHPl4lSuV4b9Iyjb6n2Ec2n)" 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR-06). Vol. 2. IEEE, 2006.' 
-->

# Contrastive loss function

* Consider a sample $\left( x_0, x_1, y \right)$, where $x_0$, $x_1$ represent **features**, and $y$ is the **label**.

$$y = \begin{cases}
  1, \quad \text{if } x_0 \text{ and } x_1 \text{belong to the same category},\\
  0, \quad \text{otherwise}.
\end{cases}$$

* Let $\tilde{x}_0 = f_\theta \left( x_0 \right)$ and $\tilde{x}_1 = f_\theta \left( x_1 \right)$ be the **embedding vectors**.
* Let $\alpha$ be the **margin** of separation and $D \left( \cdot \right)$ be the **distance function**.
* The **contrastive loss function** [[1](https://ieeexplore.ieee.org/abstract/document/1640964?casa_token=s5-7Pr7LQAUAAAAA:MHfP_GruBRcXb37YixRLuvY4KkDGzZnVksgvBJBlHXsl2gtOOxqYkyHPl4lSuV4b9Iyjb6n2Ec2n)] is defined as

$$\mathcal{L}_{contr} =
\frac{1}{2}
  D^2 \left( \tilde{x}_0, \tilde{x}_1 \right) +
\frac{1}{2}
  \left( 1 - y \right)
  {\left(
    \left[
      \alpha - D \left(
        \tilde{x}_0, \tilde{x}_1
      \right)
    \right]_+
  \right)}^2.$$

---

# Triplet loss function

![height:500](./images/triplet_architecture.png)

---
<!--
_footer: '**[1]** - *Schroff, Florian, Dmitry Kalenichenko, and James Philbin.* "[Facenet: A unified embedding for face recognition and clustering.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)" Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.'
-->

# Triplet loss function

* Consider a triplet $\left( x_a, x_p, x_n \right)$ of **distinct samples**, where $x_a$, $x_p$ and $x_n$ represent the **anchor**, **positive** and **negative** sample, respectively. Let $y \left( \cdot \right)$ be the **label**.
  * Assume that $x_a \neq x_p \land y \left( x_a \right) = y \left( x_p \right)$ and $x_a \neq x_n \land y \left( x_a \right) \neq y \left( x_n \right)$.
* Let $\tilde{x}_a = f_\theta \left( x_a \right)$, $\tilde{x}_p = f_\theta \left( x_p \right)$, and $\tilde{x}_n = f_\theta \left( x_n \right)$ be the **embedding vectors**.
* Let $\alpha$ be the **margin** of separation and $D \left( \cdot \right)$ be the **distance function**.
* The **goal** is to satisfy the **relationship**
$$D \left( \tilde{x}_a, \tilde{x}_p \right) +
\alpha <
D \left( \tilde{x}_a, \tilde{x}_n \right).$$
* Therefore, the **triplet loss function** [[1](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)] can be formulated as
$$\mathcal{L}_{triplet} =
{\left[
  \alpha + D \left( \tilde{x}_a, \tilde{x}_p \right) - D \left( \tilde{x}_a, \tilde{x}_n \right)
\right]}_+.$$

---
<!--
_footer: '**[1]** - *Wu, Chao-Yuan, et al.* "[Sampling matters in deep embedding learning.](https://openaccess.thecvf.com/content_iccv_2017/html/Wu_Sampling_Matters_in_ICCV_2017_paper.html)" Proceedings of the IEEE international conference on computer vision. 2017.'
-->

# Embedding training - online triplet mining

* **Superior** to **offline** approaches.
* The way that pairs or triplets are selected **significantly** influences the training [[1](https://openaccess.thecvf.com/content_iccv_2017/html/Wu_Sampling_Matters_in_ICCV_2017_paper.html)].
* The number of possible triplets grows **cubically**, rendering the use of all of them **impractical**. The majority of those triplets would be so-called **easy triplets**.

![bg right:50% fit vertical](./images/sample_data_points.png)
![bg right:50% fit](./images/triplet_architecture_online_mining.png)

---

# Different types of triplets


## Easy triplet

$D \left( \tilde{x}_a, \tilde{x}_p \right) + \alpha < D \left( \tilde{x}_a, \tilde{x}_n \right)$

## Semi-hard triplet
$D \left( \tilde{x}_a, \tilde{x}_p \right) < D \left( \tilde{x}_a, \tilde{x}_n \right) < D \left( \tilde{x}_a, \tilde{x}_p \right) + \alpha$

## Hard triplet
$D \left( \tilde{x}_a, \tilde{x}_n \right) < D \left( \tilde{x}_a, \tilde{x}_p \right)$ 


![bg right:39% fit](./images/triplet_negatives_categories.png)

---
<!--
_footer: '$\star$ - See **appendix** for more **mathematical details**.'
-->

# Online triplet mining strategies$^{\star}$
Let $P$ be the **number** of **different objects** (e.g., people, vehicles) and $K$ be the **number** of different images of a **concrete identity** (e.g., different views of the same vehicle).

## "Batch-all" strategy

Selects **all** $PK \left( K - 1 \right) \left( PK - K \right)$ **valid triplets** and **averages** the loss **only** on the **hard** and **semi-hard triplets**.
## "Batch-hard" strategy

Selects the **hardest positive** and **hardest negative** for each **anchor**, thus $PK$ triplets.

---
<!--
_backgroundColor: #a8ddb5
-->

# Lesson learned: training triplet loss

## Facts

* Triplet loss is **notoriously difficult** to train.

## Remedies

* Properly "babysitting" the learning process is essential:
  1. Apply **batch-all** mining first, then slowly progress towards **batch-hard**.

---
<!--
_header: ''
_footer: ''
_paginate: false
_class: invert
-->

# *Part 2*: Siamese-based **Visual Object Tracking**

---
<!--
header: '*Part 2:* Fundamentals of **Siamese-based Visual Object Tracking**'
_footer: '**[1]** - *Kristan, Matej, et al.* "[A novel performance evaluation methodology for single-target trackers.](https://arxiv.org/abs/1503.01313)" IEEE transactions on pattern analysis and machine intelligence 38.11 (2016): 2137-2155.'
-->

# Visual object tracking in general

Three basic phases of **Visual Object Tracking** (**VOT**) [[1](https://arxiv.org/abs/1503.01313)].
  1. **Detection** of the object of interest.
  2. **Assignment** of a unique identifier (**ID**).
  3. Correctly **propagate** the chosen ID in future frames.

![bg right:50% fit](./images/object_tracking_pedestrians.png)

---
# Frequently occurring problems

* **Changes** in object **position** and its **shape**.
* **Variations** in **lightning** conditions.
* Object **occlusion** of varying intensity.
* Presence of **distractors** (similar-looking objects), a.k.a, **similar interference**.

![bg right:45% vertical fit](./images/object_occlusion_01.png)
![bg fit](./images/object_occlusion_02.jpg)

---
# Siamese single object tracking - general pipeline

![](./images/fully_cnn_siam_tracking_architecture.png)

---
<!--
_footer: '**[1]** - *Bertinetto, Luca, et al.* "[Fully-convolutional siamese networks for object tracking.](https://arxiv.org/abs/1606.09549)" European conference on computer vision. Springer, Cham, 2016.'
-->
# Siamese tracking - fundamental components

## Symmetry

* Let $z$ and $x$ denote the **exemplar** and **search region**, respectively. 
* **Siamese networks** apply **identical transformation** $\varphi \left( \cdot \right)$ to **both inputs** $z$ and $x$ and then **combine** the produced features using a function $g \left( \cdot \right)$, such that [[1](https://arxiv.org/abs/1606.09549)]

$$f \left( z, x \right) =
g \left( \varphi \left( z \right) , \varphi \left( x \right) \right).$$

## Fully-convolutional architecture

* It provides a way to compute a **similarity score** for each **translated** sub-window of the **exemplar image** within a larger **search region**.

---
<!--
_footer: '**[1]** - *Bertinetto, Luca, et al.* "[Fully-convolutional siamese networks for object tracking.](https://arxiv.org/abs/1606.09549)" European conference on computer vision. Springer, Cham, 2016.'
-->
# Siamese tracking - fundamental components

## Cross-correlation operation

* By exploiting the **embedding** function $\varphi \left( \cdot \right)$, the produced **feature maps** can be combined using a **cross-correlation** layer as [[1](https://arxiv.org/abs/1606.09549)]
$$f \left( z, x \right) =
\varphi \left( z \right) \star \varphi \left( z \right) + b \mathbf{1}.$$
* The term $b \mathbf{1}$ represents a **bias** (offset) which has value $b \in \mathbb{R}$ in **every location**.
* The **cross-correlation** is denoted by $\star$.
* The **output** is a **2D score map** defined on a finite **grid** of positions $\mathcal{P} \subset {\mathbb{Z}}^2_{+}$.

---
<!--
_footer: '**[1]** - *Shuai, Bing, et al.* "[Siammot: Siamese multi-object tracking.](https://arxiv.org/abs/2105.11595)" Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

**[2]** - *Ren, Shaoqing, et al.* "[Faster r-cnn: Towards real-time object detection with region proposal networks.](https://arxiv.org/abs/1506.01497)" Advances in neural information processing systems 28 (2015).'
-->

# Siamese multiple-object tracking

* **Siamese Multi-Object Tracker** (**SiamMOT**) [[1](https://arxiv.org/abs/2105.11595)].
* Back in the end of $2021$ it was the **state-of-the-art** approach.
* Ability to track **multiple objects simultaneously**.
* **Object detection** by use of **Faster R-CNN** [[2](https://arxiv.org/abs/1506.01497)].
* **Siamese tracking** is exploited for **motion modeling**.
* **Reasoning** on top of both **detector** as well as **tracker predictions**.

---
<!--
_footer: '**[1]** - *Ren, Shaoqing, et al.* "[Faster r-cnn: Towards real-time object detection with region proposal networks.](https://arxiv.org/abs/1506.01497)" Advances in neural information processing systems 28 (2015).'
-->
# Two-stage object detection

* A well-known example of a **two-stage object detector** is **Faster R-CNN** [[1](https://arxiv.org/abs/1506.01497)].

![](./images/fastercnn_diagram.png)

---

# SiamMOT architecture

![](./images/siammot_architecture.png)

---
# Utilization of re-identification

---
<!--
_backgroundColor: #a8ddb5
-->
# Lesson learned: combining ReID with Siamese tracker

---

# Utilization of feature embeddings

Inclusion of an **embedding head** into the original **end-to-end** pipeline.

![](./images/siammot_feature_emb_training.jpg)

---
<!--
_backgroundColor: #a8ddb5
-->
# Lesson learned: combining emeddings with RPN

---
<!--
_footer: '**[1]** - *Dai, Jifeng, et al.* "[Deformable convolutional networks.](https://arxiv.org/abs/1703.06211)" Proceedings of the IEEE international conference on computer vision. 2017.

**[2]** - *Zhu, Xizhou, et al.* "[Deformable convnets v2: More deformable, better results.](https://arxiv.org/abs/1811.11168)" Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

'
-->

# Deformable convolution operation

* Useful for **dense prediction** tasks (object **detection** or **segmentation**) [[1](https://arxiv.org/abs/1703.06211)].
* A more advanced version is **modulated deformable convolution** [[2](https://arxiv.org/abs/1811.11168)].

![bg right:50% fit](./images/dcn_standard_vs_deformable.png)

---
# Deformable Siamese attention

![bg right:50%](./images/dsa_attention_visualization.png)

---
# Utilization of deformable Siamese attention

![](./images/siammot_attention_training.jpg)

---

# Conclusion

---
<!--
_header: ''
_footer: ''
_paginate: false
_class: invert
-->

# *Part 3*: **Appendix** - Relevant **Vehicle Tracking Datasets**

---
<!--
header: '*Part 3*: Appendix - Overview of Relevant **Vehicle Tracking Datasets**'
_footer: '**[1]** - *Wen, Longyin, et al.* "[UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking.](https://www.sciencedirect.com/science/article/pii/S1077314220300035?casa_token=77iDof3yBUoAAAAA:y4W5dlEYjkX_so6sMFOH2eF-eZeq5ZxK2vUEqbvstj_4R3GXXYr4UGh7bOSQWzeS4_C5yGaxTg)" Computer Vision and Image Understanding 193 (2020): 102907.'
-->
# UA-DETRAC vehicle tracking dataset

* This dataset is **available** only **upon request** [[1](https://www.sciencedirect.com/science/article/pii/S1077314220300035?casa_token=77iDof3yBUoAAAAA:y4W5dlEYjkX_so6sMFOH2eF-eZeq5ZxK2vUEqbvstj_4R3GXXYr4UGh7bOSQWzeS4_C5yGaxTg)].
* $10$ hours of video captured in $24$ different places.
* $25$ FPS, $960 \times 540$ pixels, more than $140\ 000$ frames.
* $8\ 250$ manually annotated vehicles, $1,21$ million bounding boxes.

![height:280](./images/uadetrac_tracking_01.png) ![height:280](./images/uadetrac_tracking_02.png)

---

# UA-DETRAC vehicle tracking dataset

![](./images/uadetrac_samples.jpg)

---
<!--
_footer: '**[1]** - *Liu, Xinchen, et al.* "[Provid: Progressive and multimodal vehicle reidentification for large-scale urban surveillance.](https://ieeexplore.ieee.org/abstract/document/8036238?casa_token=iIk6fqQhbykAAAAA:Ypehhy5mumMJrOcVcDur_MncD7xAFdLThK1Eak7DzSvzxERzye0S-u2KA20WtIBUOE4W1xS42EIr)" IEEE Transactions on Multimedia 20.3 (2017): 645-658.'
-->
# VeRi-776 vehicle re-identification dataset

* Vehicle **ReID** in an **urban environment** [[1](https://ieeexplore.ieee.org/abstract/document/8036238?casa_token=iIk6fqQhbykAAAAA:Ypehhy5mumMJrOcVcDur_MncD7xAFdLThK1Eak7DzSvzxERzye0S-u2KA20WtIBUOE4W1xS42EIr)].
* More than $50\ 000$ images.
* $776$ vehicles captured by $20$ different cameras.

![](./images/veri776__overview.png)

---

# VeRi-776 vehicle re-identification dataset

![height:500](./images/veri776_samples.png)

---
<!--
_header: ''
_footer: ''
_paginate: false
_class: invert
-->
# *Part 4*: **Appendix** - Supplementary **Mathematics**

---
<!--
header: '*Part 4*: Appendix - Supplementary **Mathematics**'
_footer: '**[1]** - *Schroff, Florian, Dmitry Kalenichenko, and James Philbin.* "[Facenet: A unified embedding for face recognition and clustering.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)" Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.'
-->

# "Batch-all" online triplet mining strategy

[[1](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)].


---
<!--
_footer: '**[1]** - *Schroff, Florian, Dmitry Kalenichenko, and James Philbin.* "[Facenet: A unified embedding for face recognition and clustering.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)" Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.'
-->

# "Batch-hard" online triplet mining strategy

[[1](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)].

---

<!--
_footer: '**[1]** - *Dai, Jifeng, et al.* "[Deformable convolutional networks.](https://arxiv.org/abs/1703.06211)" Proceedings of the IEEE international conference on computer vision. 2017.

**[2]** - *Zhu, Xizhou, et al.* "[Deformable convnets v2: More deformable, better results.](https://arxiv.org/abs/1811.11168)" Proceedings of the IEEE/CVF conference on CVPR. 2019.
'
-->

# Deformable convolution operation

A **standard** $3 \times 3$ **convolution** with dilation $1$ employs **offets** given by
$$\mathcal{R} = \left\{ \left( −1, −1 \right), \left( −1, 0 \right), \dots, \left( 0, 1 \right), \left( 1, 1 \right) \right\}.$$
Then, for each location $\mathbf{p}_0$ within the output **feature map** $\mathbf{y}$, we compute
$$\mathbf{y} \left( \mathbf{p}_0 \right) =
\sum_{\forall \mathbf{p}_n \in \mathcal{R}}
\mathbf{w} \left( \mathbf{p}_n \right)
\cdot
\mathbf{x} \left( \mathbf{p}_0 + \mathbf{p}_n \right).$$

A **deformable convolution** **augments** the original **sampling grid** $\mathcal{R}$ with **additional offsets** $\left\{ \Delta \mathbf{p}_n \ | \ n = 1, \dots, |\mathcal{R}| \right\}$. The equation above is **reformulated** as [[1](https://arxiv.org/abs/1703.06211)]
$$\mathbf{y} \left( \mathbf{p}_0 \right) =
\sum_{\forall \mathbf{p}_n \in \mathcal{R}}
\mathbf{w} \left( \mathbf{p}_n \right)
\cdot
\mathbf{x} \left( \mathbf{p}_0 + \mathbf{p}_n + \Delta \mathbf{p}_n \right).$$
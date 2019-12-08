# Transport-Mode-GPS-CNN

## Update
We recently published an advanced version of this work in IEEE Transactions on Knowledge and Data Engineering, the second top venue in data mining and analysis, according to [Google Scholar](https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng_datamininganalysis)
In the new work, we develop a semi-supervised deep learning network for resolving the same problem. Also, the change-point detection model has been added to the architecture, which makes the mode detection framework more realistic. 

You can find the new paper in various sources at [here](https://scholar.google.com/scholar?cluster=14415315697967890786&hl=en&as_sdt=0,5)

The code repository of the work is available [here](https://github.com/sinadabiri/Deep-Semi-Supervised-GPS-Transport-Mode)

If you are interested in using codes here, I highly encourage you to use the code repository of the new paper as all data processing and supervised CNN model are also available there. 

If you are interested in this work and use the materials, please cite the following papers:

(1) *Dabiri, Sina, et al. "Semi-Supervised Deep Learning Approach for Transportation Mode Identification Using GPS Trajectory Data." IEEE Transactions on Knowledge and Data Engineering (2019).*

(2) *Dabiri, Sina, and Kevin Heaslip. "Inferring transportation modes from GPS trajectories using a convolutional neural network." Transportation research part C: emerging technologies 86 (2018): 360-371.*

**Inferring transportation modes from GPS trajectories using a Convolutional Neural Network.**

Here is the project that I have done under my PhD research. The specific objective is to predict transportation modes (e.g., walk, bike, bus, driving, and train) from only GPS trajectories using Convolutional Neural Networks (CNNs). This project has been published in the journal of Transportation Research Part C: Emerging Technologies, as one of the top-tier journals in the transportation field. The paper can be found and downloaded in https://authors.elsevier.com/a/1W8yQ,M0mR5c7L

Here, I am going to share all related codes and dataset related to this project. The detailed instruction about how to use codes is available in Definition.docx 

Further information about the methodology, results, and discussion are avaiable in the paper (the above link). 

## Paper Abstract
Identifying the distribution of users’ transportation modes is an essential part of travel demand analysis and transportation planning. With the advent of ubiquitous GPS-enabled devices (e.g., a smartphone), a cost-effective approach for inferring commuters’ mobility mode(s) is to leverage their GPS trajectories. A majority of studies have proposed mode inference models based on hand-crafted features and traditional machine learning algorithms. However, manual features engender some major drawbacks including vulnerability to traffic and environmental conditions as well as possessing human’s bias in creating efficient features. One way to overcome these issues is by utilizing Convolutional Neural Network (CNN) schemes that are capable of automatically driving high-level features from the raw input. Accordingly, in this paper, we take advantage of CNN architectures so as to predict travel modes based on only raw GPS trajectories, where the modes are labeled as walk, bike, bus, driving, and train. Our key contribution is designing the layout of the CNN’s input layer in such a way that not only is adaptable with the CNN schemes but represents fundamental motion characteristics of a moving object including speed, acceleration, jerk, and bearing rate. Furthermore, we ameliorate the quality of GPS logs through several data preprocessing steps. Using the clean input layer, a variety of CNN configurations are evaluated to achieve the best CNN architecture. The highest accuracy of 84.8% has been achieved through the ensemble of the best CNN configuration. In this research, we contrast our methodology with traditional machine learning algorithms as well as the seminal and most related studies to demonstrate the superiority of our framework. 

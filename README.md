# Car-Plate-Recognition-and-Reconstruction-with-Deep-Learning

 Automatic car plate recognition is a crucial task in the field of computer vision with wide-ranging
 applications in intelligent transportation systems, traffic monitoring, law enforcement, and access control.
 The goal is to accurately recognize and reconstruct vehicle license plates from images or video streams, often
 captured under challenging real-world conditions such as varying lighting, occlusions, motion blur, and diverse
 plate formats. Deep learning models, particularly convolutional neural networks, have significantly advanced
 the performance and reliability of car plate recognition. This project aims to explore and implement deep
 learning-based approaches for license plate recognition, emphasizing practical challenges and the impact of
 robust solutions in modern urban infrastructure and mobility management.

 Dataset: [[2]](#2)

  The objective of this project is to design and implement a deep learning-based system for license plate
 recognition, following the methodology outlined in [[1]](#1). The proposed solution is structured as a two-stage pipeline,
 leveraging the strengths of different neural network architectures to address the distinct subtasks involved in the
 recognition process. In the first stage, a YOLOv5 model is employed for license plate detection, allowing for
 fast and accurate localization of the plate region within vehicle images, even under challenging environmental
 conditions. In the second stage, the cropped plate region is passed to a specialized recognition model based on the
 PDLPR architecture. This model is responsible for decoding the sequence of alphanumeric characters on the plate,
 effectively treating the task as a sequence prediction problem. The integration of these two components aims to
 deliver a robust and efficient system for plates recognition and reconstruction suitable for deployment in real-world scenarios.
 



The project was a collaborative effort with my colleague [Filippo Casini](https://github.com/Filippo-hub).
----------------------------------------------------------------------------------------------------------
Computer vision course, Sapienza University of Rome, Artificial Intelligence and Robotics Master 
--------------------------------------------------------------------------------------------------------------




## References
<a id="1">[1]</a> 
Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). 
A Real-Time License Plate Detection and
Recognition Model in Unconstrained Scenarios. Sensors, 24(9), 2791

<a id="2">[2]</a> 
Xu, Z.; Yang, W.; Meng, A.; Lu, N.; Huang, H.; Ying, C.; Huang, L.
Towards end-to-end license plate
detection and recognition: A large dataset and baseline. In Proceedings of the European Conference on
Computer Vision (ECCV), Munich, Germany, 8–14 September 2018.

<a id="3">[3]</a> 
 R. K. Prajapati, Y. Bhardwaj, R. K. Jain and D. Kamal Kant Hiran.
”A Review Paper on Automatic Number Plate Recognition using Machine Learning : An In-Depth Analysis of Machine Learning Techniques in
Automatic Number Plate Recognition: Opportunities and Limitations,”
2023 International Conference on
Computational Intelligence, Communication Technology and Networking (CICTN), Ghaziabad, India, 2023,
pp. 527-532





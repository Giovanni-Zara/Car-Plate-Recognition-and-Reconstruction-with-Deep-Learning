# Car-Plate-Recognition-and-Reconstruction-with-Deep-Learning

 Automatic car plate recognition is a crucial task in the field of computer vision with wide-ranging
 applications in intelligent transportation systems, traffic monitoring, law enforcement, and access control.
 The goal is to accurately recognize and reconstruct vehicle license plates from images or video streams, often
 captured under challenging real-world conditions such as varying lighting, occlusions, motion blur, and diverse
 plate formats. Deep learning models, particularly convolutional neural networks, have significantly advanced
 the performance and reliability of car plate recognition. This project aims to explore and implement deep
 learning-based approaches for license plate recognition, emphasizing practical challenges and the impact of
 robust solutions in modern urban infrastructure and mobility management.

  The objective of this project is to design and implement a deep learning-based system for license plate
 recognition, following the methodology outlined in [1]. The proposed solution is structured as a two-stage pipeline,
 leveraging the strengths of different neural network architectures to address the distinct subtasks involved in the
 recognition process. In the first stage, a YOLOv5 model is employed for license plate detection, allowing for
 fast and accurate localization of the plate region within vehicle images, even under challenging environmental
 conditions. In the second stage, the cropped plate region is passed to a specialized recognition model based on the
 PDLPR architecture. This model is responsible for decoding the sequence of alphanumeric characters on the plate,
 effectively treating the task as a sequence prediction problem. The integration of these two components aims to
 deliver a robust and efficient system for plates recognition and reconstruction suitable for deployment in real-world scenarios.



The project was a collaborative effort with my colleague [Filippo Casini](https://github.com/Filippo-hub).
----------------------------------------------------------------------------------------------------------
Computer vision course at Sapienza University of Rome, Artificial Intelligence and Robotics Master 
--------------------------------------------------------------------------------------------------------------



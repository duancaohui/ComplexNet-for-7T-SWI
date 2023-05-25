# ComplexNet-for-7T-SWI
Fast and accurate reconstruction of accelerated 7T susceptibility-weighted imaging using complex-valued convolutional neural network

Susceptibility-weighted imaging (SWI) at ultra-high field 7T is a powerful tool for evaluating a wide range of pathology, but has long acquisition times. We trained a deep learning model to reconstruct 6 and 8-fold accelerated 7T SWI data. To faithfully reconstruct both MR magnitude and phase images, a ComplexNet model, specifically designed for complex-valued MR images, was implemented and trained in this work. As shown in Figure 1, ComplexNet forms an unrolled network architecture by cascading convolutional neural network (CNN) modules and data consistency layers 5 times.
![image](https://github.com/duancaohui/ComplexNet-for-7T-SWI/assets/11900034/2288c9a2-8a8d-45f8-ba53-8a2c7fd6ed5a)


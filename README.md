# leaf-classification-regression
Simple binary classification using centroid-to-regression distance approach.

## What exactly is dis?
 *Green leaf and dried leaf classification using distance of centroid-to-regression curve approach.*

First, this was mini student project from "Numerical Computation" class of Dept. of Electrical Eng. and Information Tech. UGM, Yogyakarta. This approach should not be taken seriously as research-related project or 'good way of doing it'.
The purpose of this project was to give the class insight of what regression can do. So we thought simple idea of classifying by comparing nearest distance of one's centroid to two distinct regression curve, then we determine which regression the centroid is closer to. If a point of centroid is closer to curve A than curve B, we classify that centroid as A, right? Yup that's the idea. These centroids are obtained from applying k-means clustering to every test image feature. Each image feature of test image can only have 1 centroid. Image feature used was Red and Green channel.

 Credit goes to all team members:
 - Ahmed Reza Rafsanzani
 - Awal Bakhtera Suhiyar
 - Indra Kurniawan
 - Nurman Setiawan
 - R. Cahya Hidayat
 
 ## Required Python Libraries
 - numpy
 - matplotlib
 - scikit-learn
 - shapely
 - Pillow

# Computer-Vision-CSE-559A-Forward-Motion-Deblurring
Reviewed and Implemented a 2013 ICCV paper and wrote a short writeup on limitations and advantages of the algorithm

Paper Title : Forward Motion Deblurring, Shicheng Zheng, Li Xu and Jiaya Jia In ICCV Workshops, pages 1465–1472, 2013
 
Link: http://openaccess.thecvf.com/content_iccv_2013/papers/Zheng_Forward_Motion_Deblurring_2013_ICCV_paper.pdf

Proposal :

This paper tackles the problem of a specific type of motion deblurring. i.e. forward motion deblurring which is commonly faced in situations where a camera is placed on vehicles such as dash cams or traffic cameras. In this paper they have addressed this problem by considering the variation of the plane in the z axis i.e. perpendicular to the sensor plane.

The blurred image is modeled as the summation of the product of the sharp image, a Gaussian blur kernel , a warping matrix and a weight corresponding to the duration of that pose along with additive noise. From this model we estimate the values of the weights and the sharp images using a form of regularization.

I will be implementing the algorithm and analyze why this method has limitations on certain challenging examples. I will also be studying alternate methods to address these limitations.

Please go through the report for a full breakdown of the project.

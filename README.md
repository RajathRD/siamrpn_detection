This code was written when I was working with the following repository for my research work. Please follow the guidelines given to setup the repo.

https://github.com/timy90022/One-Shot-Object-Detection

SiamRPN is primarily used for the visual object tracking task, but certain papers lately have used it as a benchmark for the query-guided object detection task (but it doesn't perform well).

SiamRPN computes correlation between query and image feature maps (using conv) to embed information of the query with the image features to generate relevant proposals in Query-Guided Object Detection task.

Use the SiamRPN implemented in rpn.py by replacing the file in ./lib/model/rpn in the repo you build with the link given above.

There are other repositories you might have to look at to include the proposal scoring and selection mechanisms as given in the paper [which pertains to a visual tracking task]. Here's a good one: https://github.com/arbitularov/SiamRPN-PyTorch

SiamRPN paper: http://www.zhengzhu.net/upload/P6938bc861e8d4583bf47d47d64ed9598.pdf

B. Li, J. Yan, W. Wu, Z. Zhu and X. Hu, "High Performance Visual Tracking with Siamese Region Proposal Network," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 8971-8980, doi: 10.1109/CVPR.2018.00935.

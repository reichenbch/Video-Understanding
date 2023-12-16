## Object Detection in Videos

Simple Solution with productionizing code, done as a Machine Learning Engineer Interview Assignment.

Domain: Video Understanding

**Task 1**:

Subtasks - 
1 - Frame Extraction.
2 - People Detection.
3 - Clustering Similar People.
4 - Plotting Results from Top-10 Clusters
5 - Evaluate the approach.

**Task 2**:

1 - Dockerize the solution.
2 - Create Webserver API for Inference.

Hi, I have created the solution for object detection from videos with the help of OpenCV, YoloV8, CLIP Model, UMAP and HDBScan.

I have created the solution with two approaches, mostly differentiating in extraction of frames from the videos. One approach is taking all frames from the video (approach 1) and Second approach is taking key frames (scene change) frames from the video.

The flow of the solution is as follows - 
1 - Video Processing/Download
2 - Frame Extraction
3 - Bounding Box Extraction and Accumulation
4 - Embedding Creation
5 - Clustering (UMAP and HDBSCAN)
6 - Visualization
7 - Deployment

Things that can be improved upon and consolidated  - 
1 - Grouping of similar bounding boxes, I tried to accomplish this via ImageHash and Embedding but the result is not upto the mark, it requires more work around and some tuning.
2 - Umap Embedding Creation will need more tuning as the embedding dimension is paramount for clustering. This needs optimisation and work.
3 - Visualization and EDA on the images need more work, this couldn't be accomplised because of inadequate image grouping.
# FaceMaskDetector

Small platform for detecting people without face masks based on [YOLOv5](https://github.com/ultralytics/yolov5). 

For system to work you should create superuser first. 
Other users are created with Django admin panel (*/admin*).

*Detailed description of the installation will be added later.*


## What platform can do

FaceMaskDetector can be customized to work not only for detecting people without face masks, but for detecting basically anything. 

I want to develop a small but universal platform, where you can upload your model and set it up for your needs. FaceMaskDetector is a prototype for this project. 

Now it can be used for processing videos using YOLOv5 model for faces and masks detection. The output file is a video where people without masks are marked.


## todo

1. Serve static files with NGINX
2. Update README.md
3. Train model with bigger dataset
4. Add faces groups
5. Optimize inference 
6. Add help menu
7. Clean up the code
8. Add progress bars for file upload and process
9. Add RTSP support and settings
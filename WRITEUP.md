# Project Write-Up

Please find the below project write for my project, answering all the required questions.

# Link of the original model

Link: https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html

Model name : person-detection-retail-0013 in FP16

command used:

Setup Openvino environment: source /opt/intel/openvino/bin/setupvars.sh 

To Run code:python main.py -m /home/workspace/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -c /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -i ./resources/Pedestrian_Detect_2_1_1.mp4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 10 -i - http://localhost:3004/fac.ffm


## Explaining Custom Layers

Custom layers are the ones that are not present in the list of suppored layers in openvino tooltik during conversion to IR (.bin and .XML) files.

The process behind converting custom layers involves the following steps

1) Build a model
2) Model extension generator is used to generate template files
3) Intermediate representation (IR) files of custom layers are generated using model optimizer
        3.1) Modify the extractor extension templete
        3.2) Modify the operator extension templete file
        3.3) use model optimizer and create .xml and .bin files
4) Modify the CPU extension template file
5) Compile the extension library
6) Finally execute the model with custom layers

Some of the potential reasons for handling custom layers are below,

1) Some models released by caffe or tensorflow might have layers that are not supported in openvino.
2) In case if a developer needs to modify the existing model with some functions that are not supported in openvino

## Comparing Model Performance

My method(s) to compare models before and after conversion are below,
1) Download the existing tensorflow, caffe or Onnx model in my workspace
2) Check the size of the model
3) Input a video or an image (if image use for loop and feed multiple times) to check the inference time.
4) Time.time() function can be used to check the prediction time of the model
5) Use a validation data and create a confusion matrix to tabulate the precision and recall scores
6) Convert this model using Openvino toolkit as an indermediate representation
7) Check the size of the converted model
8) Input a video or an image (if image use for loop and feed multiple times) to check the inference time in the newly converted model.
9) Again Time.time() function can be used to check the prediction time of the model and compare the difference in prediction time
10) Use a validation data and create a confusion matrix to tabulate the precision and recall scores with newly created model and compare with the old results

The difference between model accuracy pre- and post-conversion 

1) The model accuracy between pre and post-conversion was only 1%.
2) This accuracy change can be because of the change in precision of the model after conversion

The size of the model pre- and post-conversion.
1) There is a difference in the size of the model may be because of quantisation, freezing the layers etc
2) Around 20% of the size of the model was reduced after conversion.

The inference time of the model pre- and post-conversion
1) There is a significant change in the inference time pre and post conversion
2) The inference speed is reduced to half which in turn doubled the FPS, after using the converted IR models, This can be because of the openvino implementation.

Network requirements for cloud computing versus edge devices
1) In cloud based computing, the data acquired is transmitted to the servers/brokers and processed, in a different location compared to edge devices which does not require any network connectivity and computes,process and outputs the data in the device itself.
2) In cloud based computing a broker/server is required, where a client pulishes data/results and the other client retries or reads the published data.
3) Cloud based computing is used where the data is transmitted to the server and it can be accessed even from remote location. Latency or processing time will be more in cloud based computing compared with edge devices.
4) Cloud based computing adds cost in data transmission, processing and storage, which is not there in edge devices. Edge devices need one time hardware investment. 
4) Finally, Edge and cloud computing are not directly comparable and replacable. Both has its own advantages.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are below,

1) To count the number of people walking in streets
2) To count the number of people entering office, bus, train etc
3) To check the people enering banks, shopping malls etc 
4) To check the direction of the people entering and leaving the location
5) To monitor the people traffic in various places

Each of these use cases would be useful cases as below,

1) People entering malls will help to account for the people traffic and sales that can happen.
2) Buses, trains can be loaded only to its maximum capacity and information can be passed to the passengers waiting in the coming stops
3) High traffic of people can be alarmed as an emergency in malls, public places etc.
4) Check the queue size and divert the people to reduce their waiting time. 


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. 

The potential effects of each of these are as below,
1) Improper lighting, focal length change/ image size from the webcam or from the deployed camera may lead to improper functioning of the model. 
2) This may lead to missing the people in the frame or false predictions (predicting positive even with out people on screen)
3) Change in focal length may lead to a blurred image. It is difficult to process this kind of images, if the model is not trained with these images. The properties from the blurred image will not match with the proper focussed image. 
3) This affects the accuracy of model, which may lead to improper functioning during deployment.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD MobileNet
  - [Model Source]: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  
  - I converted the model to an Intermediate Representation with the following arguments
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

  - The model was insufficient for the app because
    The model was trained to identify so many classes and performs poor in identifying only people
  
  - I tried to improve the model for the app by import different weights but did not help
  
  
- Model 2: SqueezeNet
  - [Model Source]:https://github.com/DeepScale/SqueezeNet
  
  - I converted the model to an Intermediate Representation with the following arguments   
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt
  
  - The model was insufficient for the app because model was not predicting the people accuratly
  
  - I tried to improve the model for the app by loading different weights
  

- Model 3: MASK RCNN
  - [Model Source]: https://github.com/matterport/Mask_RCNN
  
  - I converted the model to an Intermediate Representation with the following arguments
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels 

  - The model was insufficient for the app because it is not able to identify only people
  - I tried to improve the model for the app by trying out to extract only person appearing in the screen, but did not work.

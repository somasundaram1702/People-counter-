"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2
import argparse
import openvino
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

INPUT_STREAM = "./resources/Pedestrian_Detect_2_1_1.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 1883 
MQTT_KEEPALIVE_INTERVAL = 60


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    c_desc = " The extension used, if any layers are not supported"

    # -- Required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Creating the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default= CPU_EXTENSION)
    args = parser.parse_args()

    return args

# Function to count the people coming in the Video
def count_ppl(result,counter,iflag,ppl, times, flag_in, flag_out):
    flag_in = False
    flag_out = False
    # Threshold for confirming the presence of person
    if result[0][0][0][2] > 0.9 and not iflag:
        timest1 = counter/10
        times.append(timest1)
        #print('Person {} detected in screen'.format(ppl+1))
        #print('Time of appearance {} s'.format(timest1))
        ppl +=1
        iflag = True
        flag_out = False
        flag_in = True
    if result[0][0][0][2] < 0.1 and iflag: 
        timest2 = counter/10
        #print('Out time of person:{}'.format(timest2))
        times.append(timest2)
        flag_out = True
        flag_in = False
    if result[0][0][0][2] < 0.1:
        iflag = False    
        
    return iflag,ppl,times,flag_in,flag_out

# Draw bounding box if a person is identified 
def draw_bb(result,width,height,img):
  if result[0][0][0][2] > 0.9:   
    x1 = int(width*result[0][0][0][3])
    y1 = int(height*result[0][0][0][4])
    x2 = int(width*result[0][0][0][5])
    y2 = int(height*result[0][0][0][6])
    img_b = cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0), 2)
    return(img_b)

#Inference if the input is a video
def infer_on_video(args):
    
    # Establish connection with client   
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)           
    flag_in = False
    flag_out = False
    
    # Initialize the Inference Engine
    plugin = Network()
    time_stamp = []
    
    # Load the network model into the IE
    plugin.load_model(args.m, args.d, args.c)
    net_input_shape = plugin.get_input_shape() 
    if args.i =='CAM':
       args.i = 0

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)  
    width = int(cap.get(3))
    height = int(cap.get(4))
    #Creating a video recorder
    fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out_res.avi', fourcc, 10, (width,height))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print('Frames per second is {}'.format(fps))
    
    # Process frames until the video ends, or process is exited
    times = []
    counter_frame = 0
    ppl = 0
    iflag = False
    n_ppl =1

        
    while cap.isOpened():
          
        # Read the next frame
        flag, frame = cap.read()
        #print(flag)
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        counter_frame += 1
        
        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
          
        # Perform inference on the frame
        plugin.exec_net(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.get_output()
            
            ### Process the output
            iflag, ppl, times, flag_in, flag_out = count_ppl(result,counter_frame,iflag,ppl,times,flag_in,flag_out)
            
            out_img = draw_bb(result,width,height,frame)
            out.write(out_img)
            
            if flag_in == True:
                #print('Person {} detected in screen'.format(ppl))
                #print('Time of appearance {} s'.format(times[-1]))
                person = {'total':ppl,'count':n_ppl}
                print('Total no. of ppl appeared so far is {}'.format(person['total']))
                print('Current count of ppl {}'.format(person['count']))
                client.publish('person',json.dumps({'total':ppl,'count':n_ppl}))
            elif flag_out == True:
                #print('Out time of person:{}'.format(times[-1]))
                Dur = round(times[-1]-times[-2],2) 
                person = {'person/duration':Dur}
                print('This person has spent {0:.2f} s'.format(person['person/duration']))
                client.publish('Person/Duration',json.dumps({'duration':Dur}))
                
            
            
            #Publish information in the client server
            #client.publish('person',json.dumps({'total':ppl,'count':n_ppl}))
            #client.publish('Duration',json.dumps({'duration':times}))
            
            #Publish the image
            sys.stdout.buffer.write(out_img)
            sys.stdout.flush()
            
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    return(counter_frame,ppl,time_stamp,times)

def infer_on_image(args):
    
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
          
    
    # Initialize the Inference Engine
    plugin = Network()
    time_stamp = []
    
    # Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    img = cv2.imread(args.i)  
    height,width,_ = img.shape 
    
    #Preprocess the image
    p_frame = cv2.resize(img, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    
    #Execute the network
    plugin.exec_net(p_frame)
    
    #Extract result
    result = plugin.get_output()
    
    #statistics on image
    ppl=0
    times = []
    counter_frame = 10
    iflag = False
    iflag, ppl, times = count_ppl(result,counter_frame,iflag,ppl,times)
    
    #Draw bounding box
    out_img = draw_bb(result,width,height,img)
    cv2.imwrite('file.jpg',out_img)
      
    client.publish('person',json.dumps({'count':ppl}))
    #client.publish('Duration',json.dumps({'duration':times}))
            
    #Publish the image
    sys.stdout.buffer.write(out_img)
    sys.stdout.flush()      
        
    client.disconnect()    
    return(ppl)



def time_spent(ppl,t):
    t1=[t[i] for i in range(len(t)) if i%2 ==0]  
    t2=[t[i+1] for i in range(len(t)) if i%2 ==0]
    for i in range(len(t1)):
        print('Time spent by person {b} : {k:.2f} secs'.format(b=i+1,k=t2[i]-t1[i]))
    

def main():
    args = get_args()
    if args.i.endswith('.jpg') or args.i.endswith('.png'):
        ppl= infer_on_image(args)
    elif args.i.endswith('.avi') or args.i.endswith('.mp4'):
        cnt,ppl,time_stamp,times = infer_on_video(args)
        tim = cnt/10
        print('Total number of people {}'.format(ppl))
        time_spent(ppl,times)
    else:
        print('Error :Input file should be an image of \'jpg\' or \'png\' or a Video with \'.avi\' or \'.mp4\'')
     
if __name__ == "__main__":
    main()

    
    



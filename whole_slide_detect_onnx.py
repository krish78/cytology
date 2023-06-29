# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:21:57 2023

@author: Lucid
"""

import argparse
import openslide 
import os, sys
# from pathlib import Path
import xml.etree.ElementTree as ET
import openslide
from openslide.deepzoom import DeepZoomGenerator                    
import numpy as np   
import cv2
import time
import random

import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
from tqdm import tqdm
import time
from tifffile import imsave
# from utils.dataloader import create_dataloader_custom
# from dataloader import load_custom

#import torch
# import yaml
# from models.experimental import attempt_load
# from utils.datasets import create_dataloader_custom
# from utils.general import check_img_size, box_iou, non_max_suppression,\
#     scale_coords, xyxy2xywh, xywh2xyxy, set_logging, colorstr
# # from utils.metrics import ap_per_class, ConfusionMatrix
# # from utils.plots import plot_images, output_to_target, plot_study_txt
# from utils.torch_utils import select_device, TracedModel

# from threading import thread

def update_annote_id(annotations):
    max_id = 0
    for elem in annotations.iter():
        #print(elem.tag)
        if elem.tag == 'ndpviewstate':
            _id_ = elem.attrib.get('id')                        
            if int(_id_) > max_id :
                max_id = int(_id_)
    return max_id
def get_box_list(xml_path,nm_p=221):    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x1,y1,x2,y2 = 0,0,0,0    
    box_list =[]
    for elem in root.iter():
        #print(elem.tag)
        # if elem.tag == 'ndpviewstate':
            #_id = elem.attrib.get('id')        
        x = []
        y = []
        if elem.tag == 'pointlist':
            for sub in elem.iter(tag='point'):
                x.append(int(sub.find('x').text))                    
                y.append(int(sub.find('y').text))                    
            x1=int(min(x)/nm_p)
            x2=int(max(x)/nm_p)
            y1=int(min(y)/nm_p)
            y2=int(max(y)/nm_p)
            #breath = abs(x2-x1)
            #length = abs(y2-y1)            
            row = (x1,y1,x2,y2)                
            box_list.append(row)             
    return box_list

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def write_annotation(annotations,_id,x1,y1,x2,y2,conf,X_Reference,Y_Reference,nm_p=221):
    sub_elem  = ET.SubElement(annotations,'ndpviewstate')
    sub_elem.set('id',str(_id))
    sub_elem1 = ET.SubElement(sub_elem,'title')
    sub_elem1.text = "predict" + str(_id) # str(conf)
    sub_elem2 = ET.SubElement(sub_elem,'coordformat')    
    sub_elem2.text = 'nanometers'
    sub_elem3 = ET.SubElement(sub_elem,'lens')
    sub_elem3.text = '40.0' 
    sub_elem4 = ET.SubElement(sub_elem,'fp-tp')
    sub_elem4.text = str('none')    
    sub_elemX,sub_elemY, sub_elemZ = ET.SubElement(sub_elem,'x'), ET.SubElement(sub_elem,'y'),ET.SubElement(sub_elem,'z')
    sub_show = ET.SubElement(sub_elem,'showtitle')
    sub_show.text = str(1)
    sub_show = ET.SubElement(sub_elem,'conf')
    sub_show.text = str(conf)    

    sub_show = ET.SubElement(sub_elem,'showhistogram')
    sub_show.text = str(0)
    sub_show = ET.SubElement(sub_elem,'showlineprofile')
    sub_show.text = str(0) 
    sub_elemX.text,sub_elemY.text,sub_elemZ.text = str(int((x1+x2)*nm_p/2 -X_Reference)), str(int((y1+y2)*nm_p/2 -Y_Reference)),  '0' 
    #print(sub_elemX.text,sub_elemY.text,sub_elemZ.text)
      
    anote = ET.SubElement(sub_elem,'annotation')
    anote.set('type',"freehand")
    anote.set('displayname',"AnnotateRectangle")
    color = '#90EE90'
    if conf >= 0.5 and conf < 0.7 : 
        color    = "#9acd32"        
    elif conf >= 0.7 and conf < 0.9 :
        color = '#FFA500'
    elif conf >=0.9: 
        color = '#FFFF00'     
    anote.set('color', color)
    measure_type =ET.SubElement(anote,'measuretype')
    measure_type.text = str(3)
    Pointlist = ET.SubElement(anote, 'pointlist')
    point1 = ET.SubElement(Pointlist,'point')
    ndpa_x1 = ET.SubElement(point1,'x')
    ndpa_y1 = ET.SubElement(point1,'y')
    
    ndpa_x1.text = str(int(x1*nm_p-X_Reference)) 
    ndpa_y1.text = str(int(y1*nm_p-Y_Reference))
    
    
    #print(ndpa_x1.text,ndpa_y1.text)    
    point2 = ET.SubElement(Pointlist,'point')
    
    ndpa_x2 = ET.SubElement(point2,'x')
    ndpa_y2 = ET.SubElement(point2,'y')     
    
    ndpa_x2.text = ndpa_x1.text
    ndpa_y2.text = str(int(y2*nm_p-Y_Reference))
    
    point3 = ET.SubElement(Pointlist,'point')
    ndpa_x3 = ET.SubElement(point3,'x')
    ndpa_y3 = ET.SubElement(point3,'y')
    ndpa_x3.text = str(int(x2*nm_p -X_Reference))
    
    ndpa_y3.text = ndpa_y2.text
        
    point4 = ET.SubElement(Pointlist,'point')
      
    ndpa_x4 = ET.SubElement(point4,'x')
    ndpa_y4 = ET.SubElement(point4,'y')                            
    ndpa_x4.text = ndpa_x3.text
    ndpa_y4.text = ndpa_y1.text
        
    anote_type =ET.SubElement(anote,'specialtype')
    anote_type.text = 'rectangle'
    anote_type =ET.SubElement(anote,'closed')
    anote_type.text = '1'     


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def infer_batch(session,imag,ratios,coords,anote):
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        # inp = {inname[0]:imag}        
        imag = np.ascontiguousarray(imag/255)   
        # print(imag.shape)
        outputs = session.run(outname,{'images':imag})[0]            
        if outputs.any():
            # print(outputs.shape)    
            for a,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):                
                _id = int(batch_id)                
                ratio = ratios[_id]
                coord = coords[_id]                   
                x_1 = coord[0] + (x0/ratio)
                y_1 = coord[1] + (y0/ratio)
                x_2 = coord[0] + (x1/ratio)
                y_2 = coord[1] + (y1/ratio)
                anote.append([x_1,y_1,x_2,y_2,round(float(score),3)])                        
    # end = time.time()     
    # print("inference time:",  (end-start) * 10**3, "ms")        
    # return anote    


def load_tiles_batch(tiles, level_num, imgsz = 640,overlap=0,batch_size=32,tile_size=1024):
    images = []
    shapes = []
    coords =[]
    batchs = []
    ratios =[]
    batch_i = 0
    # batching inference
    anote =[]    
    #set_logging()
    stride =32
    device,imgsz = opt.device,opt.img_size    
    print(device)
    w = "./best_reparam_fp16.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if (opt.device != 'cpu') else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)
    
    # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    names = ['abnormal']
    # batching inference
    start = time.time()
    i = 0
    breaker = False
    cols, rows = tiles.level_tiles[level_num]        
    #print(stride)
    #total tiles = cols*rows
    for row in range(rows):
        #image_rows =np.array()        
        for col in range(cols):               
            tile = tiles.get_tile(level_num, (col, row))
            coord  = tiles.get_tile_coordinates(level_num, (col, row))            
            tile_RGB = tile.convert('RGB')
            img0 = np.array(tile_RGB)                            
            dims =(tile_size+2*overlap,tile_size+2*overlap)
            img0 = cv2.resize(img0, dsize= dims, interpolation=cv2.INTER_CUBIC)
            #img = letterbox(img0, imgsz, stride=stride)[0]
            # img0 = np.transpose(img0 , (2, 0, 1)).astype(np.float16)
            img, ratio, dwdh = letterbox(img0.copy(), auto=False)
            img = np.transpose(img , (2, 0, 1)).astype(np.float16)
            # img = np.expand_dims(img, axis=0)    
            # img = img[:, :, ::-1].transpose(2, 0, 1)
            # img0 = img0[:, :, ::-1].transpose(2, 0, 1)
            img = np.expand_dims(img, 0)            
            img = np.ascontiguousarray(img)
            images.append(img)
            ratios.append(ratio)
            shapes.append(dwdh)
            #print(coord)
            coords.append(coord[0])
            i +=1
            if i % batch_size == 0 : 
                batch_i +=1 
                #print(batch_i,len(coords))
                imag = np.concatenate(images.copy()) 
                # imag = np.array(images)                                      
                batchs.append((batch_i,imag, ratios.copy(),shapes.copy(),coords.copy()))                
                images.clear()
                ratios.clear()
                shapes.clear()
                coords.clear()               
                print('batch processed',batch_i)
        #     if i > 160:
        #         breaker = True
        #         break
        # if breaker : break
    end = time.time()
    print('tiles processed',i-1)
    print('total batchs of 32:', batch_i)    
    print("The time of loading & infer tiles:",  (end-start) * 10**3, "ms")        
    return batchs    
    



# def run_predict_wsi_multithread_onnx(path,overlap,tile_size=1024,batch_size=32):
def run_predict_wsi_multithread_onnx(tiles,level_num,overlap,tile_size=1024,batch_size=32):
    anote =[]    
    #set_logging()
    stride =32
    device,imgsz = opt.device,opt.img_size    
    print(device)
    w = "./best_reparam_fp16.onnx"
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if (opt.device != 'cpu') else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)
    
    # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    names = ['abnormal']
    batches = load_tiles_batch(tiles, level_num, imgsz, batch_size=32) 
    # print('through till here')
    # dataloader = create_dataloader_custom(path, imgsz, stride, opt , batch_size=32, tile_size=1024 ,overlap=128)[0]
    # dataloader = load_custom(path, imgsz, tile_size, overlap)
    # print(len(batches))
    start = time.time()
    for i, (batch_i, imag,ratios,shapes,coords) in enumerate(tqdm(batches)) : #enumerate(tqdm(batch, desc=s)):
    # batches = dataloader.collate_fn()    
    # for batch_i, (img, ratios, coords) in enumerate(tqdm(batches)):            
        # print(batch_i,img.shape,ratios.shape,coords.shape)
        
        # print(batch_i,len(img),len(ratios),len(shapes),len(coords))
        # if batch_i >1 :
        # print(batch_i,imag.shape,len(ratios),len(shapes),len(coords))        
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        # inp = {inname[0]:imag}        
        imag = np.ascontiguousarray(imag/255)   
        # print(imag.shape)
        outputs = session.run(outname,{'images':imag})[0]            
        if outputs.any():
            # print(outputs.shape)    
            for a,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):                
                _id = int(batch_id)                
                ratio = ratios[_id]
                coord = coords[_id]                   
                x_1 = coord[0] + (x0/ratio)
                y_1 = coord[1] + (y0/ratio)
                x_2 = coord[0] + (x1/ratio)
                y_2 = coord[1] + (y1/ratio)
                anote.append([x_1,y_1,x_2,y_2,round(float(score),3)])                        
    end = time.time()     
    print("inference time:",  (end-start) * 10**3, "ms")        
    return anote    


def replace_bigger_box(boxA,boxB):            
    x1 = min(boxA[0],boxB[0])
    y1 = min(boxA[1],boxB[1])
    x2 = max(boxA[2],boxB[2])
    y2 = max(boxA[3],boxB[3])
    conf = (float(boxA[4])+ float(boxB[4]))/2    
    return x1,y1,x2,y2,conf

def prune_list(box_list):
    
    # box_list = get_box_list(wsi_path, predict_xml_path,221)
    print('before prune:',len(box_list))
    
    bool_list =[True for i in range(len(box_list))]    
    try: 
        for i in range(len(box_list)-1):
            # if i == 16:
            # print('check',bool_list[i],box_list[i])
            if bool_list[i] == True:            
                for j in range(i+1, len(box_list)):                
                    iou =bb_intersection_over_union(box_list[i],box_list[j])                                     
                    if iou > 0.01:                       
                        # replace rectlist{j] with largest of rectlist{i] and rectlist{j]
                        
                        x1,y1,x2,y2,conf = replace_bigger_box(box_list[i],box_list[j])
                        box_list[j] = (x1,y1,x2,y2,conf) 
                        bool_list[i] = False
        				# loop continue = true
                        break
        # collect all with true
        annote_final =[]
        for i in range(len(box_list)):        
            # print('check change',i,bool_list[i],box_list[i])
            if bool_list[i] == True:
                annote_final.append(box_list[i])
    except:
        print('something wrong in prune')
    print('after prune :',len(annote_final))
    return annote_final


def get_referance(wsi_path,nm_p):
    slide = openslide.open_slide(wsi_path)    
    
    w = int(slide.properties.get('openslide.level[0].width'))
    h = int(slide.properties.get('openslide.level[0].height'))
        
    ImageCenter_X = (w/2)*nm_p
    ImageCenter_Y = (h/2)*nm_p
    
    OffSet_From_Image_Center_X = slide.properties.get('hamamatsu.XOffsetFromSlideCentre')
    OffSet_From_Image_Center_Y = slide.properties.get('hamamatsu.YOffsetFromSlideCentre')
    
    print("offset from Img center units?", OffSet_From_Image_Center_X,OffSet_From_Image_Center_Y)
    
    X_Ref = float(ImageCenter_X) - float(OffSet_From_Image_Center_X)
    Y_Ref = float(ImageCenter_Y) - float(OffSet_From_Image_Center_Y)
        
    #print(ImageCenter_X,ImageCenter_Y)    
    #print(X_Reference,Y_Reference)
    return X_Ref,Y_Ref


def write_ndpa(tile_size=1024,nm_p=221, overlap=128):
    start = time.time()
    ndpi_folder = opt.folder
    wsi = opt.wsi           
    # getting the x and y reference from the wholeslide info 
    # for file in os.listdir(ndpi_folder):
    #     if file.endswith('.ndpi'):
            # wsi = file.split('.')[0]              
    wsi_path = os.path.join(ndpi_folder,wsi+".ndpi")         
    
    X_Reference,Y_Reference = get_referance(wsi_path,nm_p)
    
    xml_path= os.path.join(ndpi_folder, wsi + '_predicts.xml')    
    
    annotations = ET.Element('annotations')
        
    GT_xml_path= os.path.join(ndpi_folder, wsi +".ndpi.ndpa")    
    # if not os.path.isfile(xml_path):
    #     annotations = ET.Element('annotations')
    #     print('dont exists')
    # else :
    #     tree = ET.parse(GT_xml_path)
    #     annotations = tree.getroot()
    
    # update id for writing into the existing file.
    
    start_id =  update_annote_id(annotations)     
    # print("current :",start_id)        
    

    slide = openslide.open_slide(wsi_path)    
    tiles = DeepZoomGenerator(slide, tile_size = tile_size, overlap= overlap, limit_bounds=False)                         
    #print(tiles.level_count - 1)
    
    level_num = tiles.level_count-1
    # load a batch of images and then predict on them, 
    # anote_list = run_predict_wsi_multithread_onnx(tiles, level_num, overlap)     
    # anote_list = run_predict_wsi_multithread_onnx(wsi_path, overlap)
    # anote_list = run_predict_wsi_onnx(tiles,level_num)
    #print(anote_list)
    anote_list = load_tiles_batch(tiles, level_num)
    anote_final = prune_list(anote_list)    
    # # print(anote_final)
    anote_xml = write_xml(annotations,start_id,anote_final,X_Reference,Y_Reference)     
      
    with open(xml_path, "wb") as f:
          f.write(anote_xml)
    end = time.time()    
    if os.path.isfile(GT_xml_path):
        dump_results(GT_xml_path, xml_path)
      
    print("The total of processing:",  (end-start) * 10**3, "ms")
    # break
        
        
def write_xml(annotations,start_id,anote_list,X_Reference,Y_Reference)     :    
    id_ = start_id
    for line in anote_list : 
        write_annotation(annotations,id_,line[0],line[1],line[2],line[3],line[4],X_Reference,Y_Reference)   
        id_ +=1
    b_xml = ET.tostring(annotations)       
    return b_xml
        

def dump_results(gt_xml_path,predicts_xml_path):    
    gt_box_list = get_box_list(gt_xml_path)
    predict_box_list = get_box_list(predicts_xml_path)    
    tp_count=0       
    #write =[]
        
    for gt_box in gt_box_list:
        
        for p_box in predict_box_list:
            iou = bb_intersection_over_union(gt_box,p_box)
            if iou > 0.3:
                tp_count +=1
            #print(,iou)
            #total_predicts = len(predict_box_list)        
    
    fp_count =(len(predict_box_list)-tp_count)
    tp_rate = tp_count/len(gt_box_list)
    fp_rate = fp_count/len(predict_box_list)
    
    print('tp_count',tp_count,'recall or tp rate :', tp_rate)                
    print('fp_count',fp_count,'precsion or fp rate',fp_rate)                
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='D:\Marked_cytology\backup\ndpi_exp', help='source') 
    parser.add_argument('--wsi', type=str, default='C22-385HSIL2F', help='source')     
    parser.add_argument('--weights', nargs='+', type=str, default='best_reparam.onnx', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')        
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--augment', action='store_true', help='augmented inference')        
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))   
    start = time.time()
    try:
        write_ndpa(tile_size = 1024, overlap=128)
    except Exception as e:
        # Print the exception message
        print(f"An error occurred: {e}")
        # Prompt the user to press a key before exiting
        if sys.platform.startswith('win'):
            import msvcrt
            print("\nPress any key to exit...")
            msvcrt.getch()
        else:
            input("\nPress Enter to exit...")
    end = time.time()          
    print("The time of execution of one whole slide:", (end-start) * 10**3, "ms")
   



"""
# load and checked individual tiles 
# deprecated 

def load_tiles(tiles, level_num,overlap,tile_size=1024):
    images = []
    # shapes = []
    # coords =[]
    # batch = []
    batch_i = 0
    start = time.time()
    i = 0
    breaker = False
    cols, rows = tiles.level_tiles[level_num]        
    #print(stride)
    
    for row in range(rows):
        #image_rows =np.array()
        for col in range(cols):            
            tile = tiles.get_tile(level_num, (col, row))
            coord  = tiles.get_tile_coordinates(level_num, (col, row))            
            tile_RGB = tile.convert('RGB')
            img0 = np.array(tile_RGB)                            
            dims =(tile_size+2*overlap,tile_size+2*overlap)
            img0 = cv2.resize(img0, dsize= dims, interpolation=cv2.INTER_CUBIC)
            # img = letterbox(img0, imgsz, stride=stride)[0]
            # img = img[:, :, ::-1].transpose(2, 0, 1)
            # img0 = img0[:, :, ::-1].transpose(2, 0, 1)
            images.append((row,col,coord,img0))            
            i += 1
            if i > 200:
                breaker = True
                break
        if breaker : break
    end = time.time()
    print('tiles processed',i-1)    
    print("The time of loading tiles:",  (end-start) * 10**3, "ms")        
    return images

def run_predict_wsi_onnx(tiles,level_num,overlap=0,tile_size=1024,batch_size=32):
    anote =[]    
    #set_logging()
    device,imgsz = opt.device,opt.img_size    
    w = "./best_reparam_fp16.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if (opt.device != 'cpu') else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)
    
    # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    names = ['abnormal']
    images = load_tiles(tiles, level_num,overlap)
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    count =0
    for col,row, coords, img0 in images:
        img, ratio, dwdh = letterbox(img0, auto=False)
        left,top = coords[0]
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.transpose(img, (2, 0, 1)).astype(np.float16)
        img = np.ascontiguousarray(img)        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img, axis=0)    
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]:img}        
        outputs = session.run(outname,inp)[0]
        # print(outputs.shape)
        if outputs.any():
            ori_images = [img0.copy()]
            for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
                image = ori_images[int(batch_id)]
                box = np.array([x0,y0,x1,y1])
                box -= np.array(dwdh*2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                cls_id = int(cls_id)
                score = round(float(score),3)
                name = names[cls_id]
                color = colors[name]
                name += ' '+str(score)
                cv2.rectangle(image,box[:2],box[2:],color,2)
                cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
            count +=1
            imsave(r'd:\dump\tile_{0}.tif'.format(count),ori_images[0])
    return anote

"""

"""
batch predict on a whole slide

"""

    
    
    
    
    
    
    
    
    
    
    


import cv2
import os
import json
import numpy as np
from PIL import ImageColor
import configparser
from dataloader import SNV3Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from SoccerNet.Evaluation.utils import INVERSE_FRAME_CLASS_DICTIONARY, FRAME_CLASS_COLOR_DICTIONARY
from argparse import ArgumentParser
import torch

def torch2cv2(image):

    img = torch.permute(image.clone(), (1, 2, 0) ).numpy()
    tmp_channel = img[:,:,0].copy()
    img[:,:,0] = img[:,:,2]
    img[:,:,2] = tmp_channel
    return img.copy()

def get_color(color):
    box_color = ImageColor.getcolor(color, "RGB")
    return (box_color[2],box_color[1],box_color[0])

def draw_bboxes(image, bboxes, thickness=8):

    width = image.size()[-1]
    height = image.size()[-2]
    image_bbox = torch2cv2(image[0])
    for bbox in bboxes[0]:
        image_bbox = cv2.rectangle(image_bbox, (int(bbox[0]*width),int(bbox[1]*height)), (int((bbox[0]+bbox[2])*width),int((bbox[1]+bbox[3])*height)), get_color(FRAME_CLASS_COLOR_DICTIONARY[INVERSE_FRAME_CLASS_DICTIONARY[int(bbox[4])]]), thickness)

    return image_bbox

def draw_lines(image, lines, thickness=8):

    width = image.size()[-1]
    height = image.size()[-2]
    image_line = torch2cv2(image[0])
    for line in lines:
        for i in np.arange(len(line[0][0])//2-1):
            image_line = cv2.line(image_line, (int(line[0][0][i*2]*width),int(line[0][0][i*2+1]*height)), (int(line[0][0][i*2+2]*width),int(line[0][0][i*2+3]*height)), get_color(FRAME_CLASS_COLOR_DICTIONARY[INVERSE_FRAME_CLASS_DICTIONARY[int(line[1][0][0])]]), thickness)
    return image_line

def draw_links(data,thickness=8):


    width = data[0]["image"].shape[-1]
    height = data[0]["image"].shape[-2]
    num_images = len(data)
    final_image = np.zeros((height*2,width*2,3))

    for i, d in enumerate(data[:4]):

        image = torch2cv2(d["image"][0])

        bboxes = d["bboxes"]
        links = d["links"]

        anchor_h = i%2*height
        anchor_w = i//2*width

        final_image[anchor_h:anchor_h+height,anchor_w:anchor_w+width] = image

        for bbox in bboxes[0]:
            final_image = cv2.rectangle(final_image, (anchor_w+int(bbox[0]*width),anchor_h+int(bbox[1]*height)), (anchor_w+int((bbox[0]+bbox[2])*width),anchor_h+int((bbox[1]+bbox[3])*height)), get_color(FRAME_CLASS_COLOR_DICTIONARY[INVERSE_FRAME_CLASS_DICTIONARY[int(bbox[4])]]), thickness)
        
        for link in links[0]:

            if link[2] > 3:
                continue
            final_image = cv2.line(final_image, (int(anchor_w+data[int(link[0])]["bboxes"][0][int(link[1])][0]*width),
                                                int(anchor_h+data[int(link[0])]["bboxes"][0][int(link[1])][1]*height)),
                                                (int(link[2]//2*width+data[int(link[2])]["bboxes"][0][int(link[3])][0]*width),
                                                int(link[2]%2*height+data[int(link[2])]["bboxes"][0][int(link[3])][1]*height)), 
                                                get_color(FRAME_CLASS_COLOR_DICTIONARY[INVERSE_FRAME_CLASS_DICTIONARY[int(data[int(link[0])]["bboxes"][0][int(link[1])][4])]]),
                                                thickness )

    return final_image


if __name__ == "__main__":


    # Default parameters# Load the arguments
    parser = ArgumentParser(description='Visualization tool')
    
    parser.add_argument('--SoccerNet_path',   required=True, type=str, help='Path to the SoccerNet-V3 dataset folder' )
    parser.add_argument('--save_path',   required=True, type=str, help='Path to save the images' )
    parser.add_argument('--tiny',   required=False, type=int, default=None, help='Select a subset of x games' )
    parser.add_argument('--split',   required=False, type=str, default="all", help='Select the split of data' )
    parser.add_argument('--num_workers',   required=False, type=int, default=8, help='number of workers for the dataloader' )
    parser.add_argument('--resolution_width',   required=False, type=int, default=1920, help='width resolution of the images' )
    parser.add_argument('--resolution_height',   required=False, type=int, default=1080, help='height resolution of the images' )
    parser.add_argument('--zipped_images', action='store_true', help="Read images from zipped folder")

    args = parser.parse_args()

    # Load the dataset
    soccernet = SNV3Dataset(args.SoccerNet_path, split=args.split, resolution=(args.resolution_width,args.resolution_height), preload_images=False, tiny=args.tiny,zipped_images=args.zipped_images)
    soccernet_loader = DataLoader(soccernet, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    with tqdm(enumerate(soccernet_loader), total=len(soccernet_loader), ncols=160) as t:
        for i, data in t: 
            for j, d in enumerate(data):
                cv2.imwrite(args.save_path + str(i) + "_" + str(j) + "_original.png", torch2cv2(d["image"][0]))
                img_bbox = draw_bboxes(d["image"],d["bboxes"])
                cv2.imwrite(args.save_path + str(i) + "_" + str(j) + "_bboxes.png", img_bbox)
                img_line = draw_lines(d["image"],d["lines"])
                cv2.imwrite(args.save_path + str(i) + "_" + str(j) + "_lines.png", img_line)
            img_links = draw_links(data)
            cv2.imwrite(args.save_path + str(i) + "_links.png", img_links)
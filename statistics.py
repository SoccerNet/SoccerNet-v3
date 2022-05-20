import os
import json
import numpy as np
from dataloader import SNV3Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from SoccerNet.Evaluation.utils import INVERSE_FRAME_CLASS_DICTIONARY, FRAME_CLASS_COLOR_DICTIONARY
from argparse import ArgumentParser
import torch

def define_dict():

    stats = dict()
    stats["num_games"] = 0
    stats["num_actions"] = 0
    stats["num_replays"] = 0
    stats["num_links"] = 0
    stats["num_bboxes"] = 0
    stats["num_lines"] = 0
    stats["num_line_points"] = 0
    stats["classes"] = dict()
    #Load the classes
    for c in FRAME_CLASS_COLOR_DICTIONARY.keys():
        stats["classes"][c] = dict()
        stats["classes"][c]["linked_ID"] = 0
        stats["classes"][c]["number_ID"] = 0
        stats["classes"][c]["num_points"] = 0
        stats["classes"][c]["num"] = 0
    return stats

def count_bbox(stats,bboxes):

    for box in bboxes[0]:
        stats["num_bboxes"] += 1
        stats["classes"][INVERSE_FRAME_CLASS_DICTIONARY[int(box[4])]]["num"] += 1
        stats["classes"][INVERSE_FRAME_CLASS_DICTIONARY[int(box[4])]]["num_points"] += 2
        if box[5] >= 0:
            stats["classes"][INVERSE_FRAME_CLASS_DICTIONARY[int(box[4])]]["number_ID"] += 1

def count_lines(stats,lines):

    for line in lines:
        stats["num_lines"] += 1
        stats["num_line_points"] += len(line[0][0])//2
        stats["classes"][INVERSE_FRAME_CLASS_DICTIONARY[int(line[1][0][0])]]["num"] += 1
        stats["classes"][INVERSE_FRAME_CLASS_DICTIONARY[int(line[1][0][0])]]["num_points"] += len(line[0][0])//2

def count_links(stats,bboxes,links):
    already_counted = list() 
    for link in links[0]:
        stats["num_links"] += 1
        if int(link[1]) not in already_counted:
            stats["classes"][INVERSE_FRAME_CLASS_DICTIONARY[int(bboxes[0][int(link[1])][4])]]["linked_ID"] += 1
            already_counted.append(int(link[1]))

if __name__ == "__main__":


    # Default parameters# Load the arguments
    parser = ArgumentParser(description='Statistics')
    
    parser.add_argument('--SoccerNet_path',   required=True, type=str, help='Path to the SoccerNet-V3 dataset folder' )
    parser.add_argument('--save_path',   required=True, type=str, help='Path to save the images' )
    parser.add_argument('--tiny',   required=False, type=int, default=None, help='Select a subset of x games' )
    parser.add_argument('--split',   required=False, type=str, default="all", help='Select the split of data' )
    parser.add_argument('--num_workers',   required=False, type=int, default=8, help='number of workers for the dataloader' )
    parser.add_argument('--resolution_width',   required=False, type=int, default=1920, help='width resolution of the images' )
    parser.add_argument('--resolution_height',   required=False, type=int, default=1080, help='height resolution of the images' )
    parser.add_argument('--zipped_images', action='store_true', help="Read images from zipped folder")

    args = parser.parse_args()

    stats = define_dict()

    # Load the dataset
    soccernet = SNV3Dataset(args.SoccerNet_path, split=args.split, resolution=(args.resolution_width,args.resolution_height), preload_images=False, tiny=args.tiny,zipped_images=args.zipped_images)
    soccernet_loader = DataLoader(soccernet, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    stats["num_games"] = len(soccernet.list_games)
    with tqdm(enumerate(soccernet_loader), total=len(soccernet_loader), ncols=160) as t:
        for i, data in t: 
            stats["num_actions"] += 1 
            stats["num_replays"] += len(data)-1 
            for j, d in enumerate(data):
                count_bbox(stats, d["bboxes"])
                count_lines(stats, d["lines"])
                count_links(stats,d["bboxes"], d["links"])

    stats["num_links"] = stats["num_links"]//2
    with open(args.save_path + 'stats_' + args.split + '.json', 'w') as f:
        json.dump(stats, f, indent=4)
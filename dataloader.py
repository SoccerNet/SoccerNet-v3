import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import zipfile
from PIL import Image
from tqdm import tqdm
import os
import json
from SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY
import copy
import time
from SoccerNet.utils import getListGames
from argparse import ArgumentParser

class SNV3Dataset(Dataset):

	def __init__(self, path, split="all", resolution=(1920,1080), preload_images=False, tiny=None, zipped_images=False):

		# Path for the SoccerNet-v3 dataset 
		# containing the images and labels
		self.path = path

		# Get the list of the selected subset of games
		self.list_games = getListGames(split, task="frames")
		if tiny is not None:
			self.list_games = self.list_games[:tiny]

		# Resolution of the images to load (width, height)
		self.resolution = resolution
		self.resize = torchvision.transforms.Resize((resolution[1],resolution[0]), antialias=True)
		self.preload_images = preload_images
		self.zipped_images = zipped_images

		# Variable to store the metadata
		print("Reading the annotation files")
		self.metadata = list()
		for game in tqdm(self.list_games):
			self.metadata.append(json.load(open(os.path.join(self.path, game, "Labels-v3.json"))))

		# Variables to store the preloaded images and annotations
		# Each element in the list is a list of images and annotations linked to an action
		self.data = list()
		for annotations in tqdm(self.metadata):

			# Retrieve each action in the game
			for action_name in annotations["GameMetadata"]["list_actions"]:

				# concatenate the replays of each action with itself
				img_list = [action_name] + annotations["actions"][action_name]["linked_replays"]
				self.data.append(list())
				IDs_list = list()

				zipfilepath = os.path.join(self.path,annotations["GameMetadata"]["UrlLocal"], 'Frames-v3.zip')
				if self.zipped_images:
					zippedFrames = zipfile.ZipFile(zipfilepath, 'r')

				# For each image extract the images and annotations
				for i, img in enumerate(img_list):

					# Variable to save the annotation
					data_tmp = dict()
					data_tmp["image"] = None

					# Only the first frame is an action, the rest are replays
					img_type = "actions"
					if i > 0 :
						img_type="replays"

					filepath = os.path.join(self.path,annotations["GameMetadata"]["UrlLocal"], "v3_frames", img)
					if self.preload_images:
						with torch.no_grad():
							if self.zipped_images:
								imginfo = zippedFrames.open(img)
								data_tmp["image"] = self.resize(transforms.ToTensor()(Image.open(imginfo))*255)
							else:
								data_tmp["image"] = self.resize(torchvision.io.read_image(filepath))

					data_tmp["zipfilepath"] = zipfilepath
					data_tmp["imagefilepath"] = img	
					data_tmp["filepath"] = filepath					
				
					data_tmp["bboxes"], ID_tmp = self.format_bboxes(annotations[img_type][img]["bboxes"], annotations[img_type][img]["imageMetadata"])
					
					data_tmp["lines"] = self.format_lines(annotations[img_type][img]["lines"], annotations[img_type][img]["imageMetadata"])
					
					data_tmp["links"] = None

					IDs_list.append(ID_tmp)

					self.data[-1].append(data_tmp)

				self.format_links(IDs_list)

	def format_bboxes(self, bboxes, image_metadata):

		# Bounding boxes in x_top, y_top, width, height, cls_idx, num_idx
		data = list()

		IDs = list()

		for i, bbox in enumerate(bboxes):

			if bbox["class"] is not None:

				tmp_data = torch.zeros((4+1+1,), dtype=torch.float)-1
				tmp_data[0] = bbox["points"]["x1"]/image_metadata["width"]	
				tmp_data[1] = bbox["points"]["y1"]/image_metadata["height"]
				tmp_data[2] = abs(bbox["points"]["x2"]-bbox["points"]["x1"])/image_metadata["width"]	
				tmp_data[3] = abs(bbox["points"]["y2"]-bbox["points"]["y1"])/image_metadata["height"]
				tmp_data[4] = float(FRAME_CLASS_DICTIONARY[bbox["class"]])
				if bbox["ID"] is not None: 
					if bbox["ID"].isnumeric():
						tmp_data[5] = float(bbox["ID"])
				IDs.append([bbox["ID"],FRAME_CLASS_DICTIONARY[bbox["class"]]])
				data.append(tmp_data)

		data = torch.stack(data)
		return data, IDs

	def format_lines(self, lines, image_metadata):

		# Each element is a list with list of points, cls_idx
		data = list()

		for line in lines:

			if line["class"] is not None:
				points = torch.FloatTensor(line["points"])
				points[::2] = points[::2]/image_metadata["width"]
				points[1::2] = points[1::2]/image_metadata["height"]
				data.append([points,torch.FloatTensor([FRAME_CLASS_DICTIONARY[line["class"]]])])
		return data

	def format_links(self, IDs_list):

		# Links are stored as (index of the current image, 
		# index of the bounding box in the first image,
		# index of the second image, 
		# index of the bounding box in the second image)

		for i, IDs_1 in enumerate(IDs_list):

			list_of_links = list()

			for j, IDs_2 in enumerate(IDs_list):

				if i == j:
					continue

				for k, ID_1 in enumerate(IDs_1):

					for l, ID_2 in enumerate(IDs_2):
						if ID_1[1] == ID_2[1]:
							if ID_1[0] is not None and ID_2[0] is not None:
								if ID_1[0] == ID_2[0]:
									list_of_links.append([i,k,j,l])
									continue
			
			self.data[-1][i]["links"] = torch.FloatTensor(list_of_links)


	def __getitem__(self, index):

		if not self.preload_images:
			data = copy.deepcopy(self.data[index])
			with torch.no_grad():
				image_list = list()
				for i, d in enumerate(data):
					if self.zipped_images:
						imginfo = zipfile.ZipFile(d["zipfilepath"], 'r').open(d["imagefilepath"])
						img = transforms.ToTensor()(Image.open(imginfo))*255
						data[i]["image"] = self.resize(img)
			
					else:
						data[i]["image"] = self.resize(torchvision.io.read_image(d["filepath"]))
				return data

		return self.data[index]

	def __len__(self):

		return len(self.data)

if __name__ == "__main__":

	# Load the arguments
	parser = ArgumentParser(description='dataloader')
	
	parser.add_argument('--SoccerNet_path',   required=True, type=str, help='Path to the SoccerNet-V3 dataset folder' )
	parser.add_argument('--tiny',   required=False, type=int, default=None, help='Select a subset of x games' )
	parser.add_argument('--split',   required=False, type=str, default="all", help='Select the split of data' )
	parser.add_argument('--num_workers',   required=False, type=int, default=4, help='number of workers for the dataloader' )
	parser.add_argument('--resolution_width',   required=False, type=int, default=1920, help='width resolution of the images' )
	parser.add_argument('--resolution_height',   required=False, type=int, default=1080, help='height resolution of the images' )
	parser.add_argument('--preload_images', action='store_true', help="Preload the images when constructing the dataset")
	parser.add_argument('--zipped_images', action='store_true', help="Read images from zipped folder")

	args = parser.parse_args()

	
	start_time = time.time()
	soccernet = SNV3Dataset(args.SoccerNet_path, split=args.split, resolution=(args.resolution_width,args.resolution_height), preload_images=args.preload_images, zipped_images=args.zipped_images, tiny=args.tiny)
	soccernet_loader = DataLoader(soccernet, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
	with tqdm(enumerate(soccernet_loader), total=len(soccernet_loader), ncols=160) as t:
		for i, data in t: 
			continue
	end_time = time.time()
	print(end_time-start_time)

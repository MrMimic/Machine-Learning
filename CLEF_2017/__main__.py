#/usr/bin/env python3

# https://sites.google.com/site/clefehealth2016/task-2

__version__ = '0.0.1'
__author__ = 'emeric'


import os
import re
import json
from nltk.corpus import stopwords
from collections import Counter


class Utilities(object):
	""""""

	def split_text(self, string):
		tokens = re.findall(r"\w+(?:[-]{1,2})?(?:\w+)?(?:[-]{1,2})?(?:\w+)?", str(string))
		return tokens

	def find_ngrams(self, input_list, n):
		return list(zip(*[input_list[i:] for i in range(int(n))]))


class ImportDataset(object):

	def __init__(self):

		self.train_path = '/home/emeric/1_Github/Machine-Learning/CLEF_2017/dataset/corpus/train/MEDLINE'
		self.stopwords = stopwords.words('french')

	def files_statistics(self):
		""""""

		# Number of files
		text_files = []
		annotation_files = []
		for file_name in os.listdir(self.train_path):
			if re.search('[A-Za-z0-9]{1,30}\.txt', str(file_name)):
				text_files.append(file_name)
			elif re.search('[A-Za-z0-9]{1,30}\.ann', str(file_name)):
				annotation_files.append(file_name)
		print('{} files have been retrived ({} texts, {} annotations).'.format(len(text_files) + len(annotation_files), len(text_files), len(annotation_files)))

		# Number of concepts
		unique_concepts = []
		all_concepts = []
		for file_name in annotation_files:
			with open('{}/{}'.format(self.train_path, file_name), 'r') as current_file:
				raw_data = current_file.readlines()
				for raw_line in raw_data:
					splitted_line = raw_line.strip('\n').split('\t')
					if re.search('^\#', splitted_line[0]):
						all_concepts.append(splitted_line[2])
						if splitted_line[2] not in unique_concepts:
							unique_concepts.append(splitted_line[2])
		true_uniques = [concept[0] for concept in Counter(all_concepts).most_common() if concept[1] == 1]
		print('{} unique concepts have been extracted ({} true unique).'.format(len(unique_concepts), len(true_uniques)))

		return text_files, annotation_files

	def load_data(self, text_files, annotation_files):
		""""""

		dataset = {}
		for text_name in text_files:

			text_details = {}
			with open('{}/{}'.format(self.train_path, text_name), 'r') as current_file:
				raw_text = current_file.read()
			with open('{}/{}'.format(self.train_path, re.sub('txt', 'ann', str(text_name)), 'r')) as current_file:
				raw_annotation = current_file.readlines()
			
			# Let's link the two annotation lines together
			text_annotations = []
			for annotation_1 in raw_annotation:
				if re.search('^T[0-9]{1,2}', str(annotation_1.strip('\n'))):
					id_annotation_1 = re.findall('^T([0-9]{1,2})', str(annotation_1.strip('\n')))[0]
					for annotation_2 in raw_annotation:
						if annotation_1 != annotation_2:
							if re.search('^\#{}'.format(id_annotation_1), str(annotation_2.strip('\n'))):
								annonations = {annotation_1.strip('\n').split('\t')[2]: annotation_2.strip('\n').split('\t')[2]}
								text_annotations.append(annonations)

			# Process text
			text_details['raw_text'] = raw_text.lower().splitlines()
			text_details['raw_annotation'] = text_annotations
			dataset[re.sub('\.txt', '', str(text_name))] = text_details

		return dataset

	def gram_data(self, dataset):
		""""""

		utilities = Utilities()
		for data in dataset:
			
			# Split text
			splitted_text = utilities.split_text(string=dataset[data]['raw_text'])
			# Remove stopwords
			splitted_text_clean = [word for word in splitted_text if word not in self.stopwords]
			# Gram from 1 to 4
			grams = {}
			grams['G1'] = splitted_text_clean
			grams['G2'] = utilities.find_ngrams(input_list=splitted_text_clean, n=2)
			grams['G3'] = utilities.find_ngrams(input_list=splitted_text_clean, n=3)
			grams['G4'] = utilities.find_ngrams(input_list=splitted_text_clean, n=4)
			dataset[data]['grams'] = grams

		return dataset



# https://github.com/jiegzhan/multi-class-text-classification-cnn/blob/master/train.py

if __name__ == '__main__':

	# Import dataset
	dataset_loader = ImportDataset()
	text_files, annotation_files = dataset_loader.files_statistics()
	dataset = dataset_loader.load_data(text_files=text_files, annotation_files=annotation_files)
	dataset = dataset_loader.gram_data(dataset=dataset)

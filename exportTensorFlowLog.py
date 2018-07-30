import tensorflow as tf
import time
import csv
import sys
import os
import collections

# Import the event accumulator from Tensorboard. Location varies between Tensorflow versions. Try each known location until one works.
eventAccumulatorImported = False;
# TF version < 1.1.0
if (not eventAccumulatorImported):
	try:
		from tensorflow.python.summary import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version = 1.1.0
if (not eventAccumulatorImported):
	try:
		from tensorflow.tensorboard.backend.event_processing import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version >= 1.3.0
if (not eventAccumulatorImported):
	try:
		from tensorboard.backend.event_processing import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version = Unknown
if (not eventAccumulatorImported):
	raise ImportError('Could not locate and import Tensorflow event accumulator.')

summariesDefault = ['scalars','histograms','images','audio','compressedHistograms'];

class Timer(object):
	# Source: https://stackoverflow.com/a/5849861
	def __init__(self, name=None):
		self.name = name

	def __enter__(self):
		self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		if self.name:
			print('[%s]' % self.name)
			print('Elapsed: %s' % (time.time() - self.tstart))

def exitWithUsage():
	print(' ');
	print('Usage:');
	print('   python readLogs.py <output-folder> <output-path-to-csv> <summaries>');
	print('Inputs:');
	print('   <input-path-to-logfile>  - Path to TensorFlow logfile.');
	print('   <output-folder>          - Path to output folder.');
	print('   <summaries>              - (Optional) Comma separated list of summaries to save in output-folder. Default: ' + ', '.join(summariesDefault));
	print(' ');
	sys.exit();

if (len(sys.argv) < 3):
	exitWithUsage();

inputLogFile = sys.argv[1];
outputFolder = sys.argv[2];

if (len(sys.argv) < 4):
	summaries = summariesDefault;
else:
	if (sys.argv[3] == 'all'):
		summaries = summariesDefault;
	else:
		summaries = sys.argv[3].split(',');

print(' ');
print('> Log file: ' + inputLogFile);
print('> Output folder: ' + outputFolder);
print('> Summaries: ' + ', '.join(summaries));

if any(x not in summariesDefault for x in summaries):
	print('Unknown summary! See usage for acceptable summaries.');
	exitWithUsage();


print(' ');
print('Setting up event accumulator...');
with Timer():
	ea = event_accumulator.EventAccumulator(inputLogFile,
  	size_guidance={
      	event_accumulator.COMPRESSED_HISTOGRAMS: 0, # 0 = grab all
      	event_accumulator.IMAGES: 0,
      	event_accumulator.AUDIO: 0,
      	event_accumulator.SCALARS: 0,
      	event_accumulator.HISTOGRAMS: 0,
	})

print(' ');
print('Loading events from file*...');
print('* This might take a while. Sit back, relax and enjoy a cup of coffee :-)');
with Timer():
	ea.Reload() # loads events from file

print(' ');
print('Log summary:');
tags = ea.Tags();
for t in tags:
	tagSum = []
	if (isinstance(tags[t],collections.Sequence)):
		tagSum = str(len(tags[t])) + ' summaries';
	else:
		tagSum = str(tags[t]);
	print('   ' + t + ': ' + tagSum);

if not os.path.isdir(outputFolder):
	os.makedirs(outputFolder);

if ('audio' in summaries):
	print(' ');
	print('Exporting audio...');
	with Timer():
		print('   Audio is not yet supported!');

if ('compressedHistograms' in summaries):
	print(' ');
	print('Exporting compressedHistograms...');
	with Timer():
		print('   Compressed histograms are not yet supported!');


if ('histograms' in summaries):
	print(' ');
	print('Exporting histograms...');
	with Timer():
		print('   Histograms are not yet supported!');

if ('images' in summaries):
	print(' ');
	print('Exporting images...');
	imageDir = outputFolder + 'images'
	print('Image dir: ' + imageDir);
	with Timer():
		imageTags = tags['images'];
		for imageTag in imageTags:
			images = ea.Images(imageTag);
			imageTagDir = imageDir + '/' + imageTag;
			if not os.path.isdir(imageTagDir):
				os.makedirs(imageTagDir);
			for image in images:
				imageFilename = imageTagDir + '/' + str(image.step) + '.png';
				with open(imageFilename,'wb') as f:
					f.write(image.encoded_image_string);

if ('scalars' in summaries):
	print(' ');
	csvFileName =  os.path.join(outputFolder,'scalars.csv');
	print('Exporting scalars to csv-file...');
	print('   CSV-path: ' + csvFileName);
	scalarTags = tags['scalars'];
	with Timer():
		with open(csvFileName,'w') as csvfile:
			logWriter = csv.writer(csvfile, delimiter=',');

			# Write headers to columns
			headers = ['wall_time','step'];
			for s in scalarTags:
				headers.append(s);
			logWriter.writerow(headers);
	
			vals = ea.Scalars(scalarTags[0]);
			for i in range(len(vals)):
				v = vals[i];
				data = [v.wall_time, v.step];
				for s in scalarTags:
					scalarTag = ea.Scalars(s);
					if len(scalarTag) > i:
						S = scalarTag[i];
						data.append(S.value);
					else:
						data.append("");
				logWriter.writerow(data);

print(' ');
print('Bye bye...');


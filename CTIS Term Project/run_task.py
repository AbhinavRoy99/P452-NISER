#WORKING
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from numpy import asarray as ar,exp
import seaborn as sns
import pandas as pd
import spectral
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import concurrent.futures
import multiprocessing as mp
from lib_hyperspec import *
import os

data_path = "/home/royabhinav/Desktop/CTIS/Data_Apr19/"

output_path = "/home/royabhinav/Desktop/CTIS/Output_Apr19/"


def calibration_task_1(filename):
	filepath = data_path + filename + ".tif"

	#making a folder for the output
	foldername = output_path + "calibration" + "/"
	os.makedirs(foldername, exist_ok=True)

	#reading the data
	with rasterio.open(filepath) as src:
		image = src.read()
		#print(image.shape)

	#plotting the data
	plot_img(image, filename, foldername, save_ok=True)

	# Initial guess for the parameters
	initial_guess_col = [30, 1000, 50, 100, 1500, 50, 30, 2000, 50]

	# Initial guess for the parameters
	initial_guess_row = [30, 1700, 100, 150, 2200, 100, 30, 2700, 100]

	region_1, region_2, region_3, region_4, central_region = get_regions(image, initial_guess_col, initial_guess_row, filename, foldername, save_ok=True)

	# writing regions to a single text file
	write_regions(region_1, region_2, region_3, region_4, central_region, filename, foldername)

	#plotting the regions
	peaks_1 = plot_region_1(region_1, image, filename, foldername, 'Left', plot_peak=True, save_ok=True)
	peaks_2 = plot_region_1(region_2, image, filename, foldername, 'Right', plot_peak=True, save_ok=True)

	#plotting the regions
	peaks_3 = plot_region_2(region_3, image, filename, foldername, 'Top', plot_peak=True, save_ok=True)
	peaks_4 = plot_region_2(region_4, image, filename, foldername, 'Bottom', plot_peak=True, save_ok=True)

	#plotting the central region
	center_y, center_x = plot_region_3(central_region, image, filename, foldername, 'Central', save_ok=True)

	#write peaks to csv file
	write_peaks(peaks_1, peaks_2, peaks_3, peaks_4, filename, foldername)


def calibration_task_2(filename, known_peaks):
	filepath = data_path + filename + ".tif"

	foldername = output_path + "calibration" + "/"
	os.makedirs(foldername, exist_ok=True)

	#reading peaks from csv file
	peaks_1, peaks_2, peaks_3, peaks_4 = read_peaks(filename, foldername)

	peaks_1 = peaks_1[::-1]
	#peaks_2 = peaks_2
	peaks_3 = peaks_3[::-1]
	#peaks_4 = peaks_4

	calibration_left, calibration_right, calibration_top, calibration_bottom = calibrate_peaks_wavelengths(peaks_1, peaks_2, peaks_3, peaks_4, known_peaks)

	#write calibration to csv file
	write_calibration(calibration_left, calibration_right, calibration_top, calibration_bottom, filename, foldername)

	#plotting the calibration curves
	plot_calibration_curve(calibration_left, peaks_1, known_peaks, 'Left', filename, foldername, save_ok=True)
	plot_calibration_curve(calibration_right, peaks_2, known_peaks, 'Right', filename, foldername, save_ok=True)
	plot_calibration_curve(calibration_top, peaks_3, known_peaks, 'Top', filename, foldername, save_ok=True)
	plot_calibration_curve(calibration_bottom, peaks_4, known_peaks, 'Bottom', filename, foldername, save_ok=True)



def img_task_1(filename):
	filepath = data_path + filename + ".tif"

	foldername = output_path + filename + "/"
	os.makedirs(foldername, exist_ok=True)

	#reading the data
	with rasterio.open(filepath) as src:
		image = src.read()
		#print(image.shape)
	
	#plotting the data
	plot_img(image, filename, foldername, save_ok=True)

	# Initial guess for the parameters
	initial_guess_col = [30, 1000, 50, 100, 1500, 50, 30, 2000, 50]

	# Initial guess for the parameters
	initial_guess_row = [30, 1700, 100, 150, 2200, 100, 30, 2700, 100]

	region_1, region_2, region_3, region_4, central_region = get_regions(image, initial_guess_col, initial_guess_row, filename, foldername, save_ok=True)

	# writing regions to a single text file
	write_regions(region_1, region_2, region_3, region_4, central_region, filename, foldername)

	#plotting the regions
	plot_region_1(region_1, image, filename, foldername, 'Left', plot_peak=False, save_ok=True)
	plot_region_1(region_2, image, filename, foldername, 'Right', plot_peak=False, save_ok=True)

	#plotting the regions
	plot_region_2(region_3, image, filename, foldername, 'Top', plot_peak=False, save_ok=True)
	plot_region_2(region_4, image, filename, foldername, 'Bottom', plot_peak=False, save_ok=True)

	#plotting the central region
	center_y, center_x = plot_region_3(central_region, image, filename, foldername, 'Central', save_ok=True)
	

def task_cube(filename, calibration_filename):

	filepath = data_path + filename + ".tif"

	foldername = output_path + filename + "/"
	os.makedirs(foldername, exist_ok=True)

	#reading the data
	with rasterio.open(filepath) as src:
		image = src.read()
		#print(image.shape)

	#reading the regions
	region_1, region_2, region_3, region_4, central_region = read_regions(filename, foldername)

	#reading the calibration
	calibration_path = output_path + "calibration" + "/"
	calibration_left, calibration_right, calibration_top, calibration_bottom = read_calibration(calibration_filename, calibration_path)

	#generate spectra cube
	cube, new_wavelengths = spectra_cube(image, central_region, region_1, region_2, region_3, region_4, calibration_left, calibration_right, calibration_top, calibration_bottom)

	#saving cube and wavelengths
	np.save(foldername+filename+"_cube.npy", cube)
	np.save(foldername+filename+"_wavelengths.npy", new_wavelengths)







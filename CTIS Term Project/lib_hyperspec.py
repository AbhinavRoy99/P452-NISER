#LIBRARY FOR HYPERSPECTRAL DATA ANALYSIS
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


def data_grey(data):
	# Convert to grayscale
	gray = 0.2989 * data[0,:,:] + 0.5870 * data[1,:,:] + 0.1140 * data[2,:,:]
	return gray

# Gaussian function
def gaussian(x, amplitude, mean, stddev):
	return amplitude * exp(-((x - mean) / stddev) ** 2 / 2)

# Function for three Gaussian peaks
def three_gaussians(x, a1, x01, sigma1, a2, x02, sigma2, a3, x03, sigma3):
	return gaussian(x, a1, x01, sigma1) + gaussian(x, a2, x02, sigma2) + gaussian(x, a3, x03, sigma3)

def fit_gaussian(x, y):
	p0 = [np.max(y), np.argmax(y), 1.0]  # Initial guess for parameters
	popt, _ = curve_fit(gaussian, x, y, p0=p0)
	return popt

def calculate_fwhm(stddev):
	return 2 * np.sqrt(2 * np.log(2)) * stddev

def plot_img(image, filename, output_path, save_ok=False):

	data = image.astype('float32')

	# Convert to grayscale
	gray = data_grey(data)

	# Create a figure
	fig, ax = plt.subplots(figsize=(10, 10))

	# Plot the grayscale image
	ax.imshow(gray, cmap='gray')
	ax.set_title(f'Grayscale Image of {filename}')

	# Show the figure
	plt.tight_layout()
	if save_ok:
		plt.savefig(output_path + filename + "_grayscale.png")
		plt.close()
	#plt.show()


	# Create a figure
	fig, axs = plt.subplots(3, 1, figsize=(10, 10))

	# Plot the data for the first three bands
	for i, ax in enumerate(axs):
		plot_data = np.squeeze(data[i,:,:])
		ax.imshow(plot_data, cmap='gray')
		ax.set_title(f'Band {i+1} of {filename}')

	# Show the figure
	plt.tight_layout()
	if save_ok:
		plt.savefig(output_path + filename + "_bands.png")
		plt.close()
	#plt.show()

def get_regions(image, initial_guess_col, initial_guess_row, filename, output_path, save_ok=False):
		
		# Create a figure with two subplots, one above the other
		fig, axs = plt.subplots(2, 1, figsize=(20, 17))

		# Calculate the mean intensity for each x position
		mean_intensity_x = np.max(np.sum(image, axis=0), axis=1)

		x = ar(range(len(mean_intensity_x)))

		# Initial guess for the parameters
		#initial_guess_col = [30, 1100, 50, 100, 1500, 50, 30, 1700, 50]

		# Fit the function to the data
		popt_x, pcov = curve_fit(three_gaussians, x, mean_intensity_x, p0=initial_guess_col)

		# Plot the mean intensity versus the column index on the first subplot
		axs[0].plot(mean_intensity_x)
		axs[0].plot(three_gaussians(x, *popt_x), 'r-', label='fit_x')

		points = np.array([popt_x[1],popt_x[4],popt_x[7]])
		axs[0].plot(x[points.astype(int)], mean_intensity_x[points.astype(int)], 'o', color='black', label='peaks_x')

		base_points_x = np.array([popt_x[1]+popt_x[2]+100,popt_x[1]-popt_x[2]-100,popt_x[4]+popt_x[5]+70,popt_x[4]-popt_x[5]-70,popt_x[7]+popt_x[8]+100,popt_x[7]-popt_x[8]-100])
		#print("Base points: ", base_points_x)
		axs[0].plot(x[base_points_x.astype(int)], mean_intensity_x[base_points_x.astype(int)], 'o', color='red', label='bases_x')

		# Set labels
		axs[0].set_xlabel('Column Index')
		axs[0].set_ylabel('Mean Intensity')

		# Print the parameters for x
		# print("Parameters for x:")
		# print("Base (center) of Gaussian 1: ", popt_x[1],popt_x[2]+100)
		# print("Base (center) of Gaussian 2: ", popt_x[4],popt_x[5]+70)
		# print("Base (center) of Gaussian 3: ", popt_x[7],popt_x[8]+100)




		# Calculate the mean intensity for each y position
		mean_intensity_y = np.max(np.sum(image, axis=0), axis=0)

		y = ar(range(len(mean_intensity_y)))

		# Initial guess for the parameters
		#initial_guess = [30, 1700, 100, 150, 2200, 100, 30, 2700, 100]

		# Fit the function to the data
		popt_y, pcov = curve_fit(three_gaussians, y, mean_intensity_y, p0=initial_guess_row)

		# Plot the mean intensity versus the row index on the second subplot
		axs[1].plot(mean_intensity_y)
		axs[1].plot(three_gaussians(y, *popt_y), 'r-', label='fit_y')
		points = np.array([popt_y[1],popt_y[4],popt_y[7]])
		axs[1].plot(y[points.astype(int)], mean_intensity_y[points.astype(int)], 'o', color='black', label='peaks_y')

		base_points_y = np.array([popt_y[1]+popt_y[2]+100,popt_y[1]-popt_y[2]-100,popt_y[4]+popt_y[5]+70,popt_y[4]-popt_y[5]-70,popt_y[7]+popt_y[8]+100,popt_y[7]-popt_y[8]-100])
		#print("Base points: ", base_points_y)
		axs[1].plot(y[base_points_y.astype(int)], mean_intensity_y[base_points_y.astype(int)], 'o', color='red', label='bases_y')

		# Set labels
		axs[1].set_xlabel('Row Index')
		axs[1].set_ylabel('Mean Intensity')

		# Print the parameters for y
		# print("Parameters for y:")
		# print("Base (center) of Gaussian 1: ", popt_y[1],popt_y[2]+100)
		# print("Base (center) of Gaussian 2: ", popt_y[4],popt_y[5]+70)
		# print("Base (center) of Gaussian 3: ", popt_y[7],popt_y[8]+100)

		# Show the plot
		plt.title(f'Gaussian Fitting and Region Extraction for {filename}')
		plt.tight_layout()
		plt.legend()

		if save_ok:
			plt.savefig(output_path + filename + "_gaussian_fitting.png")
			plt.close()


		#region extraction
		# print("Base points: ", base_points_y)
		# print("Base points: ", base_points_x)

		region_1 = [base_points_y[0],base_points_y[1],base_points_x[2],base_points_x[3]]
		region_2 = [base_points_y[4],base_points_y[5],base_points_x[2],base_points_x[3]]
		region_3 = [base_points_y[2],base_points_y[3],base_points_x[0],base_points_x[1]]
		region_4 = [base_points_y[2],base_points_y[3],base_points_x[4],base_points_x[5]]

		central_region = [base_points_y[2],base_points_y[3],base_points_x[2],base_points_x[3]]

		#plt.show()

		region_1 = [int(i) for i in region_1]
		region_2 = [int(i) for i in region_2]
		region_3 = [int(i) for i in region_3]
		region_4 = [int(i) for i in region_4]
		central_region = [int(i) for i in central_region]

		return region_1, region_2, region_3, region_4, central_region


def write_regions(region_1, region_2, region_3, region_4, central_region, filename, output_path):
	# Write the region coordinates to a csv file
	region_df = pd.DataFrame({
		'Region 1': region_1,
		'Region 2': region_2,
		'Region 3': region_3,
		'Region 4': region_4,
		'Central Region': central_region
	})
	region_df.to_csv(output_path + filename + "_regions.csv")


def read_regions(filename, output_path):
	# Read the region coordinates from a csv file
	region_df = pd.read_csv(output_path + filename + "_regions.csv", index_col=0)
	region_1 = region_df['Region 1'].values
	region_2 = region_df['Region 2'].values
	region_3 = region_df['Region 3'].values
	region_4 = region_df['Region 4'].values
	central_region = region_df['Central Region'].values
	return region_1, region_2, region_3, region_4, central_region


def plot_region_1(region, image, filename, output_path, orient, plot_peak = True, save_ok=False):
	#change region vals to ints
	#region = [int(i) for i in region]

	data = image.astype('float32')
	
	# Create a figure
	fig, axs = plt.subplots(1, 3, figsize=(20, 15))

	# Plot the data and histogram for the first three bands
	for i, ax in enumerate(axs):
		plot_data = data[i,region[3]:region[2],region[1]:region[0]]
		extent = [region[1], region[0], region[3], region[2]]  # Define the extent
		ax.imshow(plot_data, cmap='gray', extent=extent)  # Add the extent parameter
		ax.set_title(f'Band {i+1} for {orient} Projection')
		#ax.axis('equal')  # Set the aspect ratio to 1
		
	# Show the figure
	plt.tight_layout()
	if save_ok:
		plt.savefig(output_path + filename + orient + "_bands.png")
		plt.close()
	#plt.show()

	# Create a figure
	fig, ax = plt.subplots(figsize=(10, 10))

	# Plot the data and histogram for the first three bands
	plot_data = data_grey(data[:,region[3]:region[2],region[1]:region[0]])
	extent = [region[1], region[0], region[3], region[2]]  # Define the extent
	ax.imshow(plot_data, cmap='gray', extent=extent)  # Add the extent parameter
	ax.set_title(f'All Bands for {orient} Projection')
	#ax.axis('equal')  # Set the aspect ratio to 1
		
	# Show the figure
	plt.tight_layout()
	if save_ok:
		plt.savefig(output_path + filename + orient + "_all_bands.png")
		plt.close()
	#plt.show()

	if plot_peak:
		# print(plot_data.shape)
		# Process the image column-wise to obtain intensity values
		intensity_values = np.sum(plot_data, axis=0)
		# print(len(intensity_values))
		# Find peaks in the intensity values (adjust parameters as needed)
		min_distance = 1  # Minimum pixel distance between peaks
		prominence = 100   # Minimum intensity difference between peak and surrounding area
		peaks, _ = find_peaks(intensity_values, distance=min_distance, prominence=prominence)

		col_values = np.arange(len(intensity_values))
		col_values = col_values + region[1]
		# print(region[1], region[0], region[3], region[2])

		# Plot the spectrum with detected peaks
		plt.figure(figsize=(10, 6))
		plt.plot(col_values, intensity_values, '-b', label='Spectrum')
		plt.plot(peaks+region[1], intensity_values[peaks], '-ro', markersize=8, label='Detected Peaks in lab setup')
		plt.xlabel('Pixel Position')
		plt.ylabel('Intensity')
		plt.title('Spectrum with Detected Peaks')
		plt.legend()
		plt.grid(True)
		#plt.axis('equal')  # Set the aspect ratio to 1

		# Display the pixel positions and corresponding intensity values of the detected peaks
		# print('Detected Peaks:')
		# for peak in peaks:
		# 	print(f'Pixel Position: {peak+region[1]}, Intensity: {intensity_values[peak]}')
		# print(len(peaks))

		# Show the plot
		plt.tight_layout()
		if save_ok:
			plt.savefig(output_path + filename + orient + "_spectrum_peaks.png")
			plt.close()
		#plt.show()

		return peaks+region[1]


def plot_region_2(region, image, filename, output_path, orient, plot_peak = True, save_ok=False):
	#change region vals to ints
	#region = [int(i) for i in region]

	data = image.astype('float32')
	
	# Create a figure
	fig, axs = plt.subplots(1, 3, figsize=(20, 15))

	# Plot the data and histogram for the first three bands
	for i, ax in enumerate(axs):
		plot_data = data[i,region[3]:region[2],region[1]:region[0]]
		extent = [region[1], region[0], region[3], region[2]]  # Define the extent
		ax.imshow(plot_data, cmap='gray', extent=extent)  # Add the extent parameter
		ax.set_title(f'Band {i+1} for {orient} Projection')
		#ax.axis('equal')  # Set the aspect ratio to 1
		
	# Show the figure
	plt.tight_layout()
	if save_ok:
		plt.savefig(output_path + filename + orient + "_bands.png")
		plt.close()
	#plt.show()

	# Create a figure
	fig, ax = plt.subplots(figsize=(10, 10))

	# Plot the data and histogram for the first three bands
	plot_data = data_grey(data[:,region[3]:region[2],region[1]:region[0]])
	extent = [region[1], region[0], region[3], region[2]]  # Define the extent
	ax.imshow(plot_data, cmap='gray', extent=extent)  # Add the extent parameter
	ax.set_title(f'All Bands for {orient} Projection')
	#ax.axis('equal')  # Set the aspect ratio to 1
		
	# Show the figure
	plt.tight_layout()
	if save_ok:
		plt.savefig(output_path + filename + orient + "_all_bands.png")
		plt.close()
	#plt.show()

	# print(plot_data.shape)
	if plot_peak:
		# Process the image column-wise to obtain intensity values
		intensity_values = np.sum(plot_data, axis=1)
		# print(len(intensity_values))

		# Find peaks in the intensity values (adjust parameters as needed)
		min_distance = 4  # Minimum pixel distance between peaks
		prominence = 100   # Minimum intensity difference between peak and surrounding area
		peaks, _ = find_peaks(intensity_values, distance=min_distance, prominence=prominence)

		col_values = np.arange(len(intensity_values))
		col_values = col_values + region[3]
		# print(region[1], region[0], region[3], region[2])

		# Plot the spectrum with detected peaks
		plt.figure(figsize=(10, 6))
		plt.plot(col_values, intensity_values, '-b', label='Spectrum')
		plt.plot(peaks+region[3], intensity_values[peaks], '-ro', markersize=8, label='Detected Peaks in lab setup')
		plt.xlabel('Pixel Position')
		plt.ylabel('Intensity')
		plt.title('Spectrum with Detected Peaks')
		plt.legend()
		plt.grid(True)
		#plt.axis('equal')  # Set the aspect ratio to 1

		# Display the pixel positions and corresponding intensity values of the detected peaks
		# print('Detected Peaks:')
		# for peak in peaks:
		# 	print(f'Pixel Position: {peak+region[3]}, Intensity: {intensity_values[peak]}')
		# print(len(peaks))

		# Show the plot
		plt.tight_layout()
		if save_ok:
			plt.savefig(output_path + filename + orient + "_spectrum_peaks.png")
			plt.close()
		#plt.show()

		return peaks+region[3]


def plot_region_3(region, image, filename, output_path, orient, save_ok=False):
	#change region vals to ints
	#region = [int(i) for i in region]

	data = image.astype('float32')
	
	# Create a figure
	fig, axs = plt.subplots(1, 3, figsize=(20, 15))

	# Plot the data and histogram for the first three bands
	for i, ax in enumerate(axs):
		plot_data = data[i,region[3]:region[2],region[1]:region[0]]
		extent = [region[1], region[0], region[3], region[2]]  # Define the extent
		ax.imshow(plot_data, cmap='gray', extent=extent)  # Add the extent parameter
		ax.set_title(f'Band {i+1} for {orient} Projection')
		#ax.axis('equal')  # Set the aspect ratio to 1
		
	# Show the figure
	plt.tight_layout()
	if save_ok:
		plt.savefig(output_path + filename + orient + "_bands.png")
		plt.close()
	#plt.show()

	# Create a figure
	fig, ax = plt.subplots(figsize=(10, 10))

	# Plot the data and histogram for the first three bands
	plot_data = data_grey(data[:,region[3]:region[2],region[1]:region[0]])
	extent = [region[1], region[0], region[3], region[2]]  # Define the extent
	ax.imshow(plot_data, cmap='gray', extent=extent)  # Add the extent parameter
	ax.set_title(f'All Bands for {orient} Projection')
	#ax.axis('equal')  # Set the aspect ratio to 1
		
	# Show the figure
	plt.tight_layout()
	if save_ok:
		plt.savefig(output_path + filename + orient + "_all_bands.png")
		plt.close()
	#plt.show()
	#"""
	# Process the image column-wise to obtain intensity values
	intensity_values_along_y = np.sum(plot_data, axis=0)
	intensity_values_along_x = np.sum(plot_data, axis=1)

	#Gaussian Fitting to determine the center of the peak
	y_values = np.arange(len(intensity_values_along_x))
	params = fit_gaussian(y_values, intensity_values_along_x)
	fit_curve = gaussian(y_values, *params)

	fwhm = calculate_fwhm(params[2])
	peak_centery = params[1]

	# print(region[3]+peak_centery, fwhm)	
	
	#Gaussian Fitting to determine the center of the peak
	y_values = np.arange(len(intensity_values_along_y))
	params = fit_gaussian(y_values, intensity_values_along_y)
	fit_curve = gaussian(y_values, *params)

	fwhm = calculate_fwhm(params[2])
	peak_centerx = params[1]

	# print(region[1]+peak_centerx, fwhm)

	return region[3]+peak_centery, region[1]+peak_centerx

def write_peaks(peaks_1, peaks_2, peaks_3, peaks_4, filename, output_path):
	# Write the peak coordinates to a csv file
	#peak_data = np.array([peaks_1, peaks_2, peaks_3, peaks_4])

	# Find the maximum length
	max_len = max(len(peaks_1), len(peaks_2), len(peaks_3), len(peaks_4))

	# Pad the lists with np.nan
	peaks_1 = np.pad(peaks_1, (0, max_len - len(peaks_1)), 'constant', constant_values=-1)
	peaks_2 = np.pad(peaks_2, (0, max_len - len(peaks_2)), 'constant', constant_values=-1)
	peaks_3 = np.pad(peaks_3, (0, max_len - len(peaks_3)), 'constant', constant_values=-1)
	peaks_4 = np.pad(peaks_4, (0, max_len - len(peaks_4)), 'constant', constant_values=-1)

	# Create the DataFrame
	peak_df = pd.DataFrame({
		'peaks1' : peaks_1,
		'peaks2' : peaks_2,
		'peaks3' : peaks_3,
		'peaks4' : peaks_4
	}).astype('int64')
	peak_df.to_csv(output_path + filename + "_peaks.csv", index=False)

def read_peaks(filename, output_path):
	# Read the peak coordinates from a csv file
	peak_df = pd.read_csv(output_path + filename + "_peaks.csv", dtype=np.int64)
	print(peak_df.keys())
	peaks_1 = peak_df['peaks1'].values
	peaks_2 = peak_df['peaks2'].values
	peaks_3 = peak_df['peaks3'].values
	peaks_4 = peak_df['peaks4'].values

	# remove -1 from the peaks
	peaks_1 = peaks_1[peaks_1 != -1]
	peaks_2 = peaks_2[peaks_2 != -1]
	peaks_3 = peaks_3[peaks_3 != -1]
	peaks_4 = peaks_4[peaks_4 != -1]

	#check if all peak arrays are equal in length
	assert len(peaks_1) == len(peaks_2) == len(peaks_3) == len(peaks_4)

	return peaks_1, peaks_2, peaks_3, peaks_4

def write_calibration(calibration_left, calibration_right, calibration_top, calibration_bottom, filename, output_path):
	df = pd.DataFrame({
		'calibration_left': calibration_left,
		'calibration_right': calibration_right,
		'calibration_top': calibration_top,
		'calibration_bottom': calibration_bottom
	}).astype('float64')

	# Save the DataFrame to a CSV file
	df.to_csv(output_path + filename + "_calibration.csv", index=False)

def read_calibration(filename, output_path):
	# Read the calibration coefficients from a csv file
	calibration_df = pd.read_csv(output_path + filename + "_calibration.csv", dtype=np.float64)
	calibration_left = calibration_df['calibration_left'].values
	calibration_right = calibration_df['calibration_right'].values
	calibration_top = calibration_df['calibration_top'].values
	calibration_bottom = calibration_df['calibration_bottom'].values
	return calibration_left, calibration_right, calibration_top, calibration_bottom

def calibrate_peaks_wavelengths(peaks_left, peaks_right, peaks_top, peaks_bottom, known_wavelengths):

	# Ensure that the number of peaks is the same as the number of known wavelengths
	assert len(peaks_left) == len(known_wavelengths)
	assert len(peaks_right) == len(known_wavelengths)
	assert len(peaks_top) == len(known_wavelengths)
	assert len(peaks_bottom) == len(known_wavelengths)

	# Fit a polynomial of degree 2 to the pixel positions and known wavelengths
	calibration_left = np.polyfit(peaks_left, known_wavelengths, 2)
	calibration_right = np.polyfit(peaks_right, known_wavelengths, 2)
	calibration_top = np.polyfit(peaks_top, known_wavelengths, 2)
	calibration_bottom = np.polyfit(peaks_bottom, known_wavelengths, 2)

	# Print the calibration functions
	# print()
	# print("Calibration function for left: y = {:.3f}x^2 + {:.3f}x + {:.3f}".format(calibration_left[0], calibration_left[1], calibration_left[2]))
	# print("Calibration function for right: y = {:.3f}x^2 + {:.3f}x + {:.3f}".format(calibration_right[0], calibration_right[1], calibration_right[2]))
	# print("Calibration function for top: y = {:.3f}x^2 + {:.3f}x + {:.3f}".format(calibration_top[0], calibration_top[1], calibration_top[2]))
	# print("Calibration function for bottom: y = {:.3f}x^2 + {:.3f}x + {:.3f}".format(calibration_bottom[0], calibration_bottom[1], calibration_bottom[2]))

	return calibration_left, calibration_right, calibration_top, calibration_bottom


def plot_calibration_curve(calibration_coefficients, pixel_positions, known_wavelengths, name, filename, output_path, save_ok=False):

	# Plot the calibration curve
	plt.figure(figsize=(8, 6))
	plt.scatter(pixel_positions, known_wavelengths, label='Calibration Points')
	pixel_positions_fit = np.linspace(min(pixel_positions), max(pixel_positions), 100)
	wavelengths_fit = np.polyval(calibration_coefficients,pixel_positions_fit)
	plt.plot(pixel_positions_fit, wavelengths_fit, label='Calibration Curve', color='red')
	plt.xlabel('Pixel Position')
	plt.ylabel('Wavelength (nm)')
	plt.title(f'Spectroscope Calibration with Helium Lamp: {name} Projection \n Calibration Equation: Wavelength (nm) = {calibration_coefficients[0]:.6f} * (x)^2 + {calibration_coefficients[1]:.4f} * x + {calibration_coefficients[2]:.4f}')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()

	# Display the calibration equation
	#print(f'')
	#print(calibration_coefficients)
	# Save or show the plot
	#plt.savefig('helium_calibration.png')
	if save_ok:
		plt.savefig(output_path + filename + name + "_calibration_curve.png")
		plt.close()
	#plt.show()


def plot_spectra_1(calibration_coefficients, pixel_positions, known_wavelengths, region, image, name):

	data = image.astype('float32')

	plot_data = data_grey(data)

	plot_data = plot_data[region[3]:region[2], region[1]:region[0]]

	intensity_values = np.sum(plot_data, axis=0)

	col_values = np.arange(len(intensity_values))
	col_values = col_values + region[1]

	#print(col_values[0], col_values[-1], len(intensity_values), len(col_values), plot_data.shape)
	#print()

	calibrated_wavelengths = np.polyval(calibration_coefficients, col_values)
	#print(calibrated_wavelengths[0], calibrated_wavelengths[-1], len(calibrated_wavelengths))
	calibrated_peaks = np.polyval(calibration_coefficients, pixel_positions)
	#print(calibrated_peaks[0], calibrated_peaks[-1], len(calibrated_peaks))

	pixel_positions = np.array(pixel_positions)
	peaks = pixel_positions - region[1]
	#print(peaks)

	plt.figure(figsize=(10, 6))
	plt.plot(calibrated_wavelengths, intensity_values, '-b', label='Spectrum')
	plt.plot(calibrated_peaks, intensity_values[peaks], '-ro', markersize=8, label='Detected Peaks in lab setup')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Intensity')
	plt.title(f'Calibrated Spectrum : {name} Projection')
	plt.legend()
	plt.grid(True)

	plt.show()


def plot_spectra_2(calibration_coefficients, pixel_positions, known_wavelengths, region, image, name):

	data = image.astype('float32')

	plot_data = data_grey(data)

	plot_data = plot_data[region[3]:region[2], region[1]:region[0]]

	intensity_values = np.sum(plot_data, axis=1)

	col_values = np.arange(len(intensity_values))
	col_values = col_values + region[3]

	#print(col_values[0], col_values[-1], len(intensity_values), len(col_values), plot_data.shape)
	#print()

	calibrated_wavelengths = np.polyval(calibration_coefficients, col_values)
	#print(calibrated_wavelengths[0], calibrated_wavelengths[-1], len(calibrated_wavelengths))
	calibrated_peaks = np.polyval(calibration_coefficients, pixel_positions)
	#print(calibrated_peaks[0], calibrated_peaks[-1], len(calibrated_peaks))

	pixel_positions = np.array(pixel_positions)
	peaks = pixel_positions - region[3]
	#print(peaks)

	plt.figure(figsize=(10, 6))
	plt.plot(calibrated_wavelengths, intensity_values, '-b', label='Spectrum')
	plt.plot(calibrated_peaks, intensity_values[peaks], '-ro', markersize=8, label='Detected Peaks in lab setup')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Intensity')
	plt.title(f'Calibrated Spectrum : {name} Projection')
	plt.legend()
	plt.grid(True)

	plt.show()


def combine_spectra_x_y(region_1_img, region_2_img, region_3_img, region_4_img, calibrated_wavelengths1, calibrated_wavelengths2, calibrated_wavelengths3, calibrated_wavelengths4, min_wavelength, max_wavelength, new_wavelengths, x, y):

	#mid_point = int(region_1_img.shape[0]/2)

	#print(mid_point)

	intensity_mid1 = region_1_img[int(x), :]
	intensity_mid2 = region_2_img[int(x), :]

	#normalise intensity 
	intensity_mid1 = intensity_mid1/np.max(intensity_mid1)
	intensity_mid2 = intensity_mid2/np.max(intensity_mid2)

	#print(intensity_mid1.shape)
	#print(intensity_mid2.shape)

	"""
	plt.figure()
	plt.plot(calibrated_wavelengths1, intensity_mid1, label='Region 1')
	plt.plot(calibrated_wavelengths2, intensity_mid2, label='Region 2')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Intensity')
	plt.legend()
	plt.show()
	#"""

	#mid_point = int(region_3_img.shape[1]/2)

	#print(mid_point)

	intensity_mid3 = region_3_img[:, y]
	intensity_mid4 = region_4_img[:, y]

	#normalise intensity
	intensity_mid3 = intensity_mid3/np.max(intensity_mid3)
	intensity_mid4 = intensity_mid4/np.max(intensity_mid4)

	#print(intensity_mid3.shape)
	#print(intensity_mid4.shape)

	"""
	plt.figure()
	plt.plot(calibrated_wavelengths3, intensity_mid3, label='Region 3')
	plt.plot(calibrated_wavelengths4, intensity_mid4, label='Region 4')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Intensity')
	plt.legend()
	plt.show()

	plt.figure()
	plt.plot(calibrated_wavelengths1, intensity_mid1, label='Region 1')
	plt.plot(calibrated_wavelengths2, intensity_mid2, label='Region 2')
	plt.plot(calibrated_wavelengths3, intensity_mid3, label='Region 3')
	plt.plot(calibrated_wavelengths4, intensity_mid4, label='Region 4')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Intensity')
	plt.legend()
	plt.show()
	#"""

	# Create interpolation functions
	interp_func1 = interp1d(calibrated_wavelengths1, intensity_mid1, kind='cubic')
	interp_func2 = interp1d(calibrated_wavelengths2, intensity_mid2, kind='cubic')
	interp_func3 = interp1d(calibrated_wavelengths3, intensity_mid3, kind='cubic')
	interp_func4 = interp1d(calibrated_wavelengths4, intensity_mid4, kind='cubic')

	# Interpolate the spectra onto the new wavelength range
	new_intensity1 = interp_func1(new_wavelengths)
	new_intensity2 = interp_func2(new_wavelengths)
	new_intensity3 = interp_func3(new_wavelengths)
	new_intensity4 = interp_func4(new_wavelengths)

	# Combine the spectra by averaging
	combined_intensity = (new_intensity1 + new_intensity2 + new_intensity3 + new_intensity4) / 4

	"""
	plt.figure()
	plt.plot(new_wavelengths, combined_intensity, label='Combined Spectrum')
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Intensity')
	plt.legend()
	plt.show()
	#"""

	return new_wavelengths, combined_intensity

def worker_cube(args):
		
		x, y, region_1_img, region_2_img, region_3_img, region_4_img, calibrated_wavelengths1, calibrated_wavelengths2, calibrated_wavelengths3, calibrated_wavelengths4, min_wavelength, max_wavelength, new_wavelengths = args
		_, combined_spectrum = combine_spectra_x_y(region_1_img, region_2_img, region_3_img, region_4_img, calibrated_wavelengths1, calibrated_wavelengths2, calibrated_wavelengths3, calibrated_wavelengths4, min_wavelength, max_wavelength, new_wavelengths, x, y)
		return x, y, combined_spectrum

def spectra_cube(image, central_region, region_1, region_2, region_3, region_4, calibration_1, calibration_2, calibration_3, calibration_4):

	grey = data_grey(image)

	# Extract the central region
	central_region_img = grey[central_region[3]:central_region[2], central_region[1]:central_region[0]]

	# Extract the region 1
	region_1_img = grey[region_1[3]:region_1[2], region_1[1]:region_1[0]]

	# Extract the region 2
	region_2_img = grey[region_2[3]:region_2[2], region_2[1]:region_2[0]]

	# Extract the region 3
	region_3_img = grey[region_3[3]:region_3[2], region_3[1]:region_3[0]]

	# Extract the region 4
	region_4_img = grey[region_4[3]:region_4[2], region_4[1]:region_4[0]]


	# print(central_region_img.shape)
	# print(region_1_img.shape)
	# print(region_2_img.shape)
	# print(region_3_img.shape)
	# print(region_4_img.shape)

	col_val_1 = np.arange(region_1_img.shape[1])
	col_val_1 = col_val_1 + region_1[1]
	#print(col_val_1)
	col_val_2 = np.arange(region_2_img.shape[1])
	col_val_2 = col_val_2 + region_2[1]
	#print(col_val_2)

	col_val_3 = np.arange(region_3_img.shape[0])
	col_val_3 = col_val_3 + region_3[3]
	#print(col_val_3)
	col_val_4 = np.arange(region_4_img.shape[0])
	col_val_4 = col_val_4 + region_4[3]
	#print(col_val_4)

	#print(col_val_1.shape)
	#print(col_val_1)
	#print(calibration_1)
	calibrated_wavelengths1 = np.polyval(calibration_1, col_val_1)
	#print(min(calibrated_wavelengths1),max(calibrated_wavelengths1), region_1[1])
	#print(col_val_2.shape)
	#print(col_val_2)
	#print(calibration_2)
	calibrated_wavelengths2 = np.polyval(calibration_2, col_val_2)
	#print(min(calibrated_wavelengths2),max(calibrated_wavelengths2), region_2[1])


	#print(col_val_3.shape)
	#print(col_val_3)
	#print(calibration_3)
	calibrated_wavelengths3 = np.polyval(calibration_3, col_val_3)
	#print(min(calibrated_wavelengths3),max(calibrated_wavelengths3), region_3[3])
	#print(col_val_4.shape)
	#print(col_val_4)
	#print(calibration_4)
	calibrated_wavelengths4 = np.polyval(calibration_4, col_val_4)
	#print(min(calibrated_wavelengths4),max(calibrated_wavelengths4), region_4[3])

	#print(min(calibrated_wavelengths1), max(calibrated_wavelengths1))
	#print(min(calibrated_wavelengths2), max(calibrated_wavelengths2))

	# Create a new wavelength range that is common to all spectra
	min_wavelength = max(min(calibrated_wavelengths1), min(calibrated_wavelengths2),min(calibrated_wavelengths3), min(calibrated_wavelengths4))
	max_wavelength = min(max(calibrated_wavelengths1), max(calibrated_wavelengths2),max(calibrated_wavelengths3), max(calibrated_wavelengths4))

	new_wavelengths = np.linspace(min_wavelength, max_wavelength, 1000)
	print(min_wavelength, max_wavelength)

	# Create a shared array to hold the combined spectra
	manager = mp.Manager()
	combined_spectra = manager.list([manager.list([manager.list([0]*len(new_wavelengths)) for _ in range(central_region_img.shape[1])]) for _ in range(central_region_img.shape[0])])

	# Create a list of arguments for each x and y coordinate
	args = [(x, y, region_1_img, region_2_img, region_3_img, region_4_img, calibrated_wavelengths1, calibrated_wavelengths2, calibrated_wavelengths3, calibrated_wavelengths4, min_wavelength, max_wavelength, new_wavelengths) for x in range(central_region_img.shape[0]) for y in range(central_region_img.shape[1])]

	# Use a ProcessPoolExecutor to run the combine_spectra_x_y function in parallel
	with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
		for x, y, combined_spectrum in executor.map(worker_cube, args):
			combined_spectra[x][y] = combined_spectrum

	# Convert the shared list to a numpy array
	combined_spectra = np.array(combined_spectra)
	
	return combined_spectra, new_wavelengths


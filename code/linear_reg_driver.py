import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
# from PIL import Image, ImageDraw, ImageFont
# import glob
# import cv2
# import numpy as np
# import re

import sys
# Add the directory where your module is located to sys.path
module_path = "../../../models/linear-regression/"
if module_path not in sys.path:
    sys.path.append(module_path)

# Now you can import your module
from linear_reg import LinearRegression 

csv_dataset = '../../../data/external/linreg.csv'
train_data_path = '../../../data/interim/1/linear-reg/lr_train.csv'
test_data_path = '../../../data/interim/1/linear-reg/lr_test.csv'
val_data_path = '../../../data/interim/1/linear-reg/lr_val.csv'
figures_directory_1 = '../figures/linear_reg_plots/'
figures_directory_2 = '../figures/'

# # Load the data from the CSV file
# data = pd.read_csv(csv_dataset)

# # Shuffle the data
# data = data.sample(frac=1).reset_index(drop=True)
# # # Split the data into features (X) and target (y)
# # X_pts = data.iloc[:, :-1].values  # Assuming all columns except last are features
# # Y_pts = data.iloc[:, -1].values  # Assuming last column is the target

# train_size = int(0.8 * len(data))
# val_size = int(0.1 * len(data))


# train_data = pd.read_csv(train_data_path)
# val_data = pd.read_csv(val_data_path)
# val_data = data[train_size:train_size+val_size]
# test_data = data[train_size+val_size:]

# # Write the training data to the specified path
# train_data.to_csv(train_data_path, index=False)

# # Write the validation data to the specified path
# val_data.to_csv(val_data_path, index=False)

# # Write the test data to the specified path
# test_data.to_csv(test_data_path, index=False)

# # plot the training-test-validation data 
# # Initialize the model
# lr_deg1 = LinearRegression(reg_lambda=0, poly_degree=1, num_features=1, learning_rate=0.1, plot_save=0, num_iter=100, regularization=None)

# # Fit the model (assuming you have the paths to the train, validation, and test datasets)
# lr_deg1.fit(train_data_path, val_data_path, test_data_path, figures_directory_1)


# # Create the scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(lr_deg1.X_train[:, 1], lr_deg1.y_train, color='blue', label='Training Data')
# plt.scatter(lr_deg1.X_val[:, 1], lr_deg1.y_val, color='green', label='Validation Data')
# plt.scatter(lr_deg1.X_test[:, 1], lr_deg1.y_test, color='red', label='Test Data')
# plt.legend()

# # Save the plot
# plot_path = os.path.join(figures_directory_1, 'train_val_test_split.png')
# plt.savefig(plot_path)
# plt.close()

# # Calculate and print variance and standard deviation for train, test, and validation sets
# train_variance = np.var(lr_deg1.X_train)
# train_std_dev = np.std(lr_deg1.X_train)

# test_variance = np.var(lr_deg1.X_test)
# test_std_dev = np.std(lr_deg1.X_test)

# val_variance = np.var(lr_deg1.X_val)
# val_std_dev = np.std(lr_deg1.X_val)

# # Create a DataFrame to display the metrics
# metrics_df = pd.DataFrame({
#     'Data Set': ['Training', 'Validation', 'Test'],
#     'Variance': [train_variance, val_variance, test_variance],
#     'Standard Deviation': [train_std_dev, val_std_dev, test_std_dev]
# })

# # Print the metrics in tabular format
# print("Distribution of Data (Train-Test-Val):")
# print(metrics_df)
# print("-----------------------")
# print()


# # plotting the final line on test data 
# test_y_pred = np.dot(lr_deg1.X_test, lr_deg1.weights)
# plt.figure(figsize=(10, 6))
# plt.scatter(lr_deg1.X_train[:, 1], lr_deg1.y_train, color='blue', label='Training Data')

# # Plot the fitted line
# plt.plot(lr_deg1.X_test[:,1], test_y_pred, color='red', label=f'Fitted Line (Iteration {lr_deg1.num_iterations})')

# # plt.plot(self.X_train[:, 1], y_pred, color='red', label=f'Fitted Line (Iteration {iteration})')

# # Display metrics on the plot
# plt.title(f'Iteration {lr_deg1.num_iterations}\nMSE: {lr_deg1.mean_squared_error(lr_deg1.y_test, test_y_pred):.4f} | Variance: {lr_deg1.variance(lr_deg1.y_test):.4f} | Std Dev: {lr_deg1.standard_deviation(lr_deg1.y_test):.4f}')
# plt.xlabel('Feature')
# plt.ylabel('Target')
# plt.legend()
# plt.show()

# # Save the plot
# plot_path = os.path.join(figures_directory_1, 'Degree_1.png')
# plt.savefig(plot_path)
# plt.close()

# def save_best_model(lr_model):
#     # Sort grad_desc_iterations by MSE in ascending order
#     lr_model.grad_desc_iterations.sort(key=lambda x: x["MSE"])
    
#     # Get the best iteration (lowest MSE)
#     best_iteration = lr_model.grad_desc_iterations[0]
    
#     # File path for saving the best iteration details
#     output_file_path = 'best_iteration.txt'
    
#     # Open the file in append mode
#     with open(output_file_path, 'a') as outfile:
#         outfile.write(f"Polynomial Degree: {lr_model.poly_degree}\n")
#         outfile.write(f"Iteration Number: {best_iteration['Iteration_Num']}\n")
#         outfile.write(f"MSE: {best_iteration['MSE']:.4f}\n")
#         outfile.write(f"Variance: {best_iteration['Variance']:.4f}\n")
#         outfile.write(f"Standard Deviation: {best_iteration['Standard Deviation']:.4f}\n")
#         outfile.write("Weights:\n")
#         outfile.write("\n".join([f"  {weight:.4f}" for weight in best_iteration['Weights']]))
#         outfile.write("\n\n")  # Add a newline for separation between entries

# def make_animation(poly_degree):

#     # Function to extract numeric parts of the filename for sorting
#     def extract_number(filename):
#         # Extract all numeric parts from the filename
#         numbers = re.findall(r'\d+', os.path.basename(filename))
#         # Convert to integers for proper numeric comparison
#         return list(map(int, numbers))
#     # Initialize some settings
#     image_folder = figures_directory_2
#     output_gif_path = f'../../assignments/1/animation/new/without_seeding/animation_deg{poly_degree}.gif'

#     duration_per_frame = 100  # milliseconds

#     # Collect all image paths
#     image_paths = glob.glob(f"{image_folder}*.png")
#     image_paths.sort(key=extract_number)  # Sort the images to maintain sequence; adjust as needed
#     # print(image_paths)

#     # Initialize an empty list to store the images
#     frames = []

#     # # Debugging lines (moved here, after frames is initialized)
#     # print("Number of frames: ", len(frames))
#     # print("Image Paths: ", image_paths)

#     # Loop through each image file to add text and append to frames
#     for image_path in image_paths:
#         img = Image.open(image_path)

#         # Reduce the frame size by 50%
#         img = img.resize((int(img.width * 0.5), int(img.height * 0.5)))

#         # Create a new draw object after resizing
#         draw = ImageDraw.Draw(img)

#         # Text to display at top-left and bottom-right corners
#         top_left_text = image_path.split("/")[-1]
#         bottom_right_text = "Add your test here to be displayed on Images"

#         # Font settings
#         font_path = "/Library/Fonts/Arial.ttf"  # Replace with the path to a .ttf file on your system
#         font_size = 20
#         font = ImageFont.truetype(font_path, font_size)

#         # Draw top-left text
#         draw.text((10, 10), top_left_text, font=font, fill=(255, 255, 255))

#         # Calculate x, y position of the bottom-right text
#         text_width, text_height = draw.textsize(bottom_right_text, font=font)
#         x = img.width - text_width - 10  # 10 pixels from the right edge
#         y = img.height - text_height - 10  # 10 pixels from the bottom edge

#         # Draw bottom-right text
#         draw.text((x, y), bottom_right_text, font=font, fill=(255, 255, 255))

#         frames.append(img)

#     # Initialize video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
#     out = cv2.VideoWriter(f'lr_degree_{1}.mp4', fourcc, 20.0, (int(img.width), int(img.height)))

#     # Loop through each image frame (assuming you have the frames in 'frames' list)
#     for img_pil in frames:
#         # Convert PIL image to numpy array (OpenCV format)
#         img_np = np.array(img_pil)

#         # Convert RGB to BGR (OpenCV uses BGR instead of RGB)
#         img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

#         # Write frame to video
#         out.write(img_bgr)

#     # Release the VideoWriter
#     out.release()

#     # Save frames as an animated GIF
#     frames[0].save(output_gif_path,
#                 save_all=True,
#                 append_images=frames[1:],
#                 duration=duration_per_frame,
#                 loop=0,
#                 optimize=True)

# # lr_deg1 = LinearRegression(reg_lambda=0, poly_degree=1, num_features=1, learning_rate=0.1,plot_save=1, num_iter=150, regularization=None)
# # lr_deg1.fit(train_data_path, val_data_path, test_data_path, figures_directory_2)
# # save_best_model(lr_deg1)
# # make_animation(lr_deg1.poly_degree)

# # lr_deg2 = LinearRegression(reg_lambda=0, poly_degree=2, num_features=1, learning_rate=0.1,plot_save=1, num_iter=150, regularization=None)
# # lr_deg2.fit(train_data_path, val_data_path, test_data_path, figures_directory_2)
# # save_best_model(lr_deg2)
# # make_animation(lr_deg2.poly_degree)


# # lr_deg3 = LinearRegression(reg_lambda=0, poly_degree=3, num_features=1, learning_rate=0.1,plot_save=1, num_iter=150, regularization=None)
# # lr_deg3.fit(train_data_path, val_data_path, test_data_path, figures_directory_2)
# # save_best_model(lr_deg3)
# # make_animation(lr_deg3.poly_degree)


# # lr_deg4 = LinearRegression(reg_lambda=0, poly_degree=4, num_features=1, learning_rate=0.1,plot_save=1, num_iter=150, regularization=None)
# # lr_deg4.fit(train_data_path, val_data_path, test_data_path, figures_directory_2)
# # save_best_model(lr_deg4)
# # make_animation(lr_deg4.poly_degree)


# # lr_deg5 = LinearRegression(reg_lambda=0, poly_degree=5, num_features=1, learning_rate=0.1,plot_save=1, num_iter=150, regularization=None)
# # lr_deg5.fit(train_data_path, val_data_path, test_data_path, figures_directory_2)
# # save_best_model(lr_deg5)
# # make_animation(lr_deg5.poly_degree)

# Initialize lists to store metrics for each degree
degrees = []
mses = []
variances = []
std_devs = []
convergences = []

# # Loop over each degree from 1 to 20
# for deg in range(1, 21):
#     lr_deg = LinearRegression(reg_lambda=0.1, poly_degree=deg, num_features=1, learning_rate=0.1, plot_save=0, num_iter=150, regularization=None)
#     lr_deg.fit(train_data_path, val_data_path, test_data_path, figures_directory_2)
    
#     # Calculate predictions on the test set
#     test_y_pred = np.dot(lr_deg.X_test, lr_deg.weights)
    
#     # Calculate metrics
#     mse = lr_deg.mean_squared_error(lr_deg.y_test, test_y_pred)
#     var = lr_deg.variance(test_y_pred)
#     std_dev = lr_deg.standard_deviation(test_y_pred)
#     convergence = lr_deg.grad_desc_iterations[-1]['Iteration_Num'] if lr_deg.grad_desc_iterations else None
    
#     # Store metrics in lists
#     degrees.append(deg)
#     mses.append(mse)
#     variances.append(var)
#     std_devs.append(std_dev)
#     convergences.append(convergence)
    
# # Create a DataFrame for easy viewing
# results_df = pd.DataFrame({
#     'Degree': degrees,
#     'MSE': mses,
#     'Variance': variances,
#     'Std Dev': std_devs,
#     'Convergence Iteration': convergences
# })

# # Print the DataFrame
# print(results_df)
# print("-----------------------")
# print()

# # Find the degree with the minimum MSE
# min_mse_degree = results_df.loc[results_df['MSE'].idxmin()]

# print(f"The degree with the minimum MSE is {min_mse_degree['Degree']} with an MSE of {min_mse_degree['MSE']:.4f}")
# print()
# # Plot MSE vs Degree
# plt.figure(figsize=(10, 6))
# plt.plot(degrees, mses, marker='o', linestyle='-', color='blue')
# plt.title('MSE vs Polynomial Degree')
# plt.xlabel('Polynomial Degree')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.xticks(degrees)
# plt.grid(True)

# # Save the plot
# plot_path = os.path.join(figures_directory_1, 'MSE_vs_Degree.png')
# plt.savefig(plot_path)
# plt.close()

# adjusted_degrees = list(range(5,21))


train_data_path = '../../../data/interim/1/reg/regu_train.csv'
test_data_path = '../../../data/interim/1/reg/regu_test.csv'
val_data_path = '../../../data/interim/1/reg/regu_val.csv'

train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)


new_variances = []
# Loop over each degree from 1 to 20
for deg in range(5, 21):
    lr_deg = LinearRegression(reg_lambda=0.1, poly_degree=deg, num_features=1, learning_rate=0.1, plot_save=0, num_iter=150, regularization=None)
    lr_deg.fit(train_data_path, val_data_path, test_data_path, figures_directory_2)
    
    # Calculate predictions on the test set
    test_y_pred = np.dot(lr_deg.X_test, lr_deg.weights)
    
    # Calculate metrics
    mse = lr_deg.mean_squared_error(lr_deg.y_test, test_y_pred)
    var = lr_deg.variance(test_y_pred)
    std_dev = lr_deg.standard_deviation(test_y_pred)
    convergence = lr_deg.grad_desc_iterations[-1]['Iteration_Num'] if lr_deg.grad_desc_iterations else None
    
    # Store metrics in lists
    degrees.append(deg)
    mses.append(mse)
    new_variances.append(var)
    std_devs.append(std_dev)
    convergences.append(convergence)



train_data_path = '../../../data/interim/1/reg/regu_train.csv'
test_data_path = '../../../data/interim/1/reg/regu_test.csv'
val_data_path = '../../../data/interim/1/reg/regu_val.csv'

adjusted_degrees = list(range(5,21))
variance_L1 = []
variance_L2 = []

# L1 REGULARIZATION 
for deg in range(5, 21):
    lr_regu = LinearRegression(reg_lambda=0.1, poly_degree=deg, num_features=1, learning_rate=0.1, plot_save=0, num_iter=150, regularization="L1")
    lr_regu.fit(train_data_path, test_data_path, val_data_path, figures_directory_1)
    
    # Calculate predictions on the test set
    test_y_pred = np.dot(lr_regu.X_test, lr_regu.weights)

    # Calculate and display metrics
    mse = lr_regu.mean_squared_error(lr_regu.y_test, test_y_pred)
    var = lr_regu.variance(test_y_pred)
    variance_L1.append(var)
    std_dev = lr_regu.standard_deviation(test_y_pred)
    
    # plt.xlabel('Feature')
    # plt.ylabel('Target')
    # plt.legend()
    plt.scatter(lr_regu.X_train[:, 1], lr_regu.y_train, color='blue', label='Training Data')
    x_range = np.linspace(min(lr_regu.X_train[:, 1]), max(lr_regu.X_train[:, 1]), 10000)
    X_poly = np.vander(x_range, N=lr_regu.poly_degree + 1, increasing=True)
    y_range_pred = np.dot(X_poly, lr_regu.weights)
    plt.plot(x_range, y_range_pred, color='red', label=f'Fitted Line')
    plt.title('Training Data and Fitted Line')
    plt.title(f'Degree {lr_regu.poly_degree}\nMSE: {mse:.4f} | Variance: {var:.4f} | Std Dev: {std_dev:.4f}')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    # Save the plot
    plot_path = os.path.join(figures_directory_1, f'L1_Regularised_Degree_{deg}.png')
    plt.savefig(plot_path)
    plt.close()




# L2 REGULARIZATION 
for deg in range(5, 21):
    lr_regu = LinearRegression(reg_lambda=0.1, poly_degree=deg, num_features=1, learning_rate=0.1, plot_save=0, num_iter=150, regularization="L2")
    lr_regu.fit(train_data_path, test_data_path, val_data_path, figures_directory_1)
    
    # Calculate predictions on the test set
    test_y_pred = np.dot(lr_regu.X_test, lr_regu.weights)

    # Calculate and display metrics
    mse = lr_regu.mean_squared_error(lr_regu.y_test, test_y_pred)
    var = lr_regu.variance(test_y_pred)
    variance_L2.append(var)
    std_dev = lr_regu.standard_deviation(test_y_pred)
    
    # plt.xlabel('Feature')
    # plt.ylabel('Target')
    # plt.legend()
    plt.scatter(lr_regu.X_train[:, 1], lr_regu.y_train, color='blue', label='Training Data')
    x_range = np.linspace(min(lr_regu.X_train[:, 1]), max(lr_regu.X_train[:, 1]), 10000)
    X_poly = np.vander(x_range, N=lr_regu.poly_degree + 1, increasing=True)
    y_range_pred = np.dot(X_poly, lr_regu.weights)
    plt.plot(x_range, y_range_pred, color='red', label=f'Fitted Line')
    plt.title('Training Data and Fitted Line')
    plt.title(f'Degree {lr_regu.poly_degree}\nMSE: {mse:.4f} | Variance: {var:.4f} | Std Dev: {std_dev:.4f}')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    # Save the plot
    plot_path = os.path.join(figures_directory_1, f'L2_Regularised_Degree_{deg}.png')
    plt.savefig(plot_path)
    plt.close()

# adjusted_degrees = degrees[4:]  # degrees from 5 to 20

# variance_unregularised = variances[4:]

# Create a DataFrame to display the results
variance_df = pd.DataFrame({
    'Degree': adjusted_degrees,
    'Variance (Unregularized)': new_variances,
    'Variance (L1)': variance_L1,
    'Variance (L2)': variance_L2
})

print(variance_df)
print("----------------------------")
print()


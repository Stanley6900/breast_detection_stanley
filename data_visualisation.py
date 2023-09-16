import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv   
import plotly.io as pio
import os
 

# This function shows the bar plot of the breast cancer data.
def show_bar_plot(breast_data, folder, image_name):
    """ This function display a bar plot of the breast tumor classes.
    Args:
        breast_data (dataframe): A dataframe that contains the breast data info.
        folder (file): Represents the folder path
        image_name (str): A string which represents the image name.
    """
    class_count = pd.DataFrame(breast_data['pathology'].value_counts())
    class_count = class_count.reset_index()
    # class_count = class_count.rename(columns={'index':'pathology','pathology':'counts'})

    bar_plot =px.bar(data_frame=class_count, x='pathology', y='count', color = 'pathology', orientation='v')
    bar_plot.update_layout(title_text='Distribution of mass cancer', title_x=0.45)
    bar_plot.show()
    # Specify the image format as PNG
    image_format = 'png'
    
    # Construct the full image path with the specified format
    full_path = os.path.join(folder, f'{image_name}.{image_format}')

    pio.write_image(bar_plot, full_path)
    
    
# This function shows the pie chart of the breast cancer data.
def show_pie_plot(breast_data, folder, image_name):
    """ This function display a pie plot of the breast tumor classes.
    Args:
        breast_data (dataframe): A dataframe that contains the breast data info.
        folder (file): Represents the folder path
        image_name (str): A string which represents the image name.
    """
    class_count = pd.DataFrame(breast_data['pathology'].value_counts())
    class_count = class_count.reset_index()
    pie_plot = px.pie(data_frame = class_count, names= 'pathology', values='count', color = 'pathology')
    pie_plot.update_layout(title_text = 'The percentages of mass cancer', title_x=0.45)
    pie_plot.show()
    
    # Specify the image format as PNG
    image_format = 'png'
    
    # Construct the full image path with the specified format
    full_path = os.path.join(folder, f'{image_name}.{image_format}')

    pio.write_image(pie_plot, full_path)


# This function plots the confusion matrix of the data.
def plot_matrix(y_true, y_pred, type_split):
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix ' + type_split)
    plt.show()
    
def show_image(axis, image, title):
    """
    This function displays the image.
    Args:
        image: represents the image to be displayed.
        axis: represents the axis to display the image, since we are displaying it in a grid.
        title: represents the title of the image to be displayed.
    """

    img = cv.imread(image, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img,(400,400))
    axis.imshow(img)
    axis.axis('off')
    axis.set_title(title)


def show_image_roi(axis, image, title):
    """
    This function displays the image for the mask mammogram dataset.
    Args:
        axis: represents the axis to be plotted. The axis can be on a specific row or column
        image: represents the image to be displayed.
        title: represents the title of the image to be displayed or plotted.
    """
    img = cv.imread(image, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img,(400,400))
    axis.imshow(img, cmap='gray')
    axis.axis('off')
    axis.set_title(title)


# lung-cancer-image-classification
 Lung cancer image classification in Python using LIDC dataset. 

## Project Objective
- To identify the best local feature extraction method for lung cancer classification
- To develop a model for lung cancer classification
- To develop a prototype of image classification tool to categorize malignant and benign lung nodules

### Methods Used
* Image Transformation
* Dimension Reduction
* Machine Learning

### Technologies
* Python
* Python scikit-learn
* Python pandas, flask
* Jupyter

## Project Description
* `config.py` - global variables
* `preporcessing.py` - preprocessing methods
* `image_processing.py` - image transformations methods
* `import_data.py` - read and convert raw data
* `data_lidc.py` - generates features from LIDC dataset
* `main.py` - train models
* `Models Comparison.ipynb` - models comparison

Data source from [cancerimagingarchive.net](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI). 1018 cases of labelled CT scans.

## Process Flow
- frontend development
- data collection
- data processing/cleaning
- image transformation
- model training
- writeup/reporting

## Future Improvements
This is my first time experimenting on a large dataset. Make use of data pipeline for clean and reusable codes. Try on hadoop to handle insufficient memory.

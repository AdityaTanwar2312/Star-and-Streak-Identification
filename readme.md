### Before runnig the script
Ensure the given directory structure before running the script

your_project/<br>
├── R_CNN.ipynb<br>
├── annotation.ipynb<br>
├── data_preprocessing.ipynb<br>
├── requirements.txt

├── model_output/<br>
│   ├── model_checkpoint.pth<br>
│   └── training_log.csv<br>

├── Datasets/<br>
│   ├── Raw_Images/<br>
│   └── Reference_Images/<br>

└── annotation_data/<br>
    

You just need to ensure the given directories and naming. The CSV files and model_checkpoint.pth will be automatically generated when the Script is run

- Running Script:-
    - Setup the virtual environment using python -m venv "virtual environment name"
    - Ensure you download all dependencies in requirements.txt
    - Open jupyter notebooks and run in the given order
        - data_preprocessing.ipynb
        - annotation.ipynb
        - R_CNN.ipynb

### 1. Data Preprocessing and Enhancement
The initial step in the pipeline involves preprocessing raw astronomical images to enhance visual features such as stars and streaks. This is crucial for improving annotation quality and aiding the model in learning meaningful patterns.

#### 1. Raw Data Source
-	Input Format: Grayscale .tiff images.

#### 2. Enhancement Technique
To improve the contrast and highlight subtle features in astronomical imagery, Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied. CLAHE is particularly effective in low-light or high-contrast conditions, common in astronomical datasets.

-	CLAHE Parameters:
    -	clipLimit = 3.0: Prevents over-amplification of noise.
    -	tileGridSize = (8, 8): Applies localized histogram equalization.

#### 3. Visualization and Storage
-	Enhanced images are visualized using with grayscale and saved as high-quality .png plots for inspection.

<img width="400" height="400" alt="image" src="images/Raw_Observation_035_Set4_enhanced_plot.png" />  
<img width="400" height="400" alt="image" src="images/Raw_Observation_015_Set2_enhanced_plot.png" />  

### 2. Automated Annotation Generation
To prepare data for supervised learning, annotations indicating the locations of stars and streaks were generated through an automated image processing pipeline. This eliminates manual annotation and ensures consistency across the dataset.

#### 1. Image Input
-	Enhanced grayscale images produced during preprocessing.

#### 2. Processing Pipeline
1. High-Pass Filtering
    - Applied a Gaussian blur (σ = 2) and subtracted it from the original image to isolate high-frequency components such as edges and streaks.
2. Contrast Enhancement
    - Used equalize_adapthist for localized histogram equalization, improving edge definition for faint streaks and stars.
3. Edge Detection
    - Performed Canny edge detection to extract crisp boundaries of objects in the image.
4. Morphological Operations
    - Applied dilation using a disk structuring element to close small gaps in detected streaks, improving region connectivity.

#### 3. Region Labeling and Annotation
Using skimage.measure.label and regionprops, connected regions in the processed binary mask were analyzed:
- Streaks: Identified as elongated regions with area > 40. Bounding rectangles were drawn.
- Stars: Smaller regions with area < 30. Annotated using circular overlays based on the calculated centroid and area.

This heuristic approach effectively separates small point-like sources (stars) from elongated streaks, preparing them for detection/classification tasks.

#### 4. Output
- Annotated visualizations are saved with bounding shapes overlaid on the original image.
- Format: PNG, preserving filename for traceability.
 
### 3. Annotation Export to CSV Format
To streamline model training, the detected star and streak annotations were exported in a structured CSV format compatible with object detection pipelines

#### Processing Steps
Each image undergoes the following process:

#### 1. Repetition of Preprocessing:
The image is reprocessed with the same high-pass filtering, contrast enhancement, and edge detection used during visualization. This ensures consistent feature extraction.

#### 2. Region Analysis:
Each connected region is analyzed using regionprops. Based on area, each region is labeled as:
- Star: 
Regions with area < 30. Centroid is used to estimate a circular region, which is then converted into an approximate bounding box.
- Streak: 
Elongated regions with area > 40. Bounding box is derived directly from the region's bounding rectangle.

<img width="920" alt="image" src="annotation_data/image.png" />
<img width="920" alt="image" src="annotation_data/image2.png" />  
<img width="920" alt="image" src="annotation_data/image3.png" /> 

### 4. Model Training and Evaluation
#### 1. Model Architecture
The model used is Faster R-CNN with a ResNet-50 FPN backbone, pretrained on COCO. The classification head is replaced to suit our 3-class problem (background, star, streak)

#### 2. Training Procedure
- Optimizer: Adam optimizer with learning rate = 0.0005
- LR Scheduler: Reduces LR when validation loss plateaus (patience=2, factor=0.1)
- Loss Function: Automatically computed by FasterRCNN; it returns a dictionary of losses (classification, bbox regression, objectness, etc.)
- Epochs: Maximum 10
- Early Stopping: Monitors validation loss, stops if no improvement for 5 consecutive epochs

#### 3. Epoch Training Function
Each epoch performs the following:
- Iterates over the training loader
- Computes the total loss for the batch
- Logs batch loss using tqdm

#### 4. Evaluation Function
- Computes and logs total loss
- Sets the model to train mode during loss computation, since FasterRCNN only returns losses in train mode

#### 5. Checkpointing and Logging
- Best model based on validation loss is saved
- Training and validation losses per epoch are logged to training_log.csv

#### Model Training results:-

Training: Detection Accuracy: 59.70%

Testing: Detection Accuracy: 75.00%

#### Train Loss: 0.1689, Validation Loss: 0.0862

### 6. Inference and Prediction
#### 1. Prediction Pipeline
The trained Faster R-CNN model is used to detect and classify stars and streaks in unseen astronomical images. The inference pipeline performs the following tasks:
- Image Loading & Preprocessing
    - Converts image to RGB
    - Transforms it into a PyTorch tensor
- Model Inference
    - The image is passed through the model in evaluation mode
    - Output contains predicted bounding boxes, class labels, and confidence scores
- Prediction Filtering
    - Detections with confidence scores ≥ 0.5 are retained
    - Class labels are mapped to human-readable format

<img width="500" height="500" alt="image" src="images/Raw_Observation_025_Set3_enhanced_plot_annotated.png" /> 

<img width="500" height="500" alt="image" src="images/Raw_Observation_035_Set4_enhanced_plot_annotated.png" />

#### 2. Centroid Calculation
For each valid bounding box:
- The centroid is computed using:
- Coordinates are printed with corresponding label and confidence score

<img width="500" height="500" alt="image" src="images/Raw_Observation_001_Set1_enhanced_plot_annotated.png" /> 

#### Detected Objects with Centroid Coordinates:
- Star (Score: 0.99) -> Centroid: (x=351.50, y=387.20)
- Star (Score: 0.97) -> Centroid: (x=148.10, y=106.87)
- Star (Score: 0.96) -> Centroid: (x=526.77, y=474.30)
- Streak (Score: 0.94) -> Centroid: (x=153.10, y=199.76)
- Star (Score: 0.85) -> Centroid: (x=441.34, y=521.68)

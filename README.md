# Eye Disease Classifier
Eye Disease Classifier is a machine learning project designed to automatically detect and classify common eye diseases from retinal images.

![image](https://github.com/user-attachments/assets/ffaffdd4-a49c-4055-98f9-308f457e7bc8)

## Live Demo
You can try the live demo of the Eye Disease Classifier here: [Eye Disease Classifier Demo](https://eye-disease-classifier-taiga.streamlit.app)

## Dataset
Kaggle: [Eye Disease Retinal Images](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data?select=dataset)

## Classes
- Cataract
- Diabetic Retinopathy
- Glaucoma
- Normal

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- (Optional) CUDA-enabled GPU for faster model training/inference

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TaigaTi/Eye-Disease-Classifier.git
   cd Eye-Disease-Classifier
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

### Training the Model
```bash
python main.py
```

## Project Structure

```
Eye-Disease-Classifier/
│
├── models/                              # Trained models
├── confusion_matrices/                  # Confusion Matrices
├── main.py                              # Main script
├── eye_classification_results.xlsx      # Results from model training
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

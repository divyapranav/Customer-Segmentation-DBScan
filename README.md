# Mall Customer Segmentation using DBSCAN

This project performs **customer segmentation** on the Mall Customers dataset using the **DBSCAN clustering algorithm**.  
It is deployed as a **Flask web application**, allowing users to input a customer's **Annual Income** and **Spending Score** to find out which cluster they belong to.

---

## Project Overview
The project uses:
- **DBSCAN** for unsupervised customer segmentation.
- **StandardScaler** to normalize input features.
- **Nearest Neighbors** to assign new incoming customers to the nearest cluster (since DBSCAN cannot predict new samples directly).
- **Flask** for building a web interface.

---

## Project Structure
├── app.py # Flask application
├── model_training.py # Script to train DBSCAN & save models
├── scaler.pkl # Saved StandardScaler
├── dbscan.pkl # Saved DBSCAN model
├── nn.pkl # Saved NearestNeighbors model
├── labels.pkl # Saved cluster labels from training
├── templates
│ └── index.html # HTML form for user input
├── Mall_Customers.csv # Dataset
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/divyapranav/Customer-Segmentation-DBScan.git
cd Customer-Segmentation-DBScan
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python model.py
```

### Run the Flask app
```bash
python app.py
```

### Open Browser
Go to:
```cpp
http://127.0.0.1:5000/
```

## Usage
1. Enter Annual Income and Spending Score in the form.
2. Click Predict Cluster.
3. The application will display the cluster number assigned to the customer.

## Notes
DBSCAN labels -1 represent noise (outliers).
eps and min_samples in DBSCAN can be tuned for better results.


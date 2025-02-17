# E-commerce Transaction Analysis  

## Project Overview  
This project focuses on analyzing an eCommerce dataset using **RFM (Recency, Frequency, Monetary) analysis** for **customer segmentation**. RFM is a widely used technique to classify customers based on their recent transactions, purchase frequency, and spending behavior. The resulting segmentation helps businesses **refine marketing strategies, enhance customer engagement, and improve retention efforts**.  

## Dataset Overview  
The dataset contains **541,909 records** across **8 columns**, detailing transaction-level information:  

- **InvoiceNo**: Unique transaction identifier (an invoice can include multiple products).  
- **StockCode**: Product codes corresponding to purchased items.  
- **Description**: Product descriptions (some missing values).  
- **Quantity**: Number of units bought per transaction.  
- **InvoiceDate**: Date and time of the transaction.  
- **UnitPrice**: Price per unit of each product.  
- **CustomerID**: Unique customer identifier (some missing values).  
- **Country**: Country where the transaction occurred.  

## Objective  
The primary goal of this project is to perform **RFM-based customer segmentation**, identifying different customer groups based on purchasing patterns. These insights enable businesses to:  
- Personalize marketing campaigns.  
- Improve customer retention.  
- Optimize pricing and promotions.  

## Technologies Used  
The project is implemented in **Python**, utilizing various libraries for **data processing, visualization, machine learning, and customer segmentation**.  

### Libraries Used  
- **Data Manipulation & Analysis**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Machine Learning**: `scikit-learn`, `catboost`  
- **Outlier Detection**: `LocalOutlierFactor`  
- **Sampling Techniques**: `imblearn` (SMOTE, over/under-sampling)  
- **Hyperparameter Tuning**: `GridSearchCV`  
- **Classification Models**: `RandomForestClassifier`, `GradientBoostingClassifier`, `DecisionTreeClassifier`, `MLPClassifier`, `LogisticRegression`  

## Implementation Steps  
1. **Data Preprocessing**  
   - Handle missing values in `CustomerID` and `Description`.  
   - Remove duplicates and filter out invalid transactions.  
2. **Feature Engineering**  
   - Calculate RFM scores for each customer.  
   - Scale and normalize data where necessary.  
3. **Customer Segmentation**  
   - Cluster customers based on their RFM scores.  
   - Label different customer groups for targeted marketing.  
4. **Model Training & Evaluation**  
   - Train multiple classification models.  
   - Evaluate model performance using **accuracy, F1-score, ROC-AUC, and confusion matrices**.  
   - Perform hyperparameter tuning using **GridSearchCV**.  

## Results & Insights  
- Customers are **grouped into meaningful segments** based on RFM scores.  
- **High-value customers** (frequent buyers with high monetary contributions) can be **targeted for loyalty programs**.  
- **Churn-prone customers** (low recency and frequency scores) can be **re-engaged through promotional offers**.  

## Future Improvements  
- Incorporate **clustering algorithms (K-Means, DBSCAN)** for enhanced segmentation.  
- Apply **deep learning models** for customer behavior prediction.  
- Develop a **real-time dashboard** to visualize RFM trends dynamically.  

## Usage  
To run this project:  
1. Install dependencies:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn catboost imbalanced-learn
   ```  
2. Load the dataset and execute the preprocessing script.  
3. Run the RFM analysis and machine learning models.  
4. Visualize and interpret segmentation results.  

## Conclusion  
This project provides **valuable insights into customer purchasing behavior** using **RFM analysis and machine learning models**. Businesses can leverage these insights to **enhance customer relationships, boost sales, and improve retention strategies**.  

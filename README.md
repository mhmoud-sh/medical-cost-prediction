# Medical Cost Prediction with SHAP Visualization

This project is an end-to-end pipeline that predicts medical costs using machine learning and provides SHAP (SHapley Additive exPlanations) visualizations to explain the model’s predictions. It leverages a trained XGBoost model and a preprocessor pipeline to predict healthcare expenses based on various features like age, BMI, smoking status, and more.

## Features
- **Medical Cost Prediction**: The model predicts the medical cost (`charges`) for each individual in the dataset based on their health and personal information.
- **SHAP Visualizations**: Displays feature importance, beeswarm plots, and waterfall plots to explain how the model makes predictions.
- **Custom Prediction**: Users can input custom data (e.g., age, BMI, smoking status) to get real-time predictions along with explanation visualizations.
- **Prediction Log**: Predictions are logged to a CSV file for analysis and record-keeping.

## Technologies Used
- **Python** (3.7+)
- **Streamlit**: For building the interactive web app.
- **XGBoost**: Machine learning model for predicting medical costs.
- **SHAP**: For interpreting and explaining the predictions.
- **Joblib**: For saving and loading the model and preprocessor.
- **Pandas**: For data manipulation and processing.
- **Matplotlib**: For creating visualizations.

## Requirements
To run the app locally, make sure you have the following installed:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/medical-cost-prediction.git
   cd medical-cost-prediction



How to Use
Upload a CSV or Use Default Data: In the sidebar, you can either upload a custom dataset (CSV) or use the default medical dataset (medical.csv) to visualize predictions.

Custom Predictions: Enter personal information (age, sex, BMI, smoking status, etc.) in the custom input section to get a prediction for that individual’s medical costs.

SHAP Explanation: After making predictions, the app will display SHAP visualizations like feature importance, beeswarm plots, and waterfall plots to explain how the model arrived at the prediction.

Save Predictions: Every prediction made through the app is saved in logs/custom_predictions_log.csv for future analysis.

Contributing
Feel free to fork this repository, open issues, and submit pull requests. Whether you're fixing bugs, improving the performance, or adding new features, all contributions are welcome!

Made with 💙 by shikheldin


### Key Sections in the `README.md`:
1. **Project Overview**: A brief introduction to the project.
2. **Features**: Key functionality the project offers.
3. **Technologies Used**: The tools and libraries used.
4. **Installation Instructions**: Detailed steps on how to get the app running locally.
5. **Project Structure**: Overview of the project files and folders.
6. **How to Use**: Instructions on how users can interact with the app.
7. **Contributing**: Information on how others can contribute to the project.
8. **License**: Licensing information for others to use the code.


## 🌐 Live Demo
Check out the deployed app here: [Medical Cost Predictor](https://medical-cost-prediction65487654.streamlit.app/)


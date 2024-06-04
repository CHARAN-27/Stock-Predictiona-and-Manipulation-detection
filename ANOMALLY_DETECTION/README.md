The Recommendation happens in Pred.ipynb file . Here we are implementing the code to predict the future 15 days for the given list of companies.
we are using the stock prediction model creation files to create the model and use in pred.ipynb file for future prediction
the lstm_autoenocder.ipynb file has the model and is used for training purposes.
In the price_manipulation_dtection.ipynb file,we are using this model for testing purposes.
We have the manpiulated stock list in a folder.These are csv files.we added a new column named anonmaly for these csv ,the anomaly column contains 0's and 1's. anomaly 1 indicates that the stock is manipulatedon that date.
We get the manipulated stock list from internet and their manipulation period.For present we used only five datasets for testing.
In the recommendation ,the list of the stocks that we are predicting are sent to the finding_anomalies.ipynb file to detect any anomalies fo rthose stocks in the past 6 six months.These period can be adjusted in the code.
We get the list of dates on which the company data is anomalous and this list is sent to pred.ipynb file.
In recommendation ,we remove the companies which got predicted as anomalous by our model . The recommendation is based on prediction and anomaly detection.

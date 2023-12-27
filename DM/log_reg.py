import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


url = "https://github.com/JWarmenhoven/ISLR-python/raw/master/Notebooks/Data/Credit.csv"
data = pd.read_csv(url)

data['Income_Greater_50'] = (data['Income'] > 50).astype(int)

income_predictors = ['Limit', 'Rating', 'Cards', 'Age', 'Education', 'Balance']
X_income = data[income_predictors]
y_income = data['Income_Greater_50']

credit_card_predictors = ['Income', 'Limit', 'Rating', 'Age', 'Education', 'Balance']
X_credit_card = data[credit_card_predictors]
y_credit_card = data['Cards']

X_income_train, X_income_test, y_income_train, y_income_test = train_test_split(X_income, y_income, test_size=0.2, random_state=42)
X_credit_card_train, X_credit_card_test, y_credit_card_train, y_credit_card_test = train_test_split(X_credit_card, y_credit_card, test_size=0.2, random_state=42)

logreg_income = LogisticRegression(max_iter=10000, solver='saga', random_state=42)
logreg_credit_card = LogisticRegression(max_iter=10000, solver='saga', random_state=42)

logreg_income.fit(X_income_train, y_income_train)
logreg_credit_card.fit(X_credit_card_train, y_credit_card_train)

y_income_pred = logreg_income.predict(X_income_test)
y_credit_card_pred = logreg_credit_card.predict(X_credit_card_test)

accuracy_income = accuracy_score(y_income_test, y_income_pred)
accuracy_credit_card = accuracy_score(y_credit_card_test, y_credit_card_pred)

print(f'Logistic Regression Accuracy (Income Prediction): {accuracy_income:.2f}')
print(f'Logistic Regression Accuracy (Credit Card Prediction): {accuracy_credit_card:.2f}')

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

penguins_df = pd.read_csv('penguins.csv')

penguins_df = penguins_df.dropna()

output = penguins_df['species']
features = penguins_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)

X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=.8, stratify=output, random_state=8)

rfc = RandomForestClassifier(random_state=8)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

score = accuracy_score(y_pred, y_test)

print("Our accuracy for this model is {}".format(score))

rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

fig, ax = plt.subplots() 
ax = sns.barplot(rfc.feature_importances_, features.columns) 
plt.title('Which features are the most important for species prediction?') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.tight_layout() 
fig.savefig('feature_importance.png')
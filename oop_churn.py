import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
import pickle

#designing OOP for the best model
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path 
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

        
# ModelHandler Class
class ModelHandler:
    def __init__(self, data, input_data, output_data):
        self.data = data
        self.input_data = input_data 
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
        self.x_train_encoded, self.x_test_encoded, self.x_train_scaled, self.x_test_scaled, self.x_train_smote, self.y_train_smote = [None] * 6

    def split_data(self, test_size=0.2, random_state=0):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def fillna(self, col):
        self.x_train[col].fillna(self.x_train[col].mode()[0], inplace=True)
        self.x_test[col].fillna(self.x_test[col].mode()[0], inplace=True)

    def remove(self,list_col):
        self.x_train.drop(columns=list_col, inplace=True)
        self.x_test.drop(columns=list_col, inplace=True)

    def feature_encode(self,list_col):
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

        self.x_train = self.x_train.reset_index(drop=True)
        encoded_features = one_hot_encoder.fit_transform(self.x_train[list_col])
        self.x_train_encoded = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(list_col))
        self.x_train_encoded = pd.concat([self.x_train, self.x_train_encoded], axis=1).drop(list_col, axis=1)

        self.x_test = self.x_test.reset_index(drop=True)
        encoded_features = one_hot_encoder.transform(self.x_test[list_col])
        self.x_test_encoded = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(list_col))
        self.x_test_encoded = pd.concat([self.x_test, self.x_test_encoded], axis=1).drop(list_col, axis=1)
    
    def scale(self):
        scaler = RobustScaler()
        self.x_train_scaled = scaler.fit_transform(self.x_train_encoded)
        self.x_test_scaled = scaler.transform(self.x_test_encoded)

        self.x_train_scaled = pd.DataFrame(self.x_train_scaled, columns=self.x_train_encoded.columns)
        self.x_test_scaled = pd.DataFrame(self.x_test_scaled, columns=self.x_test_encoded.columns)
        print("anjjjjjj:", self.x_train_scaled.columns)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test_scaled) 
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict))
               
    def createModel(self):
        self.model = GradientBoostingClassifier(learning_rate= 0.1,max_depth= 5, min_samples_leaf= 4, min_samples_split= 2, n_estimators= 100) 

    def train_model(self):
        smote = SMOTE(random_state=0)
        self.x_train_smote, self.y_train_smote = smote.fit_resample(self.x_train_scaled, self.y_train)
        self.model.fit(self.x_train_smote, self.y_train_smote)

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  
            pickle.dump(self.model, file) 


#data handling
file_path = "churn.csv"
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('churn')
data = data_handler.data
input_df = data_handler.input_df
output_df = data_handler.output_df

#model handling
model_handler = ModelHandler(data, input_df, output_df)
model_handler.split_data()
remove_list=["Unnamed: 0", "id", "CustomerId", "Surname"]
model_handler.fillna("CreditScore")
model_handler.remove(remove_list)
encode_list = ['Geography', 'Gender']
model_handler.feature_encode(encode_list)
model_handler.scale()

print("The best model)")
model_handler.train_model()
model_handler.makePrediction()
model_handler.createReport()

model_handler.save_model_to_file('trained_model.pkl') 










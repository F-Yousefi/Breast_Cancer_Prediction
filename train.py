import os
from optparse import OptionParser
from sklearn.ensemble import RandomForestClassifier
import warnings
from joblib import load
from dataset.dataset import Dataset
from utils import remove_outliers, transform_to_norm_dist, maxmin_norm, test_train_spliter
warnings.filterwarnings('ignore')


parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path",
                  help="Path to training data.", default=None)

(options, args) = parser.parse_args()


if options.train_path is not None:
    if os.path.isfile(options.train_path):
        TRAIN_DATA_PATH = options.train_path
    else:
        raise FileNotFoundError(

            f": Sorry, .csv file cannot be found in the path {options.train_path}")
else:
    TRAIN_DATA_PATH = './dataset/breast-cancer-wisconsin-data/data.csv'

dataset = Dataset(TRAIN_DATA_PATH)
print(
    f"The dataset placed in {TRAIN_DATA_PATH} has been extracted successfully!")
print(
    f"The dataset comprises {dataset.shape[0]} various reports based on real cases of breast cancer.")
dataset.dropna()
dataset.to_numerical()
x_train, y_train = dataset.generate()
x_train, y_train = remove_outliers(x_train, y_train)
print(
    f"{dataset.shape[0] - x_train.shape[0]} outliers are detected and removed.")
x_train = transform_to_norm_dist(x_train)
x_train = maxmin_norm(x_train)
x_train, x_test, y_train, y_test = test_train_spliter(x_train, y_train)


model = RandomForestClassifier(
    n_estimators=40,
    min_samples_split=2,
    max_depth=10,
    max_features=2,
    bootstrap=True,
    min_samples_leaf=1)
model.fit(x_train, y_train)
print(
    f"Random Forest Classifier Train Accuracy: {model.score(x_train,y_train)*100:.2f}%")
print(
    f"Random Forest Classifier Test Accuracy: {model.score(x_test,y_test)*100:.2f}%")

model, x_train, x_test, y_train, y_test = load('./best_model/RFC.joblib')
print(
    f"\nThe Best RFC Train Accuracy: {model.score(x_train,y_train)*100:.2f}%")
print(f"The Best RFC Test Accuracy: {model.score(x_test,y_test)*100:.2f}%")

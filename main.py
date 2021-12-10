import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
import statistics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics


def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:
    import numpy as np
    from pathlib import Path
    from pandas import DataFrame

    COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
    if kfold_scores.shape != (1, 6):
        raise ValueError("DataFrame must have 1 row and 6 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

    if np.min(df.values) < 0 or np.max(df.values) > 0.10:
        raise ValueError(
            "Your K-Fold error rates are too extreme. Ensure they are the raw error rates,\r\n"
            "and NOT percentage error rates. Also ensure your DataFrame contains error rates,\r\n"
            "and not accuracies. If you are sure you have not made either of the above mistakes,\r\n"
            "there is probably something else wrong with your code. Contact the TA for help.\r\n"
        )

    if df.loc["err", "svm_linear"] > 0.07:
        raise ValueError("Your svm_linear error rate is too high. There is likely an error in your code.")
    if df.loc["err", "svm_rbf"] > 0.03:
        raise ValueError("Your svm_rbf error rate is too high. There is likely an error in your code.")
    if df.loc["err", "rf"] > 0.05:
        raise ValueError("Your Random Forest error rate is too high. There is likely an error in your code.")
    if df.loc["err", ["knn1", "knn5", "knn10"]].min() > 0.04:
        raise ValueError("One of your KNN error rates is too high. There is likely an error in your code.")

    outfile = Path(__file__).resolve().parent / "kfold_mnist.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")


def save_data_kfold(kfold_scores: pd.DataFrame) -> None:
    import numpy as np
    from pandas import DataFrame
    from pathlib import Path

    COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
    if kfold_scores.shape != (1, 6):
        raise ValueError("DataFrame must have 1 row and 6 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

    outfile = Path(__file__).resolve().parent / "kfold_data.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for individual dataset successfully saved to {outfile}")



def load():
    ROOT = Path(__file__).resolve().parent
    DATA_FILE = ROOT/"NumberRecognitionBigger.mat"
    fid = open(DATA_FILE,'rb')
    readed_data=loadmat(fid)
    ys = np.array(readed_data['y']).transpose()
    input_train_X = np.reshape(np.array(readed_data["X"]),[-1,np.array(readed_data["X"]).shape[-1]]).transpose()
    data = np.column_stack((ys, input_train_X))
    frame = pd.DataFrame(data)
    frame.sort_values(0, axis = 0, ascending = True, inplace = True, na_position ='last') 
    frame = np.array(frame[frame[0].isin([8,9])])
    data = frame[:,1:785] 
    lables = frame[:,0]
    return data, lables
    # loading the data
    # reshaping the array x and joining it with the labled array, then sorting the array.
    # picking 8 and 9 and getting the data ready



def train(input_data,input_label,kernel_type_lin,kernel_type_rbf):
    all_error = []
    errork1=[]
    errork5=[]
    errork10=[]
    error_svm_linear=[]
    error_svm_rbf=[]
    error_random_forest=[]
    StratifiedK = StratifiedShuffleSplit(n_splits=5,train_size=0.8,test_size=0.2, random_state=0)    
    StratifiedK.get_n_splits(input_data, input_label)

    for data_train, data_test in StratifiedK.split(input_data, input_label):
         
        training_set, testing_set = input_data[data_train], input_data[data_test]
        train_label, test_label = input_label[data_train],input_label[data_test]

        knn1 = KNeighborsClassifier(n_neighbors=1)
        knn5 = KNeighborsClassifier(n_neighbors=5)
        knn10 = KNeighborsClassifier(n_neighbors=10)
        ran_clf = RandomForestClassifier(n_estimators=100)
        svm_lin_clf = svm.SVC(kernel=kernel_type_lin, gamma='scale')
        svm_rbf_clf = svm.SVC(kernel=kernel_type_rbf, gamma='scale')


        svm_lin_clf.fit(training_set, train_label)
        output_svm_lin = svm_lin_clf.predict(testing_set)

        svm_rbf_clf.fit(training_set, train_label)
        output_svm_rbf = svm_rbf_clf.predict(testing_set)

        ran_clf.fit(training_set, train_label)
        ran_output = ran_clf.predict(testing_set)

        knn1.fit(training_set, train_label)
        outputk1 = knn1.predict(testing_set)

        knn5.fit(training_set, train_label)
        outputk5 = knn5.predict(testing_set)

        knn10.fit(training_set, train_label)
        outputk10 = knn10.predict(testing_set)

        error_svm_linear.append(evaluate_score(output_svm_lin,test_label))
        error_svm_rbf.append(evaluate_score(output_svm_rbf,test_label))
        error_random_forest.append(evaluate_score(ran_output,test_label))
        errork1.append(evaluate_score(outputk1,test_label))
        errork5.append (evaluate_score(outputk5,test_label))
        errork10.append(evaluate_score(outputk10,test_label))
        # create a loop and feeding all the classifiers for training and testing

    
    all_error.append(statistics.mean(error_svm_linear))
    all_error.append(statistics.mean(error_svm_rbf))
    all_error.append(statistics.mean(error_random_forest))
    all_error.append(statistics.mean(errork1))
    all_error.append(statistics.mean(errork5))
    all_error.append(statistics.mean(errork10))
    # creating the array with the error of each classifier.
    #https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
    #https://www.geeksforgeeks.org/python-statistics-mean-function/
    #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    #https://scikit-learn.org/stable/modules/svm.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    return all_error
    
def evaluate_score(output,test_label):
    accuracy=metrics.accuracy_score(test_label, output)
    error = 1 - accuracy
    return error
    #error is equal to accuracy - 1 which is also same as 1-the fraction of misclassified cases
    #creating the mean of each error value and making an array
    #https://stats.stackexchange.com/questions/133458/is-accuracy-1-test-error-rate
def intersetand_notinterest(inputs):  
    group_of_Interest = []
    group_of_NonInterest = []
    for counter,rows in inputs.iterrows():
        if rows['ca_cervix'] == 1:
            group_of_Interest.append(rows)
        else :
            group_of_NonInterest.append(rows)
    return group_of_Interest,group_of_NonInterest
    #making a array for giving me the inputs for group of intrest and group not of intrest for the training data

def question1():
    input_data,input_label = load()
    all_error = train(input_data,input_label,'linear','rbf')
    column_names = ['svm_linear', 'svm_rbf', "rf", 'knn1', 'knn5', 'knn10']
    kfold_scores = pd.DataFrame([all_error],index=['err'],columns = column_names)
    save_mnist_kfold(pd.DataFrame(kfold_scores))



def question2():
      ROOT = Path(__file__).resolve().parent
      DATA_FILE = ROOT / "mydata.csv"
      data = pd.read_csv(DATA_FILE)
      group_intersst, g_not_interest= intersetand_notinterest(data)

      FEAT_NAMES = [] 
      for col in data.columns:
          if(col !='ca_cervix'):
             FEAT_NAMES.append(col) 
      # taking all the rows in my data set       
     
      COLS = ["Feature", "AUC","sorted"] 
      aucs = pd.DataFrame(columns=COLS,data=np.zeros([len(FEAT_NAMES), len(COLS)]),)
      #creating an data frame with Feature, AUC and sorted.
      #sorted collum for sorting the data
      
      for i, feat_name in enumerate(FEAT_NAMES):
        auc = roc_auc_score(y_true=data['ca_cervix'], y_score=data[feat_name])
        auc_s = auc -.5
        aucs.iloc[i] = (feat_name, auc,np.abs(auc_s))

      aucs_sorted =aucs.sort_values('sorted', ascending=False)
      aucs_sorted = aucs_sorted.nlargest(10,['sorted'])
      aucs_sorted = aucs_sorted.drop(['sorted'], axis=1)
      aucs_sorted.to_json(Path(__file__).resolve().parent / "aucs.json")
      #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
      #https://cmdlinetips.com/2019/03/how-to-select-top-n-rows-with-the-largest-values-in-a-columns-in-pandas/
      #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
      #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
      #running code for roc auc for each row in my data set and putting the value of the values in the column named auc
      #auc value of farther from 0.5 is the least interesting 
      #sorting the auc values of the bases of sorted column 
      #taking top 10 values farthest from 0.5
      #and then dropping the sorting array wich containg the auc-0.5 values
      print(aucs_sorted.round(3).to_markdown())


def question3():
      ROOT = Path(__file__).resolve().parent
      DATA_FILE = ROOT / "mydata.csv"
      data = pd.read_csv(DATA_FILE)
      inputs = np.array(data.iloc[:,0:19])
      #creating the array for all the values from column 1 to 19
      labels= np.array(data.iloc[:,-1])
     
      sc = StandardScaler()
      all_error = train(inputs,labels,'linear','rbf')
      column_names = ['svm_linear', 'svm_rbf', "rf", 'knn1', 'knn5', 'knn10']
      kfold_scores = pd.DataFrame([all_error],index=['err'],columns = column_names)
      save_data_kfold(kfold_scores)



if __name__ == "__main__":
    question1()
    question2()
    question3()




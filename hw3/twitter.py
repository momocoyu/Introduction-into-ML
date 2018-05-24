
from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_listz
        for line in fid:
            t_line = extract_words(line)
            for j in t_line:
                if not (j in word_list):
                    word_list[j] = len(word_list)
        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        row_count = 0
        for line in fid:
            t_line = extract_words(line)
            for t in t_line:
                feature_matrix[row_count, word_list[t]] = 1
            row_count += 1
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    pm = {}
    pm["accuracy"] = metrics.accuracy_score(y_true, y_label)
    pm["f1-score"] = metrics.f1_score(y_true, y_label)
    pm["auroc"] = metrics.roc_auc_score(y_true, y_label)
    pm["precision"] = metrics.precision_score(y_true, y_label)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_label).ravel()
    pm["sensitivity"] = tp/float(tp + fn)
    pm["specificity"] = tn/float(tn+fp)
    return pm
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance    
    skf = kf(y, n_folds=5)
    pm = {'accuracy': 0, 'f1-score': 0, 'auroc': 0, 'precision': 0, 'sensitivity': 0, 'specificity': 0}
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.decision_function(X_test)
        pm_t = performance(y_test, y_pred, metric="accuracy")
        pm["accuracy"] = pm["accuracy"] + pm_t.get("accuracy")
        pm["f1-score"] = pm["f1-score"] + pm_t.get("f1-score")
        pm["auroc"] = pm["auroc"] + pm_t.get("auroc")
        pm["precision"] = pm["precision"] + pm_t.get("precision")
        pm["sensitivity"] = pm["sensitivity"] + pm_t.get("sensitivity")
        pm["specificity"] = pm["specificity"] + pm_t.get("specificity")

    pm["accuracy"] = pm["accuracy"] / 5.0
    pm["f1-score"] = pm["f1-score"] / 5.0
    pm["auroc"] = pm["auroc"] / 5.0
    pm["precision"] = pm["precision"] / 5.0
    pm["sensitivity"] = pm["sensitivity"] / 5.0
    pm["specificity"] = pm["specificity"] / 5.0

    #for t in pm:
      #  print(pm[t])
    return pm[metric]
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print ('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation


    record=0
    c_record=0
    for c in C_range:
        print(c)
        clf_linear = SVC(kernel='linear',C=c)
        temp=cv_performance(clf_linear, X, y, kf, metric="accuracy")
        if temp > record:
            c_record=c
        print('-------')
    return c_record
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print ('RBF SVM Hyperparameter Selection based on ' + str(metric) + ':')
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    metric_list = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    for metric in metric_list:
         print("# Tuning hyper-parameters for %s" % metric)
         gamma=[10000,1000,100,10,1,1e-1,1e-2,1e-3,1e-4]
         C=[10,1,1e-1,1e-2,1e-3,1e-4,100,1000,10000]
         temp=0
         c_record=0
         g_record=0
         best=0
         for c in C:
             for g in gamma:
                 clf=SVC(kernel='rbf',C=c,gamma=g)
                 temp=cv_performance(clf,X,y,kf,metric=metric)
                 if temp>best:
                     best=temp
                     c_record=c
                     g_record=g
         print(best)
         print(c_record)
         print(g_record)
         print('-------')
    return ''
    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance

    y_pred = clf.decision_function(X)
    pm = performance(y, y_pred, metric="accuracy")
    return pm
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    y_train = y[:560]
    y_test = y[560:]
    X_train = X[:560, :]
    X_test = X[560:, :]

    # part 2b: create stratified folds (5-fold CV)
    #clf_linear = SVC(kernel='linear')
    #print(cv_performance(clf_linear, X_train, y_train, StratifiedKFold, metric="accuracy"))
    
    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    #print(select_param_linear(X_train,y_train,StratifiedKFold,metric="accuracy"))
    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV
    select_param_rbf(X_train,y_train,StratifiedKFold)
    # # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    # print("MMMAAAAAAAAAAXXXXXXX")
    # for tt in pm_max:
    #     print(tt, pm_max[tt])

    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    
    # part 4c: report performance on test data
'''
    clf_linear = SVC(kernel='linear', C=100)
    clf_linear.fit(X_train, y_train)
    pm_linear = performance_test(clf_linear, X_test, y_test, metric="accuracy")
    # rbf kernel
    clf_rbf = SVC(kernel='rbf', C=100, gamma=0.01)
    clf_rbf.fit(X_train, y_train)
    pm_rbf = performance_test(clf_rbf, X_test, y_test, metric="accuracy")

    print("Linear result:")
    for tt in pm_linear:
        print(tt, pm_linear[tt])

    print("RBF result:")
    for tt in pm_rbf:
        print(tt, pm_rbf[tt])
'''
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()

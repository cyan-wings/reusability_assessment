#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier

#scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import normalize

#tools
from sklearn.feature_selection import VarianceThreshold
#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

#dimensional reduction & feature selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

#cross_val
from sklearn.model_selection import RepeatedStratifiedKFold

class CLF():
    
    def __init__(self, X, y, classifier, preprocess_method = None, d_reduction = 0,
                 f_selection = None, seed = 27033074, vt = False, resampling = False, 
                 cv = 4, cv_r = 10, permute = False, top = 10, **params):
        
        self.X = X
        self.y = y
        self.classifier = classifier
        self.pm = preprocess_method
        self.dr = d_reduction
        self.fs = f_selection
        self.vt = vt
        self.resampling = resampling
        self.p = params
        self.results_array = [[], [], [], []]  #f1, pre, rec, acc
        
        if permute == False:
            self.skf = RepeatedStratifiedKFold(n_splits = cv, n_repeats = cv_r, random_state=seed)
            for train_index, test_index in self.skf.split(X, y):
                self.X_train, self.X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                self.y_train, self.y_test = self.y[train_index], self.y[test_index]
                self.clf = make_pipeline(self.sampling(), self.variance_t(), self.preprocess(),
                                         self.dim_reduction(), self.fea_selection(), self.model())
                self.clf.fit(self.X_train, self.y_train)
                self.y_pred = self.clf.predict(self.X_test)
                self.results_array[0].append(self.get_f1())
                self.results_array[1].append(self.get_pre())
                self.results_array[2].append(self.get_rec())
                self.results_array[3].append(self.get_acc())
                #print(classification_report(self.y_test, self.y_pred))

            #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=seed)
            #self.clf = Pipeline([('preprocessor', self.preprocess()), ('model', self.model())])
        else:
            #take top 10 important features (default)
            self.skf = RepeatedStratifiedKFold(n_splits = cv, n_repeats = cv_r, random_state=seed)
            for train_index, test_index in self.skf.split(X, y):
                self.X_train, self.X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                self.y_train, self.y_test = self.y[train_index], self.y[test_index]
                self.clf = make_pipeline(self.sampling(), self.variance_t(), self.preprocess(),
                                         self.dim_reduction(), self.fea_selection(), self.model())
                self.clf.fit(self.X_train, self.y_train)
                self.y_pred = self.clf.predict(self.X_test)
                print('Computing permutation importance...')
                start = time.time()
                result = permutation_importance(self.clf, self.X_test, self.y_test, random_state=seed, n_jobs=-1)
                print('Process completed at ' + str(round((time.time() - start)/60, 3)) + ' min')
                with open("PI-" + str(self.get_model())[:10] + "_" + str(self.get_preprocess()) + ".csv", 'a') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    w = 1
                    for i in result.importances_mean.argsort()[::-1]:
                        #if round(result.importances_mean[i] - 2 * result.importances_std[i], 4) > 0:
                        writer.writerow([self.X.columns[i], 
                                         round(result.importances_mean[i], 3), 
                                         round(result.importances_std[i], 3), self.get_f1()])
                        w += 1
                        if w > 10:
                            break
        
    def variance_t(self, t = 0.8 * (1 - 0.8)):
        
        if self.vt == True:
            return VarianceThreshold(threshold = t)
        
        return None
    
    def sampling(self):
        if self.resampling == True:
            return RandomOverSampler(sampling_strategy='minority', random_state=27033074)
        
        return None

    def preprocess(self, seed = 27033074):
        if self.pm == 'ss':
            return StandardScaler()
        elif self.pm == 'mms':
            return MinMaxScaler()
        elif self.pm == 'mas':
            return MaxAbsScaler()
        elif self.pm == 'rs':
            return RobustScaler()
        elif self.pm == 'pty':
            return PowerTransformer()
        elif self.pm == 'ptb':
            return PowerTransformer(method = 'box-cox')
        elif self.pm == 'qtu':
            return QuantileTransformer(random_state = seed)
        elif self.pm == 'qtn':
            return QuantileTransformer(output_distribution = 'normal', random_state = seed)
        elif self.pm == 'n':
            self.X_train = normalize(self.X_train, axis = 0)
            self.X_test = normalize(self.X_test, axis = 0)
            return None
        else:
            return None
        
    def dim_reduction(self, seed = 27033074):
        
        if self.dr == 0:
            return None
        elif self.dr == 1:
            return PCA(n_components=self.p['f'], random_state=seed)
        else:
            return
        
    def fea_selection(self, seed = 27033074):
        
        if self.fs == None:
            return None
        elif self.fs == 'kb':
            return SelectKBest(self.p['kbest_f'], k=self.p['f'])
        elif self.fs == 'rf':
            return SelectFromModel(RandomForestClassifier(n_estimators = self.p['fs_n'], n_jobs = -1, random_state=seed), 
                                   threshold=-np.inf, max_features=self.p['f'])
        else:
            return
    
    def model(self, seed = 27033074):
        if self.classifier == 'rf':
            return RandomForestClassifier(n_estimators = self.p['n'], n_jobs = -1, random_state=seed)
        elif self.classifier == 'knn':
            return KNeighborsClassifier(n_neighbors = self.p['n'], n_jobs = -1)
        elif self.classifier == 'mlp':
            return MLPClassifier(random_state = seed, max_iter=10000)
        elif self.classifier == 'dt':
            return DecisionTreeClassifier(random_state = seed)
        elif self.classifier == 'sgd':
            return SGDClassifier(loss = self.p['l'], n_jobs = -1, random_state = seed)
        elif self.classifier == 'r':
            return RidgeClassifierCV(cv = self.p['n'])
        elif self.classifier == 'svm':
            return SVC(gamma = self.p['gamma'], random_state = seed)
        elif self.classifier == 'xg':
            return xgb.XGBClassifier(random_state = seed)
        elif self.classifier == 'gpc':
            return GaussianProcessClassifier(random_state = seed)
        elif self.classifier == 'qda':
            return QuadraticDiscriminantAnalysis()
        elif self.classifier == 'ada':
            return AdaBoostClassifier(random_state = seed)
        elif self.classifier == 'gb':
            return GradientBoostingClassifier(n_estimators=self.p['n'], random_state=seed)
        else:
            print('No input model!')
    
    def mean_results_array(self):
        return [np.mean(self.results_array[0]), np.mean(self.results_array[1]), 
                np.mean(self.results_array[2]), np.mean(self.results_array[3]), 
                self.get_model(), self.num_of_features(), self.get_preprocess(), self.fs, self.dr, self.vt, 
                classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)['1']['support'],
                self.resampling]
    
    def write_result(self, filename):
        
        isNone = not os.path.isfile(filename + '.csv')
        
        with open(filename +  ".csv", 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if isNone:
                writer.writerow(['F1-Score', 'Precision', 'Recall', 'Accuracy', 'Model', 
                                 'Num_Features', 'Preprocessing', 'Fea_Selection', 
                                 'Dim_Reduction', 'Var_Threshold', 'support', 'resampling'])
            writer.writerow(self.mean_results_array())
        
        return
    
    def num_of_features(self):
        try:
            return self.p['f']
        except:
            return len(self.X.columns)
        
    def get_preprocess(self):
        
        if self.pm == 'n':
            return 'normalize'
        else:
            return self.clf[2]
    
    def get_model(self):
        return self.clf[-1]
    
    def get_model_params(self):
        return self.clf[-1].get_params()
    
    def __str__(self):
        return str(self.clf.steps)
    
    def get_acc(self):
        return round(self.clf.score(self.X_test, self.y_test), 3)
    
    def get_f1(self):
        #print(classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)['1']['support'])
        return round(classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)['1']['f1-score'],3)
    
    def get_pre(self):
        return round(classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)['1']['precision'],3)
    
    def get_rec(self):
        return round(classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)['1']['recall'], 3)
    
    def get_cmatrix(self):
        return confusion_matrix(self.y_test, self.y_pred)
    
    def p_importance(self, train_or_test, seed = 27033074):
        
        print('Computing permutation importance...')
        start = time.time()
        if train_or_test == 'train':
            result = permutation_importance(self.clf, self.X_train, self.y_train, random_state=seed, n_jobs=-1)
        elif train_or_test == 'test':
            result = permutation_importance(self.clf, self.X_test, self.y_test, random_state=seed, n_jobs=-1)
        else:
            print('Input train or test!')
            return
        
        print('Process completed at ' + str(round((time.time() - start)/60, 3)) + ' min')
        
        with open("PI-" + str(self.get_model()) + "_" + str(self.get_preprocess()) + ".csv", 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in result.importances_mean.argsort()[::-1]:
                #if round(result.importances_mean[i] - 2 * result.importances_std[i], 4) > 0:
                writer.writerow([self.X.columns[i], 
                                 round(result.importances_mean[i], 3), 
                                 round(result.importances_std[i], 3)])
        
        return
    
    def f_importance(self):
        #problem with the feature removals
        x = list(zip(self.clf[-1].feature_importances_, self.X.columns.values))
        x = pd.DataFrame(x, columns=["Importance","Feature_Name"])
        x = x.sort_values(by=['Importance'], ascending=False)
        x.to_csv('FI-' + str(self.get_model()) + "_" + str(self.get_preprocess()) +  ".csv", index=False)
        return
    
    def coefs(self):
        
        x = list(zip(self.clf[-1].feature_importances_, self.X.columns.values))
        x = pd.DataFrame(x, columns=["Importance","Feature_Name"])
        x = x.sort_values(by=['Importance'], ascending=False)
        x.to_csv('FI-' + str(self.get_model()) + "_" + str(self.get_preprocess()) +  ".csv", index=False)
        return

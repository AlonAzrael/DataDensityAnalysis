
import pandas as pd

import numpy as np
from numpy import random

from numpy import histogram, histogramdd

from statsmodels.nonparametric.kernel_density import KDEMultivariate, EstimatorSettings

from sklearn.cluster import DBSCAN as DBSCAN_, MeanShift as MeanShift_
from sklearn.mixture import GMM as GMM_
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import minmax_scale
from sklearn.grid_search import GridSearchCV

class NDHist():

    """
    only suitable for low dimensional data, and will be replaced by histogram-cluster someday
    """

    def __init__(self, df):
        self.df = df

    def hist(self, values, bins=3):

        H, edges = histogramdd(values, bins=bins, normed=False)
        
        # print H
        # print edges

        # try:
        #     x = len(bins)
        #     mbins = bins
        # except:
        #     mbins = np.asarray([bins]*values.shape[1], dtype=np.int)

        n_samples = len(values)
        ndhist = []
        for index, x in np.ndenumerate(H):
            percent_samples = np.float16(1.0*x/n_samples)
            
            edge_features = []
            for edge_row,edge_column in enumerate(index):
                edge_feature = [edges[edge_row][edge_column], edges[edge_row][edge_column+1]]
                edge_features.append(np.asarray(edge_feature,dtype=np.float32))

            new_hist = [percent_samples, edge_features]
            ndhist.append(new_hist)

        # print ndhist

        return ndhist

    def report_hist(self, ndhist, column_names):
        ndhist_v = filter(lambda x:x[0]>1e-3, ndhist)
        ndhist_v = sorted(ndhist_v, key=lambda x:x[0], reverse=True)

        header = [str(name) for name in column_names] + ["percent"]
        report = [header]
        for x in ndhist_v:
            percent_samples, edge_features = x
            line = [" - ".join([str(f) for f in ef]) for ef in edge_features]
            line.append(str(percent_samples))
        
            report.append(line)

        # report_lines = "\n".join(["\t".join(line) for line in report])
        report_lines = tabulate(report)

        return report_lines

    def export_hist(self, columns=None, bins=3):
        df = self.df

        if columns is None:
            values = df.values
            columns = df.columns.values
        else:
            values = df[columns].values

        ndhist = self.hist(values, bins)
        print self.report_hist(ndhist, columns)


"""
KDE
======================================================
"""

class KDE_sm():

    def __init__(self, params=None):
        self.params = params

    def fit(self, values, var_type=None):
        values = np.asarray(values)
        if var_type is None:
            var_type = "".join(["c" for _ in xrange(values.shape[1])])

        settings = EstimatorSettings(efficient=False, randomize=False, n_res=25, n_sub=50, return_median=False, return_only_bw=False, n_jobs=-1)
        model = KDEMultivariate(data=values, var_type=var_type, bw="normal_reference", defaults=settings)
        self.model = model
        return self

    def predict(self, values):
        values = np.asarray(values)
        return self.model.pdf(values)

class KDE_sklearn():

    def __init__(self, params=None):
        self.params = params

    def fit(self, values):
        values = np.asarray(values)
        cv_mode = True

        # just open cv
        if cv_mode:
            grid = GridSearchCV(KernelDensity(), param_grid=dict(bandwidth=np.linspace(0.05, 2.0, 10), algorithm=['auto'], kernel=['gaussian'], metric=['euclidean'], atol=[0], rtol=[0], breadth_first=[True], leaf_size=[40], metric_params=[None]), cv=6, n_jobs=-1) # 20-fold cross-validation
            grid.fit(values)
            model = grid.best_estimator_
        else:
            model = KernelDensity(bandwidth=1.0, algorithm='auto', kernel='gaussian', metric='euclidean', atol=0, rtol=0, breadth_first=True, leaf_size=40, metric_params=None)
            model.fit(values)

        self.model = model
        return self

    def predict(self, values):
        values = np.asarray(values)
        logproba = self.model.score_samples(values)
        return np.exp(logproba)

class KDE():

    """
    KDE is short for kernel density estimation, which is suitable for high dimensional data histogram
    """

    def __init__(self, df, model_type="sm", type_settings=None):
        self.df = df
        values = df.values
        self.values = values
        columns = df.columns

        var_type = [0]*len(columns)
        var_type_dict = {col:i for i,col in enumerate(columns)}
        for i,dtype in enumerate(df.dtypes):
            if "int" in dtype.name:
                var_type[i] = "u"
            else:
                var_type[i] = "c"

        if type_settings is not None:
            for key,s in type_settings.items():
                var_type[var_type_dict[key]] = s

        var_type = "".join(var_type)

        if model_type == "sm":
            model = KDE_sm()
            model.fit(values, var_type)
        elif model_type == "sklearn":
            model = KDE_sklearn()
            model.fit(values)
        
        self.model = model

    def sampling_pdf(self, n_sample=20):
        model = self.model
        values = self.values

        vd_samples = values[random.choice(len(values), n_sample)]
        pvalues = model.predict(vd_samples)
        pvalues = minmax_scale(pvalues, feature_range=(0, 1), axis=0, copy=False)

        header = list(self.df.columns) + ["proba"]
        body = np.concatenate([vd_samples, np.asarray([pvalues]).T], axis=1)
        sorted_body = body[body[:,-1].argsort()[::-1]]
        
        report_df = pd.DataFrame(data=sorted_body, columns=header)
        return report_df

def report_kde(df, model_type="sm", n_sample=20, type_settings=None):
    model = KDE(df, model_type=model_type, type_settings=type_settings)
    return model.sampling_pdf(n_sample)


"""
MeanShift
======================================================
"""
class MeanShift():

    def __init__(self, df, params=None):
        self.df = df
        values = df.values
        self.values = values

        default_params = dict(bandwidth=None, seeds=None, bin_seeding=True, min_bin_freq=1, cluster_all=False, n_jobs=1)
        if params is None:
            params = default_params
        else:
            default_params.update(params)
            params = default_params

        ms = MeanShift_(**params)
        ms.fit(values)
        self.model = ms

        self.centers = self.model.cluster_centers_

        labels = self.model.labels_
        self.outliers = values[labels == -1]

    def get_centers(self):
        return pd.DataFrame(self.centers, columns=self.df.columns)

    def get_outliers(self):
        return pd.DataFrame(self.outliers, columns=self.df.columns)

def report_meanshift(df):
    model = MeanShift(df, params=None)
    return model.get_centers(), model.get_outliers()



"""
DBSCAN
======================================================
"""
class DBSCAN():

    def __init__(self, df, params=None):
        self.df = df
        values = df.values
        self.values = values

        default_params = dict(eps=0.5, min_samples=10, metric="euclidean")
        if params is None:
            params = default_params
        else:
            default_params.update(params)
            params = default_params

        dbscan = DBSCAN_(**params)
        dbscan.fit(values)
        self.model = dbscan

        self.centers = dbscan.components_

        labels = dbscan.labels_
        self.outliers = values[labels == -1]

    def get_centers(self):
        return pd.DataFrame(self.centers, columns=self.df.columns)

    def get_outliers(self):
        return pd.DataFrame(self.outliers, columns=self.df.columns)


# USELESS
def report_dbscan(df, eps=1):
    model = DBSCAN(df, params=dict(eps=eps))
    return model.get_centers(), model.get_outliers()




"""
GMM
======================================================
"""
class GMM():

    def __init__(self, df, params=None):
        self.df = df
        values = df.values
        self.values = values

        default_params = dict(n_components=3, covariance_type='diag', random_state=None, tol=0.001, min_covar=0.001, n_iter=100, n_init=3, params='wmc', init_params='wmc', verbose=0)
        if params is None:
            params = default_params
        else:
            default_params.update(params)
            params = default_params

        gmm = GMM_(**params)
        gmm.fit(values)
        self.model = gmm

        self.centers = gmm.means_
        self.center_weights = gmm.weights_

    def is_converged(self):
        return self.model.converged_

    def get_centers(self):
        centers = np.concatenate([self.centers, np.asarray([self.center_weights]).T], axis=1)
        header = list(self.df.columns) + ["weights_"]
        return pd.DataFrame(centers, columns=header)


def report_gmm(df, n_components=5):
    model = GMM(df, params=dict(n_components=n_components))
    return model.get_centers(), model.is_converged()







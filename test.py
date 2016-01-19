

import pandas as pd
from sklearn.preprocessing import LabelEncoder, Imputer

from data_generator import gen_test_df
from tabulate import tabulate

from density_analysis import KDE, MeanShift, DBSCAN
import density_analysis as sd

def csv_loader():
    df = pd.read_csv("./datasets/abalone.data.csv", delimiter=",", header=None)

    return df

def load_abalone():
    df = csv_loader()
    df_m = DataFrameManager(df)

    # transform names
    names = "sex,length,diameter,height,whole_weight,shucked_weight,viscera_weight,shell_weight,rings"
    df_m.set_column_names(names.split(","))

    # transform column as label
    df_m.encode_column_as_label("sex")

    return df

class DataFrameManager():

    def __init__(self, df):
        self.df = df

    def set_column_names(self, names):
        df = self.df

        if isinstance(names, type({})):
            df.rename(columns = names, inplace = True)
        else: # expected array-like
            df.columns = names

    def encode_column_as_label(self, column_name):
        df = self.df

        column = df[column_name]
        le = LabelEncoder()
        le.fit(column)
        new_column = le.transform(column)
        df[column_name] = new_column

    def count_column_as_label(self, column_name):
        df = self.df
        
        label_count = df[column_name].value_counts()
        print label_count

    def impute_column_missing_value(self):
        Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=False)

def test_KDE():
    df = gen_test_df(n_features=3)
    
    model = KDE(df)
    report_df = model.sampling_pdf(n_sample=20)
    print report_df

    model = KDE(df, model_type="sklearn")
    report_df = model.sampling_pdf(n_sample=20)
    print report_df


def test_ClusterAnalysis():
    df = gen_test_df(n_features=3)

    model = MeanShift(df)

    centers = model.get_centers()
    outliers = model.get_outliers()

    print centers
    print outliers

def test_GMM():
    df = gen_test_df(n_features=3)
    centers, is_converged = sd.report_gmm(df, n_components=5)

    print centers
    print is_converged

def test_report():
    df = load_abalone()
    
    # report_df = sd.report_kde(df, model_type="sklearn", n_sample=50, type_settings={"rings":"o"})
    # print report_df

    centers, is_converged = sd.report_gmm(df, n_components=5)
    print centers
    print is_converged

    centers, outliers = sd.report_meanshift(df)
    print centers
    print outliers

# df = load_abalone()
# df = gen_test_df(n_features=3)


def main():
    
    pass
    # count
    # df_m.count_column_as_label("sex")

    # print df

if __name__ == '__main__':
    # test_KDE()
    # test_ClusterAnalysis()
    # test_GMM()
    test_report()



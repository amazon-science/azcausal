from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DIDRegressor

if __name__ == '__main__':

    # initialized the DiD using regression
    estimator = DIDRegressor()

    # directly providing a data frame
    df = CaliforniaProp99().load(rename=True)
    print(df.columns)

    result = estimator.fit(df)
    print(result.summary(title="CaliforniaProp99 (from DataFrame)"))

    # providing a panel
    df = CaliforniaProp99().panel()
    result = estimator.fit(df)
    print(result.summary(title="CaliforniaProp99 (from Panel)"))





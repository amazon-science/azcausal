from azcausal.data import Billboard
from azcausal.estimators.panel.did import DIDRegressor

df = Billboard().load()

result = DIDRegressor().fit(df)

print(result.summary())

# def test_billboard():
# 
#
#     df = Billboard().load()
#
#     result = DIDRegressor().fit(df)
#
#     print(result.summary())
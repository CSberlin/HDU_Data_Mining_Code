from data_load import data_load
from apyori import apriori

data = data_load()

result = list(apriori(transactions=data,min_support=3,min_confidence = 0.7))
print(result)
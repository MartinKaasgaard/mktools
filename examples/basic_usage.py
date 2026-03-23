import pandas as pd
from mktools.kstat import KStatProfiler
from mktools.kio import DataImporter

# Data import
# df = DataImporter().load("data/sample.csv")

df = pd.DataFrame({"a": [1, 2, None], "b": [" yes ", "no", None]})
profiler = KStatProfiler(df, depth="deep")
reports = profiler.profile()
print(reports["missingness"])

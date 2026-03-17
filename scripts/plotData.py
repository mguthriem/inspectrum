from inspectrum.loaders import load_mantid_csv
from inspectrum.plotting import inspect_peaks
s = load_mantid_csv('tests/test_data/SNAP059056_all_test-0.csv')[0]
pt = inspect_peaks(s)
import matplotlib.pyplot as plt; plt.show()
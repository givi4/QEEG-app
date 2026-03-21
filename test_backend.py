import mne
import matplotlib.pyplot as plt
print('after mne:', plt.get_backend())
import numpy as np
print('after numpy:', plt.get_backend())
from edf_loader import load_edf
print('after edf_loader:', plt.get_backend())
from preprocessor import preprocess
print('after preprocessor:', plt.get_backend())

# Test if a window actually opens
print('attempting to open a window...')
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title('Backend test — close this window to continue')
print('backend at show time:', plt.get_backend())
plt.show(block=True)
print('window closed successfully')
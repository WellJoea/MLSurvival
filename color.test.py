import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
elif platform.system()=='Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import fastcluster
import os
diy_color = ['#00B281', '#DA3B95', '#00C5CD', '#E7A72D', '#8B00CC', '#EE7AE9',
            '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF',
            '#FF2121', '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C',
            '#FB9A99', '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2',
            '#95A5A6', '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC',
            '#A473AE', '#FF0000', '#EE7777', '#009E73', '#ED5401', '#CC0073',]
testd = pd.DataFrame([1]* len(diy_color), index=diy_color).T
print(testd)
testd.plot.bar(color=diy_color)
plt.xticks(rotation='270')
plt.show()
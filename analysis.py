from deepforest import CascadeForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

tree_number = 100
layer = 0
with np.load(f"D:\TL\deepforest\layer_feature\\tree_{tree_number}\\feature_{layer}.npz") as F:
    feature = F['feature'][0:39*7*256]
feature = feature.reshape(-1, 7, 256)
feature_num, segment_num, channel_num = feature.shape
feature_channel = np.mean(feature, axis=1)
feature_name = ['mlf', 'wa', 'vare', 'ssi', 'myop', 'mmav2', 'mmav', 'ld', 'dasdv', ' aac', 'rms', 'wl', 'zc', 'ssc',
                'mav', 'iemg', 'ae', 'var', 'sd', 'cov', 'kurt', 'skew', 'iqr', 'mad', 'damv', 'tm', 'vo', 'dvarv',
                'ldamv', 'ldasdv', 'card', 'lcov', 'ltkeo', 'msr', 'ass', 'asm', 'fzc', 'ewl', 'emav']
just_feature=np.mean(np.mean(feature,axis=1),axis=1)
order=just_feature.argsort()[::-1]
order_name=[]
for n in range(feature_num):
    order_name.append(feature_name[order[n]])
a=np.zeros((16,16))
for k in range(5):
    a=a+feature_channel[order[k]].reshape(16, 16)


for i in range(feature_num):
    plt.figure()
    feature_select=feature_channel[i].reshape(16, 16)
    feature_select_interpolation=cv2.resize(feature_select, (64, 64), interpolation=cv2.INTER_CUBIC)
    ax=sns.heatmap(feature_select_interpolation)
    ax.set_title(f"{feature_name[i]}")
plt.show()

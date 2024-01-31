# %%
import os
import numpy as np
import pandas as pd
import yaml
import pyarrow.feather as feather
from scipy.interpolate import Rbf # 外挿・内挿のための関数．

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize # Normalizeをimport
from matplotlib.cm import ScalarMappable
import plotly.express as px
import plotly.graph_objects as go
import plotly
from IPython.display import display, HTML

plotly.offline.init_notebook_mode()
display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
))

from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.text import Annotation
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.family'] = 'IPAexGothic'
import canopy_photo_greenhouse as CP

pd.set_option('display.max_columns', None)

def cal_diurnal(rpath_params, rdir_results):
    '''
    設定ファイルおよびfeatherのデータから、積算値などを計算する
    '''
    ###################################################
    # パラメータの読み込み
    with open(rpath_params, "r") as file:
        test = yaml.safe_load(file)
    for key, val in test.items():
        exec(key + "=val")    
    ###################################################
    # 計算結果の読み込み
    rfile_results_list = os.listdir(rdir_results)
    rfile_results_list = sorted(rfile_results_list)

    ###################################################
    # 出力先
    wdir = os.path.join(rdir_results, "figs")
    if not os.path.exists(wdir):
        os.makedirs(wdir)

    ###################################################
    # パラメータリスト
    ###################################################   
    #######################
    # # ハウスに関するパラメータ
    W_ground = (W_row * n_row) + (W_path * (n_row - 1)) + W_margin * 2
    L_ground = L_row + L_margin * 2
    A_ground_in_a = W_ground * L_ground /100

    # #######################
    # # 作物群落に関するパラメータ
    LAI = H_row * W_row * L_row * n_row * LD / (W_ground * L_ground) 


    # 畝の端の座標を計算
    #######################
    # 見た目に関するパラメータ
    radius_sun_orbit = L_row / 2 *1.1 # 太陽軌道の見かけの半径
    list_edge_negative_y, list_edge_positive_y = CP.cal_row_edges(W_row, H_row, L_row, n_row, W_path, azimuth_row)

    I_A_list = []
    for rfile_results in rfile_results_list:
        if rfile_results.split(".")[-1] == "feather":
            rfile_path = os.path.join(rdir_results, rfile_results)
            df_results = feather.read_feather(rfile_path)
            A_per_ground = (df_results["dV"] * LD * df_results["A_per_LA"]).sum() / (A_ground_in_a * 100)
            I_abs_per_ground = (df_results["dV"] * LD * df_results["I_abs_per_LA"]).sum() / (A_ground_in_a * 100)
            I_A_list.append([df_results["Time"][0], df_results["I0_beam_h"][0], df_results["I0_dif_h"][0], I_abs_per_ground, A_per_ground])
            
    # 日変化をまとめる
    df_diurnal = pd.DataFrame(data = I_A_list, columns = ["Time", "I0_beam_h", "I0_dif_h", "I_abs_per_ground", "A_per_ground"])
    df_diurnal["I0_h"] = df_diurnal["I0_beam_h"] + df_diurnal["I0_dif_h"]
    A_per_ground_daily =df_diurnal["A_per_ground"].sum() * 3600 /10**6
    print("日あたりの個体群光合成速度 = {0:3.2f} mol m-2 ground s-1".format(A_per_ground_daily))
    return df_diurnal, A_per_ground_daily, LAI

# %%
rpath_params_90 = "/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/畝90度/parameter_list.yml"
rdir_results_90 = r"/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/畝90度/output/" 
df_diurnal_90, A_per_ground_daily_90, LAI = cal_diurnal(rpath_params_90, rdir_results_90)

rpath_params_0 = "/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/畝0度/parameter_list.yml"
rdir_results_0 = r"/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/畝0度/output/" 
df_diurnal_0, A_per_ground_daily_0, LAI = cal_diurnal(rpath_params_0, rdir_results_0)

# %%
xfmt = mdates.DateFormatter('%H時')#%m/%d 
nrows = 3
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(4, 6))
############
ax1=plt.subplot(3,1,1)
ax1.set_title("群落上部のPPFD",fontsize=15)
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_ylabel('$\mathrm{PPFD}$  \n ($\mathrm{\mu mol \; m^{-2}_{ground} \; s^{-1}}$)')
plt.plot(df_diurnal_0["Time"], df_diurnal_0["I0_h"], label = "全天", color = "black")        
plt.plot(df_diurnal_0["Time"], df_diurnal_0["I0_beam_h"], label = "直達", color = "magenta")        
plt.plot(df_diurnal_0["Time"], df_diurnal_0["I0_dif_h"], label = "散乱", color = "navy")        
#plt.plot(df_diurnal_90["Time"], df_diurnal_90["I0_h"])        

plt.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0,edgecolor='none',
            frameon=False,labelspacing=0.1,handlelength=.5,handleheight=.1,columnspacing=0,fontsize=12) #,prop={"family":"Times New Roman"}

ax2=plt.subplot(3,1,2)
ax2.set_title("吸収PPFD",fontsize=15)
ax2.xaxis.set_major_formatter(xfmt)
ax2.set_ylabel('$\mathrm{PPFD}$  \n ($\mathrm{\mu mol \; m^{-2}_{ground} \; s^{-1}}$)')
plt.plot(df_diurnal_0["Time"], df_diurnal_0["I_abs_per_ground"], label = "南-北 棟")        
plt.plot(df_diurnal_90["Time"], df_diurnal_90["I_abs_per_ground"], label = "東-西 棟")        

plt.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0,edgecolor='none',
            frameon=False,labelspacing=0.1,handlelength=.5,handleheight=.1,columnspacing=0,fontsize=12) #,prop={"family":"Times New Roman"}

############
ax3=plt.subplot(3,1,3)
ax3.set_title("群落光合成速度",fontsize=15)
ax3.xaxis.set_major_formatter(xfmt)
ax3.set_ylabel('群落光合成速度  \n ($\mathrm{\mu mol \; m^{-2}_{ground} \; s^{-1}}$)')
plt.plot(df_diurnal_0["Time"], df_diurnal_0["A_per_ground"], label = "南-北 棟")        
plt.plot(df_diurnal_90["Time"], df_diurnal_90["A_per_ground"], label = "東-西 棟")        

plt.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0,edgecolor='none',
            frameon=False,labelspacing=0.1,handlelength=.5,handleheight=.1,columnspacing=0,fontsize=12) #,prop={"family":"Times New Roman"}

ax1.grid('on', which='major', axis='x',linestyle="--",alpha=0.6 )
ax2.grid('on', which='major', axis='x',linestyle="--",alpha=0.6 )       
ax3.grid('on', which='major', axis='x',linestyle="--",alpha=0.6 )       

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# %%
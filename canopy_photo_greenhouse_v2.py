#%%
#%matplotlib notebook
# for creating a responsive plot
#%matplotlib widget
'''
v2  --- leaf angle distributionの修正を実施。sphericalではなくてellipsoidalに対応できるようにした。

'''


import os
import sys
import time
import numpy as np
import pandas as pd
#import modin.pandas as pd
import swifter
#swifter.register_modin()
swifter.set_defaults(progress_bar = False, npartitions = 32*20,  allow_dask_on_strings=True) # , force_parallel = True

import yaml
from types import SimpleNamespace # dictだと面倒なので
import pyarrow.feather as feather
from scipy.interpolate import Rbf # 外挿・内挿のための関数．

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib import cm
# import matplotlib.colors as mcolors
# from matplotlib.colors import Normalize # Normalizeをimport
# from matplotlib.cm import ScalarMappable
# import plotly.express as px
# import plotly.graph_objects as go

from scipy.interpolate import RegularGridInterpolator
# from mpl_toolkits.mplot3d.proj3d import proj_transform
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# from matplotlib.text import Annotation
# from matplotlib.patches import FancyArrowPatch

from timeit import default_timer as timer
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.family'] = 'IPAexGothic'

# 自作module
import leaf_photo_Baldocchi as LP
import G_functions as GF


##################################################################
# 点の作成・計算等
##################################################################
def cal_row_edges(W_row, H_row, L_row, n_row, W_path, azimuth_row):
    '''
    うねの端点の座標を返す．畝は，南北方向へ延びる畝をdefaultとしてazimuth_rowだけ，時計回りに回転させている．
    ただし，本プログラムでは，"畝座標"を基準にすることから，見た目や座標はそのまま(x-z平面が畝の正面)．
    原点はほ場の中心．
    
        入力
            W_row  -- 畝幅 (m)
            H_row  -- 畝高 (m)
            L_row  -- 畝長 (m)
            n_row  -- 畝の本数 (m)
            W_path -- 通路幅 (m)
            azimth_row_r -- 畝の向き（°; 北向きから東向きに回転した）
        出力
    '''
    azimuth_row_rad = (azimuth_row / 180 ) * np.pi

    # うねの座標（端から端まで）
    list_edge_negative_y = [] # yが負の側の，畝の端点の座標  
    list_edge_positive_y = [] # yが正の側の，畝の端点の座標
    for i in range(n_row):
        if n_row % 2 == 0:
            x1_row = (-n_row/2 + i) * (W_row + W_path) + (1.0 / 2.0) * W_path
            x2_row = x1_row + W_row
        elif n_row % 2 != 0:
            x1_row = (-(n_row -1) / 2 + i) * (W_row + W_path) - (1.0 / 2.0) * W_row
            x2_row = x1_row + W_row
        y1_row = - L_row / 2
        y2_row = L_row /2

        # 座標の形に直す．
        xyz1 = [x1_row, y1_row, H_row]
        xyz2 = [x2_row, y1_row, H_row]
        xyz3 = [x1_row, y2_row, H_row]
        xyz4 = [x2_row, y2_row, H_row]

        list_edge_negative_y.append(xyz1)
        list_edge_negative_y.append(xyz2)
        list_edge_positive_y.append(xyz3)
        list_edge_positive_y.append(xyz4)

    list_edge_negative_y = np.array(list_edge_negative_y)
    list_edge_positive_y = np.array(list_edge_positive_y)

    # 左手座標系で表示する; y軸の向きをひっくり返す
    # ax.invert_yaxis()

    return list_edge_negative_y, list_edge_positive_y

def create_grid_points_uniform(list_edge_negative_y, list_edge_positive_y, Nx_per_row, Ny_per_row, Nz_per_row, Nx_per_btm, Ny_per_btm, W_margin, L_margin):
    '''
    光合成計算用のgrid pointsを作る．等間隔に畝を分割する．端点は含まない．
    下図では，
    Nx_per_row = 4, 
    Nz_per_row = 3
    -----------------
    | x | x | x | x |
    -----------------
    | x | x | x | x |
    -----------------
    | x | x | x | x |
    -----------------

    入力
        list_edge_negative_y    --- 畝の端の点の二次元array.yが負.[point番号][x,y,z]
        list_edge_positive_y    --- 畝の端の点の二次元array.yが正.[point番号][x,y,z]
        
        Nx_per_row              --- rowの中で，光を計算する点を作る際の，x方向への分割数
        Ny_per_row              --- rowの中で，光を計算する点を作る際の，y方向への分割数
        Nz_per_row              --- rowの中で，光を計算する点を作る際の，z方向への分割数
        
        Nx_per_btm
        Ny_per_btm

        W_margin                --- 両側の畝のさらに外側の通路幅 (m)．反射光計算に使用する．
        L_margin                --- 畝の前後の外側の通路幅 (m)．反射光計算に使用する．
    '''
    
    # 畝内の点の座標
    dy_row = (list_edge_positive_y[0][1] - list_edge_negative_y[0][1]) / Ny_per_row
    y_array = np.linspace(list_edge_negative_y[0][1], list_edge_positive_y[0][1], Ny_per_row, endpoint=False) + dy_row / 2
    dz_row = list_edge_negative_y[0][2] / Nz_per_row
    z_array = np.linspace(0, list_edge_negative_y[0][2], Nz_per_row, endpoint= False) + dz_row / 2

    ix = np.shape(list_edge_negative_y)[0]
    x_array = np.array([])
    dx_row = (list_edge_negative_y[1][0] - list_edge_negative_y[0][0]) / Nx_per_row
    for i in range(0, ix, 2):
        x_array_dummy = np.linspace(list_edge_negative_y[i][0], list_edge_negative_y[i+1][0], Nx_per_row, endpoint= False) + dx_row / 2
        x_array = np.append(x_array, x_array_dummy)
    
    x_row, y_row, z_row = np.meshgrid(x_array, y_array, z_array)

    # 畝内の点の占有体積および専有面積
    dV_row = dx_row * dy_row * dz_row # m3
    dA_row = dx_row * dy_row # m2


    # 通路の底（反射光計算用）の座標．
    # 畝の周りのマージン（L_margin, W_margin）を考慮に入れている． 
    # dy_path = ((list_edge_positive_y[0][1] + L_margin) - (list_edge_negative_y[0][1] - L_margin)) / Ny_per_path_btm
    # y_array_path_btm = np.linspace(list_edge_negative_y[0][1] - L_margin, list_edge_positive_y[0][1] + L_margin, Ny_per_path_btm, endpoint= False) + dy_path / 2
    # z_array_path_btm = np.array([0])

    # x_array_path_btm = np.array([])
    # dx_path = (list_edge_negative_y[2][0] - list_edge_negative_y[1][0]) / Nx_per_path_btm
    # for i in range(1,ix-1, 2):
    #     x_array_dummy = np.linspace(list_edge_negative_y[i][0], list_edge_negative_y[i+1][0], Nx_per_path_btm, endpoint= False) + dx_path / 2
    #     x_array_path_btm = np.append(x_array_path_btm, x_array_dummy)
    
    # # 畝の外側の通路
    # x_array_negative_margin = np.linspace(list_edge_negative_y[0][0] - W_margin, list_edge_negative_y[0][0], Nx_per_path_btm, endpoint= False) + dx_path / 2
    # x_array_positive_margin = np.linspace(list_edge_negative_y[-1][0], list_edge_negative_y[-1][0] + W_margin, Nx_per_path_btm, endpoint= False) + dx_path / 2
    # x_array_path_btm = np.append(x_array_path_btm, x_array_negative_margin)
    # x_array_path_btm = np.append(x_array_path_btm, x_array_positive_margin)
    # x_array_path_btm = np.sort(x_array_path_btm)
    # x_path_btm, y_path_btm, z_path_btm = np.meshgrid(x_array_path_btm, y_array_path_btm, z_array_path_btm)

    # # 通路の底の点（反射光計算用）の専有面積
    # dA_path = dx_path * dy_path

    #####################################
    # 底（反射光計算用）の座標．
    dy_btm = ((list_edge_positive_y[0][1] + L_margin) - (list_edge_negative_y[0][1] - L_margin)) / Ny_per_btm
    y_array_btm = np.linspace(list_edge_negative_y[0][1] - L_margin, list_edge_positive_y[0][1] + L_margin, Ny_per_btm, endpoint= False) + dy_btm / 2
    z_array_btm = np.array([0])
    dx_btm = ((list_edge_negative_y[-1][0] + W_margin) - (list_edge_negative_y[0][0] - W_margin)) / Nx_per_btm
    x_array_btm = np.linspace(list_edge_negative_y[0][0] - W_margin, list_edge_negative_y[-1][0] + W_margin, Nx_per_btm, endpoint= False) + dx_btm / 2
    x_btm, y_btm, z_btm = np.meshgrid(x_array_btm, y_array_btm, z_array_btm, indexing='ij', sparse= True)

    # 底の点（反射光計算用）の専有面積
    dA_btm = dx_btm * dy_btm

    return x_row, y_row, z_row, x_btm, y_btm, z_btm, dV_row, dA_row, dA_btm

def create_grid_points_bottom(list_edge_negative_y, list_edge_positive_y, Nx_per_btm, Ny_per_btm, W_margin, L_margin):
    '''
    地表面からの反射光計算用に、gridを作成する。
    ただし、後の処理用に、indexing = "ij", sparse = Trueを使用していることに注意。

    入力
        | Nx_per_btm        --- x方向の地表面の分割数
        | Ny_per_btm        --- y方向の地表面の分割数
        | W_margin          --- x方向における、畝ゾーンからはみ出す"マージン"。このマージン部からの反射光も、のちに計算する。
        | L_margin          --- y方向における、畝ゾーンからはみ出す"マージン"。このマージン部からの反射光も、のちに計算する。
    '''
    # 底（反射光計算用）の座標．
    dy_btm = ((list_edge_positive_y[0][1] + L_margin) - (list_edge_negative_y[0][1] - L_margin)) / Ny_per_btm
    y_array_btm = np.linspace(list_edge_negative_y[0][1] - L_margin, list_edge_positive_y[0][1] + L_margin, Ny_per_path_btm, endpoint= False) + dy_btm / 2
    z_array_btm = np.array([0])
    dx_btm = ((list_edge_negative_y[-1][0] + W_margin) - (list_edge_negative_y[0][0] - W_margin)) / Nx_per_btm
    x_array_btm = np.linspace(list_edge_negative_y[0][0] - W_margin, list_edge_negative_y[-1][0] + W_margin, Nx_per_btm, endpoint= False) + dx_btm / 2
    x_btm, y_btm, z_btm = np.meshgrid(x_array_btm, y_array_btm, z_array_btm, indexing='ij', sparse= True)

    return x_btm, y_btm, z_btm

##################################################################
# 太陽の位置・放射についての関数
##################################################################
def cal_solar_elevation(ltt,lng,mrd,d_y,t_d):
    """calculate solar elevation.

    Keyword arguments:
        ltt -- latitude (radian)
        lng -- longitude (radian)
        mrd -- meridian (radian; for Japan, mrd=135/180* np.pi)
        d_y -- day of year
        t_d -- time of day

    Returns:
        solar_elevation -- solar elevation
        solar_azimuth -- solar azimuth (the angle along the horizon, with 0, 90, 180, 270 degrees corresponding to North, East, South, and West, respectively)
        time_of_sunrise -- time of sunrize
        time_of_sunset -- time of sunset
        daylength   
    """    

    solar_declination=-2*np.pi/360*23.45*np.cos(2*np.pi/360*(d_y+10)*(360/365.25))
    be=2*np.pi*(d_y-81)/365
    EoT=9.87*np.sin(2*be)-7.53*np.cos(be)-1.5*np.sin(be)
    #print("solar_declination=",solar_declination," be=",be," EoT=",EoT)

    lst=t_d - (mrd-lng) * (180/np.pi) /15 + EoT/60 #lst: local solar time
    # print("Local Solar Time =",lst)
    hour_angle=np.pi/12*(lst-12)
    aa =np.sin(ltt)*np.sin(solar_declination)
    bb = np.cos(ltt)*np.cos(solar_declination)
    solar_elevation=np.arcsin(aa+bb*np.cos(hour_angle))
    # solar_elevation=np.arcsin(np.sin(solar_declination)*np.sin(ltt)+np.cos(solar_declination)*np.cos(ltt)*np.cos(hour_angle))
    if hour_angle<0:
        solar_azimuth=np.pi-np.arccos((np.sin(ltt)*np.sin(solar_elevation)-np.sin(solar_declination))/(np.cos(ltt)*np.cos(solar_elevation)))
    else:
        solar_azimuth=np.pi+np.arccos((np.sin(ltt)*np.sin(solar_elevation)-np.sin(solar_declination))/(np.cos(ltt)*np.cos(solar_elevation)))

    S_time_of_sunset=12+12/np.pi*np.arccos(-np.tan(solar_declination)*np.tan(ltt)) #time of sunrise in solar time
    S_time_of_sunrise=24-S_time_of_sunset #time of sunset in solar time
    time_of_sunset=S_time_of_sunset+((mrd-lng)/(2*np.pi)*24-EoT/60)
    time_of_sunrise=S_time_of_sunrise+((mrd-lng)/(2*np.pi)*24-EoT/60)
    daylength=(time_of_sunset-time_of_sunrise)
    return pd.Series([solar_elevation,solar_azimuth, time_of_sunrise,time_of_sunset,daylength,lst])

def cal_solar_position(df, radius_sun_orbit, azimuth_row):
    '''
    Solar_elev, azm, radius_sun_orbitから，
    太陽の位置を計算する．
    太陽の位置は，"東西南北"座標および，"畝"座標上で計算する．
    なお，左手座標系を想定している（Gizjen and Goudriaan, 1989 と同じ座標系）ので，
    matplotlib上で出力するときは，ax.invert_yaxis()を使う必要がある．

    入力
        df -- Solar_elevやazmが入ったデータフレーム
        radius_sun_orbit -- 太陽の公転のみかけの半径．グラフに出力するときに必要．
        azimuth_row      -- "東西南北"座標から，畝がどれだけ回転しているか

    出力
        dfを出力する．ただし，
            azm_sun_in_row_coord -- 畝座標における，太陽のazimuth．もとのazimuthが北をゼロ，時計回りに+なことに注意．
            x_sun -- 太陽のx座標（マイナス:東，プラス:西）
            y_sun -- 太陽のy座標（プラス:北，マイナス:南）

            x_sun_row_coord -- 畝座標系における太陽のx座標（"東西南北"座標系よりも，azimuth_rowだけ回転している）
            y_sun_row_coord -- 畝座標系における太陽のy座標（"東西南北"座標系よりも，azimuth_rowだけ回転している）
    '''
    azimuth_row_rad = azimuth_row / 180 * np.pi
    df["H_sun"] = radius_sun_orbit * np.sin(df["Solar_elev"])
    df["dist_sun_xy"] = radius_sun_orbit * np.cos(df["Solar_elev"])
    df["x_sun"] = -df["dist_sun_xy"] * np.sin(df["azm"])
    df["y_sun"] = df["dist_sun_xy"] * np.cos(df["azm"])
    df["x_sun_row_coord"] = df["x_sun"] *np.cos(azimuth_row_rad) - df["y_sun"] * np.sin(azimuth_row_rad)
    df["y_sun_row_coord"] = df["y_sun"] *np.cos(azimuth_row_rad) + df["x_sun"] * np.sin(azimuth_row_rad)
    return df

def cal_solar_position_relative_to_row(azm_sun_row_coord, beta):
    '''
    畝座標におけるAzimuth (azm_sun_row_coord)とsolar elevation (beta)から，
    "畝座標"の点cを基準とした
    azimuth (azimuth_c), 
    solar elevation (beta_c)
    を計算する．
    
    なお，ここで，"畝座標"とは，北を基準としたazimuthではなく，畝の長辺方向を基準としたazimuth．
    Gijzen and Goudriaan (1989)では，azimuthを北をゼロとして，時計回りに計算している．
    野村の太陽高度計算でも，東西南北の座標を時計回りに回転させている．
    回転させた"結果"を，基準座標（畝座標）として用いている．
    例えば、azm_sun_row_coord = 10°とすると、南北のラインは、畝の長辺方向を基準線として、時計周りに10°だけ傾いている。
    逆に、南北のラインを基準に考えると、azm_sun_row_coord = 10°のとき、畝は半時計回りに10°だけ傾いている。
    '''
    #azm_sun_row_coord_south = azm_sun_row_coord + np.pi 
    beta_c = np.arcsin(np.cos(azm_sun_row_coord)*np.cos(beta))
    alpha_c = np.arccos(np.sin(beta) / np.cos(beta_c))
    return pd.Series([alpha_c, beta_c])

def cal_outside_diffuse_radiation(S_global, day_of_year, Solar_elev, S_sc = 1370):
    '''
    全天日射(S_global)から散乱PAR(I0_dif_h)を計算する。
    全天日射は気象庁データを使用予定。
    Spitters (1986) Iを参照。

    入力
        S_global        --- 全天日射 (W m-2)。気象庁データから取得する想定。
        S_sc            --- solar constant (1370 W m-2)
        day_of_year     --- 1月1日を1としたときの経過日数
        Solar_elev      --- 太陽高度 (rad)

    '''
    # 大気圏外, 水平地表面への短波放射S_o
    S_o = S_sc * (1 + 0.033 * np.cos((day_of_year / 365) * 2* np.pi)) * np.sin(Solar_elev)

    # S_global/S_oの値で場合分けして、短波放射, 1時間あたりの散乱光率を計算 (Spittersの式20)
    R = 0.847 - 1.61 * np.sin(Solar_elev) + 1.04 * (np.sin(Solar_elev))**2
    K = (1.47-R)/1.66
    atm_trans_ratio = S_global / S_o
    if (atm_trans_ratio <= 0.22):
        diffuse_ratio_short = 1
    elif (0.22 < atm_trans_ratio) & (atm_trans_ratio <= 0.35):
        diffuse_ratio_short = 1 - 6.4 * (atm_trans_ratio - 0.22)**2
        #print(diffuse_ratio_short)
    elif (0.35 < atm_trans_ratio)  & (atm_trans_ratio <= K):
        diffuse_ratio_short = 1.47 - 1.66 * atm_trans_ratio
    elif K < atm_trans_ratio:
        diffuse_ratio_short = R
    # 太陽光方向付近の散乱光は、ほぼ直達光とみなせるから、その値を補正する。
    # Spitters (1986)では日単位のdiffuse ratioを修正しているが、その数式 (式9)が、一時間あたりでも当てはまるものと仮定する。
    diffuse_ratio_short_modified = diffuse_ratio_short / (1 + (1 - diffuse_ratio_short**2) * (np.cos(np.pi/2 - Solar_elev))**2 * (np.cos(Solar_elev))**3)

    # PARの散乱率を計算する。
    diffuse_ratio_PAR = (1 + 0.3 * (1 - diffuse_ratio_short) ** 2) * diffuse_ratio_short_modified

    # diffuse PAR (I0_dif_h_out) およびdirectPAR (I0_beam_h_out)を計算する。
    # ハウス内のdiffuse PAR (I0_dif_h)およびdirect PAR (I0_beam_h)については、
    # I0_dif_h_outおよびI0_beam_h_outに、フィルムや骨材による補正を加えることで算出する。
    # なお、野外の合計PAR (I0_h_out)については、I0_h_out = 2.0 * S_global で算出できる想定（Jacovides）。
    I0_h_out      = 2.0 * S_global
    I0_dif_h_out  = diffuse_ratio_PAR * I0_h_out
    I0_beam_h_out = (1 - diffuse_ratio_PAR) * I0_h_out
    
    #I0_dif_h_out, I0_beam_h_out, 
    return pd.Series([I0_beam_h_out, I0_dif_h_out])

def cal_inside_radiation(I0_beam_h_out, I0_dif_h_out, transmission_coef_cover, transmission_coef_structure, beam_to_dif_conversion_ratio_cover):
    '''
    ハウス外の日射から、ハウス内の日射を計算する。
    
    入力
        I0_beam_h_out               --- 野外の直達PAR
        I0_dif_h_out                --- 野外の散乱PAR
        transmission_coef_cover     --- ハウス外張りのPAR透過率 (0 - 1)
        transmission_coef_structure --- ハウス骨材のPAR透過率 (0 - 1)
        beam_to_dif_conversion_ratio_cover      --- 直達光から散乱光への変換率 (0 - 1)
    '''
    transmission_coef   = transmission_coef_cover * transmission_coef_structure
    I0_beam_h_in        = I0_beam_h_out * (1 - beam_to_dif_conversion_ratio_cover) * transmission_coef
    I0_dif_h_in         = I0_dif_h_out * transmission_coef + I0_beam_h_out * beam_to_dif_conversion_ratio_cover * transmission_coef
    return pd.Series([I0_beam_h_in, I0_dif_h_in])

##################################################################
# path lengthの計算
##################################################################

def cal_path_length_main(x_c_original, y_c_original, z_c_original, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y):
    '''
    ある点[x_c, y_c, z_c]において，[azm_sun_row_coord, beta]方向から差し込んでくる光線の，
    畝を貫通する光路長（path_length）を計算する．
    入力
        x_c_original, y_c_original, z_c_original    --- 今問題としている点の座標．ハウスの中心を原点とする．
        azm_sun_row_coord                           --- 畝座標での太陽のazimuth
        beta                                        --- 太陽高度．°ではなくradian.
        H_row                                       --- 畝の高さ
        W_row                                       --- 畝幅
        W_path                                      --- 歩道幅
        list_edge_negative_y                        --- 畝の端点（間口側）の座標群 (yが負; 奥)．
                                                        例えば[0][1]は，左端のy座標．[-1][2]は，右端のz座標．
        list_edge_positive_y                        --- 畝の端点（間口側）の座標群 (yが正; 手前)

    Intermediate variables
        alpha_c         --- 畝座標におけるconverted azimuth.

        beta_c          --- 畝座標におけるconverted solar elevation.

        x_edge          --- 一番外側の畝の，外側のx座標．

        x_to_edge       --- x_cからx_edgeまでの距離．
    
        x               --- 畝の端からのx_cの距離 (azm_sun_row_coodによって変わる).
                            x_cが畝の外 (通路)に存在するときは0.

        x_o             --- x_cが畝の外 (通路)に存在するとき，畝の端からの距離 (azm_sun_row_coodによって変わる)．
                            x_cが畝の中に存在するときは0.

        x_hor           --- x_cからpath_startまでの水平距離を，x-z平面に投影したもの．

        x_to_path_start --- pathの計算始点までの距離．以下に等しい．
                            光線がうねの上面をつきぬけるとき            x_hor
                            光線がうねの前面 or 背面をつきぬけるとき     x_to_yedge
                            光線がうねの側面をつきぬけるとき            x_to_edge
    '''

    x_edge_negative = list_edge_negative_y[0][0]
    x_edge_positive = list_edge_negative_y[-1][0]
    y_edge_negative = list_edge_negative_y[0][1]
    y_edge_positive = list_edge_positive_y[0][1]
    
    #############################
    # 奥行き方向(y_c_original)が畝の前面・背面よりもさらに外側に存在する場合，
    # (x_c_original, y_c_original, z_c_original)を畝の前面・背面まで移動させて，path_lengthを計算する．
    # この処理は，地表面に降り注ぐ光強度を計算する際，畝ゾーン外の点があることから必要となった．
    if (y_edge_negative <= y_c_original) & (y_c_original <= y_edge_positive): # はみ出していない
        x_c = x_c_original
        y_c = y_c_original
        z_c = z_c_original
    else: # はみ出している
        if y_c_original < y_edge_negative: # y_c_originalが奥行き側にはみ出している
            y_c = y_edge_negative
        elif y_edge_positive < y_c_original: # y_g_originalが手前側にはみ出している
            y_c = y_edge_positive
        x_c = -np.tan(azm_sun_row_coord) * (y_c - y_c_original) + x_c_original
        z_c = np.tan(beta)/ np.cos(azm_sun_row_coord) * (y_c - y_c_original) + z_c_original

    xyz_isin_row, x_isin_row, y_isin_row, z_isin_row, interval_xisin = check_xyz_isin_row(x_c, y_c ,z_c, list_edge_negative_y, list_edge_positive_y)
    # print("azm_sun_row_coord, beta = {0:4.2f}°, {1:4.2f}°".format(azm_sun_row_coord/np.pi*180, beta/np.pi*180))
    # print("original ", x_c_original, y_c_original, z_c_original)
    # print("y_edgeでの値に修正 ", x_c, y_c, z_c)

    #############################

    alpha_c, beta_c = cal_solar_position_relative_to_row(azm_sun_row_coord, beta)
    x_hor = (H_row - z_c) * np.tan(alpha_c)
    alpha_y = np.arctan(np.sin(beta)/np.sin(beta_c)) # Gizjen (1989) Fig.1に書き込んだ角度．

    # (azm_sun_row_coord/np.pi*180) < 180であれば，太陽は東側にある．
    # また，x_edgeは，ハウス内の畝の最端のx座標を示す．
    # x_horがハウスの両端の畝よりはみ出ない場合は，Gijzen & Goudriaan (1989)の数式でOK.
    # はみ出る場合について考える．
    if azm_sun_row_coord/np.pi*180 <= 180:
        x_edge = x_edge_negative
        x_to_edge = x_c - x_edge            
    else:
        x_edge = x_edge_positive
        x_to_edge = x_edge - x_c

    # 太陽が畝座標の後側にあるか，手前側にあるか．
    if (90 <= (azm_sun_row_coord/np.pi*180)) & ((azm_sun_row_coord/np.pi*180) <= 270): # 畝の後ろ側
        y_edge = y_edge_negative
    else: # 畝の前側
        y_edge = y_edge_positive
    # y = y_edgeにおけるx
    x_at_y_edge = - (y_edge - y_c) * np.tan(alpha_y) * np.sin(azm_sun_row_coord) / np.tan(beta) + x_c 

    #####################
    # Gizjen (1989) Fig.1参照
    BA = (H_row - z_c) / np.tan(alpha_y) 
    BD = -(H_row - z_c) / np.tan(beta) * np.sin(azm_sun_row_coord) 

    x_at_H_row = x_c + BD
    y_at_H_row = y_c + BA
    #print("azm_sun_row_coord = {0:3.2f}, beta = {1:3.2f}, x_at_H_row = {2:3.2f}, y_at_H_row = {3:3.2f}".format(azm_sun_row_coord*180/np.pi, beta*180/np.pi, x_at_H_row, y_at_H_row))

    # x_edgeにおける光線のy, zの値を計算して，pathの計算式を場合分けする．
    y_at_x_edge = -(x_edge -x_c) * np.tan(beta) /(np.sin(azm_sun_row_coord)*np.tan(alpha_y)) + y_c
    z_at_x_edge = -(x_edge -x_c) * np.tan(beta) / np.sin(azm_sun_row_coord) + z_c

    # x_c, y_c, z_cが畝内なのか，それとも通路なのか
    if not xyz_isin_row: # 通路内．x_outside_row_from_leftは"通路"の左端からの距離であることに注意!!
        #print("(x_c, y_c, z_c) = {}, {}, {}: この点は畝の外側または通路の点です．".format(x_c, y_c, z_c))
        x = 0
        if interval_xisin == -1: # x_cは両端の畝よりもなお外側に存在する．
            x_outside_row_from_right  = x_edge_negative - x_c # 左端の畝から点[x_c, y_c, z_c]までの距離
            x_outside_row_from_left   = x_c - x_edge_positive  # 右端の畝から点[x_c, y_c, z_c]までの距離
        else:
            x_outside_row_from_left =  x_c - list_edge_negative_y[interval_xisin][0] # 点[x_c, y_c, z_c]が属する通路の左端からの距離．interval_x_isinには奇数（通路の左端）が入る．
            x_outside_row_from_right = W_path - x_outside_row_from_left # 点[x_c, y_c, z_c]が属する通路の右端からの距離
        
        # (azm_sun_row_coord/np.pi*180) < 180であれば，太陽は東側にある．このときのpath_lengthの計算にはx_outside_row_leftを使う．
        # これは畝内に(x_c, y_c, z_c)があるときとは逆であることに注意！
        if azm_sun_row_coord/np.pi*180 <= 180:
            x_o = x_outside_row_from_left
        else:
            x_o = x_outside_row_from_right

    if xyz_isin_row: # 畝内
        x_o = 0
        x_in_row_from_left =  x_c - list_edge_negative_y[interval_xisin][0] # 点[x_c, y_c, z_c]が属する畝の左端からの距離
        x_in_row_from_right = W_row - x_in_row_from_left # 点[x_c, y_c, z_c]が属する畝の右端からの距離

        # (azm_sun_row_coord/np.pi*180) < 180であれば，太陽は東側にある．このときのpath_lengthの計算にはx_in_row_rightを使う．
        if azm_sun_row_coord/np.pi*180 <= 180:
            x = x_in_row_from_right
        else:
            x = x_in_row_from_left

    # pathの始点がどこにあるのかによって場合分けして，"x_to_path_start"を計算する．    
    if (y_edge_negative <= y_at_H_row) & (y_at_H_row <= y_edge_positive):
        if (x_edge_negative <= x_at_H_row) & (x_at_H_row <= x_edge_positive): # このとき，光線はうねの上面を突き抜けるので，論文の式が使える．
            #print("うねの上面をつきぬけた")
            path_start = "upper"
            x_to_path_start = x_hor

        else: # このとき，光線はうねの側方をつきぬけるので，カスタムの式を使う．
            #print("うねの側面をつきぬけた")
            path_start = "side"
            x_to_path_start = x_to_edge

    else:
        if (x_edge_negative <= x_at_H_row) & (x_at_H_row <= x_edge_positive): # このとき，光線はうねの前後をつきぬけるので，カスタムの式を使う．
            #print("うねの前後をつきぬけた")
            path_start = "front/back"
            x_to_y_edge = abs(x_at_y_edge - x_c)
            x_to_path_start = x_to_y_edge

        else:
            #print("x_psもy_psもハウスの外側にある")
            if (x_at_y_edge < x_edge_negative) | (x_edge_positive < x_at_y_edge): # このとき，光線はうねの側面をつきぬけている
                #print("うねの側面をつきぬけた")
                path_start = "side2"
                x_to_path_start = x_to_edge

            else: # このとき，光線はうねの前面 or 後ろ面をつきぬけている
                #print("うねの前後をつきぬけた")
                path_start = "front/back2"
                x_to_y_edge = abs(x_at_y_edge - x_c)
                x_to_path_start = x_to_y_edge
                
    ###################################################################################################            
    x_residual = (x_to_path_start + x - x_o) % (W_path + W_row)
    N_unit = (x_to_path_start + x - x_o) // (W_path + W_row)
    
    if x_residual <= W_row:
        Pr_dash = (N_unit * W_row - x + x_residual) / np.sin(alpha_c)
    else:
        Pr_dash = ((N_unit + 1) * W_row - x) / np.sin(alpha_c)

    #print("x_residual = {0:3.1f}, N_unit = {1:3.1f}, Pr_dash = {2:3.1f}, alpha_c = {3:3.1f}".format(x_residual, N_unit, Pr_dash, alpha_c/np.pi * 180))

    # # 確認用に，path_startにおける座標を計算してプロットする．
    if path_start == "upper":
        x_at_path_start  = - (H_row - z_c) * np.sin(azm_sun_row_coord) / np.tan(beta) + x_c
        y_at_path_start  = (H_row - z_c) / np.tan(alpha_y) + y_c
        z_at_path_start = H_row
        #color = "r"
    elif (path_start == "side")|(path_start == "side2"):
        x_at_path_start = x_edge
        y_at_path_start = y_at_x_edge
        z_at_path_start = z_at_x_edge
        #color = "y"
    elif (path_start == "front/back") | (path_start == "front/back2"):
        x_at_path_start  = -(y_edge - y_c) * np.tan(alpha_y) * np.sin(azm_sun_row_coord) / np.tan(beta) + x_c
        y_at_path_start  = y_edge
        z_at_path_start = (y_edge - y_c) * np.tan(alpha_y) + z_c
        #color = "b"                  
    Pr = Pr_dash / np.cos(beta_c)

    # 色付用．
    dummy1 = np.sqrt((x_c - x_edge_negative)**2 + (y_c - y_edge_negative)**2 + (z_c - list_edge_negative_y[0][2])**2)
    dummy2 = np.sqrt((x_c - x_edge_positive)**2 + (y_c - list_edge_negative_y[-1][1])**2 + (z_c - list_edge_negative_y[-1][2])**2)
    dummy3 = np.sqrt((x_c - list_edge_positive_y[0][0])**2 + (y_c - y_edge_positive)**2 + (z_c - list_edge_positive_y[0][2])**2)
    dummy4 = np.sqrt((x_c - list_edge_positive_y[-1][0])**2 + (y_c - list_edge_positive_y[-1][1])**2 + (z_c - list_edge_positive_y[-1][2])**2)

    Pr_max = np.max([dummy1, dummy2, dummy3, dummy4])
    #ax.plot3D([x_c_original, x_at_path_start], [y_c_original, y_at_path_start], [z_c_original, z_at_path_start],  lw = 1, c = cm.autumn(Pr/Pr_max)) # "saddlebrown"
    if Pr<0:
        # たまに-0.000001みたいな値になることがあるので，0とする．
        #print("path length = {0}なので0にする． when azm_sun_row_coord = {1:3.2f} and Solar_elev = {2:3.2f}.".format(Pr, azm_sun_row_coord/np.pi*180, beta/np.pi*180))
        Pr = 0
    if (Pr > Pr_max):
        print("何かがおかしい！")
        print("azm_sun_row_coord, beta = {0:4.2f}°, {1:4.2f}°".format(azm_sun_row_coord/np.pi*180, beta/np.pi*180))
        print("(x_c_original, y_c_original, z_c_original) = ({0:3.2f}, {1:3.2f}, {2:3.2f})".format(x_c_original, y_c_original, z_c_original))
        print("(x_c, y_c, z_c) = ({0:3.2f}, {1:3.2f}, {2:3.2f})".format(x_c, y_c, z_c))
        print("xyz_isin_row = {}".format(xyz_isin_row))
        print("interval_xisin = {}".format(interval_xisin))
        print("x_at_y_edge = {}".format(x_at_y_edge))
        print("y_at_H_row = {}".format(y_at_H_row))
        print("x_o ={}".format(x_o))
        print("x_to_path_start = {}".format(x_to_path_start))
        print("x_residual = {0:10.5f} m".format(x_residual))
        print("W_p = {0:10.5f} m".format(W_path))
        print("path_start = {}".format(path_start))
        print("path length = {0:3.2f} m".format(Pr))
        print("Pr_max = {0:3.2f} m".format(Pr_max))
        print("alpha_c = {0:3.2f}°, beta_c = {1:3.2f}°".format(alpha_c/np.pi*180, beta_c/np.pi*180))
        print("")
    return Pr

def cal_path_length(x_c_original, y_c_original, z_c_original, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y):
    '''
    ある点[x_c, y_c, z_c]において，[azm_sun_row_coord, beta]方向から差し込んでくる光線の，
    畝を貫通する光路長（path_length）を計算する．
    (x_c, y_c, z_c)が畝ゾーンにあるうちは, cal_path_length_main()を使えば良かったが,
    畝ゾーン外の点のpath lengthを処理する際に, 場合分けの必要が生じた. 
    本関数では, その"場合分け"を扱う．

    入力
        x_c_original, y_c_original, z_c_original    --- 今問題としている点の座標．ハウスの中心を原点とする．
        azm_sun_row_coord                           --- 畝座標での太陽のazimuth (radian). 
        beta                                        --- 太陽高度. °ではなくradian.
        H_row                                       --- 畝の高さ
        W_row                                       --- 畝幅
        W_path                                      --- 歩道幅
        list_edge_negative_y                        --- 畝の端点（間口側）の座標群 (yが負; 奥)．
                                                        例えば[0][1]は，左端のy座標．[-1][2]は，右端のz座標．
        list_edge_positive_y                        --- 畝の端点（間口側）の座標群 (yが正; 手前)

    '''
    if beta > 0:
        # 場合分け．Path lengthの計算をまとめるNo.4, 5を参照のこと．
        x_edge_negative = list_edge_negative_y[0][0]
        x_edge_positive = list_edge_negative_y[-1][0]
        y_edge_negative = list_edge_negative_y[0][1]
        y_edge_positive = list_edge_positive_y[0][1]

        x_at_H_row = -np.cos(beta) * np.sin(azm_sun_row_coord) / np.sin(beta) * (H_row - z_c_original) + x_c_original
        y_at_H_row = np.cos(beta) * np.cos(azm_sun_row_coord) / np.sin(beta) * (H_row - z_c_original) + y_c_original
        #print("x_at_H_row = {0:3.2f} m, y_at_H_row = {1:3.2f}".format(x_at_H_row, y_at_H_row))

        # 奥行き方向(y_c_original)が畝の前面・背面よりもさらに外側に存在する場合，
        # (x_c_original, y_c_original, z_c_original)を畝の前面・背面まで移動させて，path_lengthを計算する．
        # この処理は，地表面に降り注ぐ光強度を計算する際，畝ゾーン外の点があることから必要となった．
        if (y_edge_negative <= y_c_original) & (y_c_original <= y_edge_positive): # はみ出していない
            x_c = x_c_original
            y_c = y_c_original
            z_c = z_c_original
        else: # はみ出している
            if y_c_original < y_edge_negative: # y_c_originalが奥行き側にはみ出している
                y_c = y_edge_negative
            elif y_edge_positive < y_c_original: # y_g_originalが手前側にはみ出している
                y_c = y_edge_positive
            x_c = -np.tan(azm_sun_row_coord) * (y_c - y_c_original) + x_c_original
            z_c = np.tan(beta)/ np.cos(azm_sun_row_coord) * (y_c - y_c_original) + z_c_original

        # 以後，(x_c, y_c, z_c)を使って検討する．

        Pr = -1
        if z_c < 0: # この場合，y_cが畝の前後をはみ出していて，光が畝と逆方向から差し込んでいるということ．なので, Pr = 0.
            Pr = 0
        elif z_c > H_row: # この場合，y_cが畝の前後をはみ出していて，光が畝方向から差し込んでいるものの，太陽高度が高すぎて畝に当たらない．なので，Pr = 0
            Pr = 0
        else:
            if (x_c < x_edge_negative):
                #print("pattern 8")            
                if (x_at_H_row < x_edge_negative):
                    Pr = 0
                else:
                    y_at_xedge = - np.cos(azm_sun_row_coord) / np.sin(azm_sun_row_coord) * (x_edge_negative - x_c) + y_c
                    if (180 <= azm_sun_row_coord/np.pi * 180) & (azm_sun_row_coord/np.pi * 180 <= 270): 
                        if y_at_xedge < y_edge_negative: # ぎり外側．
                            Pr = 0
                    elif (270 <= azm_sun_row_coord/np.pi * 180) & (azm_sun_row_coord/np.pi * 180 <= 360):
                        if y_at_xedge > y_edge_positive: # ぎり外側．
                            Pr = 0
            elif (x_edge_positive < x_c):
                #print("pattern 4")
                if (x_edge_positive < x_at_H_row):
                    Pr = 0
                else:
                    y_at_xedge = - np.cos(azm_sun_row_coord) / np.sin(azm_sun_row_coord) * (x_edge_positive - x_c) + y_c
                    if (90 <= azm_sun_row_coord/np.pi * 180) & (azm_sun_row_coord/np.pi*180 <= 180): 
                        if y_at_xedge < y_edge_negative: # ぎり外側．
                            Pr = 0
                    elif (0 <= azm_sun_row_coord/np.pi * 180) & (azm_sun_row_coord/np.pi * 180 <= 90):
                        if y_at_xedge > y_edge_positive: # ぎり外側．
                            Pr = 0

            # if (x_c < x_edge_negative):
            #     if (y_c < y_edge_negative): # 1
            #         #print("pattern 1")
            #         if (x_at_H_row < x_edge_negative) | (y_at_H_row < y_edge_negative):
            #             Pr = 0
                        
            #     elif (y_edge_negative <= y_c) & (y_c <= y_edge_positive): # 8
            #         #print("pattern 8")            
            #         if (x_at_H_row < x_edge_negative):
            #             Pr = 0

            #     elif (y_edge_positive < y_c): # 7
            #         #print("pattern 7")
            #         if (x_at_H_row < x_edge_negative) | (y_edge_positive < y_at_H_row):
            #             Pr = 0

            # elif (x_edge_positive < x_c):
            #     if (y_c < y_edge_negative): # 3
            #         #print("pattern 3")
            #         if (x_edge_positive < x_at_H_row) | (y_at_H_row < y_edge_negative):
            #             Pr = 0

            #     elif (y_edge_negative <= y_c) & (y_c <= y_edge_positive): # 4
            #         #print("pattern 4")
            #         if (x_edge_positive < x_at_H_row):
            #             Pr = 0

            #     elif (y_edge_positive < y_c): # 5
            #         #print("pattern 5")
            #         if (x_edge_positive < x_at_H_row) | (y_edge_positive < y_at_H_row):
            #             Pr = 0

            # elif (x_edge_negative <= x_c) & (x_c <= x_edge_positive):
            #     if (y_c < y_edge_negative): # 2
            #         # print("pattern 2")
            #         if (y_at_H_row < y_edge_negative):
            #             Pr = 0

            #     elif (y_edge_positive < y_c): # 6
            #         #print("pattern 6")
            #         if (y_edge_positive < y_at_H_row):
            #             Pr = 0

        # 上記以外の場合
        if Pr == -1:
            Pr = cal_path_length_main(x_c_original, y_c_original, z_c_original, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y) 
        # print("azimuth_sun_coord = {0:3.2f}°, Solar_elev = {1:3.2f}°のとき，点(x_c, y_c, z_cオリジナル) = ({2:3.1f}, {3:3.1f}, {4:3.1f})へのPr = {5:3.2f} m".format(azm_sun_row_coord/np.pi*180, beta/np.pi*180, x_c_original, y_c_original, z_c_original, Pr)) 
    else: # 太陽が登っていないときはPr = 0とする。
        Pr = 0
    return Pr

def check_xyz_isin_row(x_c, y_c, z_c, list_edge_negative_y, list_edge_positive_y):
    '''
    ある点[x_c, y_c, z_c]は畝の内部に存在する必要がある．その確認．

    return
        interval_xisin          --- (x_c, y_c, z_c)がどの畝 or 通路に属するのか．偶数であれば畝に属する．-1であれば両端の畝よりも外に属する．
    '''
    #print("(x_c, y_c, z_c) = ({}, {}, {})".format(x_c, y_c, z_c))
    
    # x座標の確認
    interval_xisin = -1 #x座標がどのintervalに属しているのか．偶数であれば畝の中に属する．
    for i, edge in enumerate(list_edge_negative_y[:-1]):
        #print(i, list_edge_negative_y[i])
        # 畝の内部かどうか
        if list_edge_negative_y[i][0] < x_c:
            if list_edge_negative_y[i+1][0] > x_c:
                interval_xisin = i
    #print(i+1, list_edge_negative_y[i+1])

    # 畝の境界点も拾う．
    x_list = [edge[0] for edge in list_edge_negative_y]
    if np.any(np.isin(x_list, x_c)):
        x_isin_boundary = np.isin(x_list, x_c).nonzero()[0].item()
        if x_isin_boundary % 2 == 0:
            interval_xisin = x_isin_boundary
        elif x_isin_boundary % 2 == 1:
            interval_xisin = x_isin_boundary - 1
    else:
        x_isin_boundary = np.nan

    # x座標の判定
    if (interval_xisin % 2 == 0):
        x_isin_row = True
    else:
        x_isin_row = False

    # y座標の確認
    if (y_c >= list_edge_negative_y[0][1]) & (y_c <= list_edge_positive_y[0][1]):
        y_isin_row = True
    else:
        y_isin_row = False
    
    # z座標の確認
    if (z_c <= list_edge_negative_y[0][2]) & (z_c >= 0):
        z_isin_row = True
    else:
        z_isin_row = False
    
    # (x_c, y_c, z_c)が畝内部の点かどうかを判定
    if x_isin_row * y_isin_row * z_isin_row == 1:
        xyz_isin_row = True
    else:
        xyz_isin_row = False

    #print("xyz_isin_row = ", xyz_isin_row)
    return xyz_isin_row, x_isin_row, y_isin_row, z_isin_row, interval_xisin

##################################################################
# 直達光の計算
##################################################################
def cal_beam_radiation(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, LD, h_to_v_ratio, I0_beam_h, scatter_coef):
    '''
    ある地点における直達光の平均値を計算する．
    水平面での値とする. 
    この値には, 以下が含まれる.

    - sunlit葉に当たる直達光
    - sunlit葉およびshaded葉に当たる, 直達光由来の散乱光

    入力
        I0_beam_h       --- 群落上部における，水平面1m2あたりのPPFD (umol m^-2 s^-1).
        beta            --- 太陽高度．°ではなくradian.
        path_length     --- 任意の点までのpath_length (m).
                            よって，通常のLAI（鉛直下向きに積算したもの）とは異なる．
        h_to_v_ratio    --- ellipsoidal leaf angle distributionの横軸長/縦軸長
        scatter_coef    --- scattering coefficient
        LD              --- 畝における葉密度 (m^2 leaf m^-3)
    
    intermediate variables
        L_t             --- 光線がcanopyを貫通した距離を, LAIで表したもの．
                            光線方向に対する垂直面において積算したLAI(m^2 m^-2 beam cross-section)．
    '''
    # path_length内で貫通するbeam垂直面1m2あたりの葉面積
    if beta <=0:
        I_beam_h =0
        return I_beam_h

    O_av = GF.cal_ellipsoidal_G_from_beta_rad(h_to_v_ratio, beta)
    Pr = cal_path_length(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y)
    L_t = LD * Pr
    I_beam_h = I0_beam_h*np.exp(-O_av * (1-scatter_coef)**0.5 * L_t)
    return I_beam_h

def cal_beam_fraction(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, LD, O_av):
    '''
    ある地点における直達光の面積割合を計算する．

    入力
        beta            --- 太陽高度．°ではなくradian.
        path_length     --- 任意の点までのpath_length (m).
                            よって，通常のLAI（鉛直下向きに積算したもの）とは異なる．
        O_av            --- 光線と鉛直方向への，葉の投影比率．sphericalの場合は0.5
        LD              --- 畝における葉密度 (m^2 leaf m^-3)
    
    intermediate variables
        L_t             --- 光線がcanopyを貫通した距離を, LAIで表したもの．
                            光線方向に対する垂直面において積算したLAI(m^2 m^-2 beam cross-section)．
    '''
    # path_length内で貫通するbeam垂直面1m2あたりの葉面積
    Pr = cal_path_length(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y)
    L_t = LD * Pr
    f_beam = np.exp(-O_av * L_t)
    return f_beam

##################################################################
# 散乱光の計算
##################################################################

def cal_diffuse_radiation_by_trapezoid(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, h_to_v_ratio, scatter_coef, LD, azm, beta):
    '''
    diffuse radiationを計算する．
    cal_two_dimensional_integral_by_trapezoidを使う想定．
    引き数の最後 (azm, beta)が，積分の変数． 
    '''
    Pr = cal_path_length(x_c,y_c,z_c, azm, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y)
    L_t = Pr * LD
    O_av = GF.cal_ellipsoidal_G_from_beta_rad(h_to_v_ratio, beta)
    f = (I0_dif_h/np.pi) *  np.exp(-O_av * (1 - scatter_coef)**0.5 * L_t) * np.sin(beta) * np.cos(beta)
    return f   

##################################################################
# reflected radiation (地面からの反射光)
##################################################################

def cal_reflected_radiation_by_trapezoid(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, h_to_v_ratio, scatter_coef, reflect_coef_ground, LD, interp_I_g_h, alpha, theta):
    '''
    以下の台形公式関数を使って，地表面(z_g = 0)におけるreflected radiationを計算する.
    cal_two_dimensional_integral_by_trapezoid(N1, N2, a1, b1, a2, b2, max_iter, acceptable_absolute_error, func, *input_for_func)

    まず，(alpha, theta)における(x_g, y_g)を計算して，その点から(x_c, y_c, z_c)への反射光を計算する．
    (x_g, y_g)に降り注ぐ光強度I_g_hは, grid interpolatorの線形補間により計算する. 

    入力
        | I_g_h                 --- (x_g, y_g, z_g)に，水平面に照射する下向きの光の強度．
        | interp_I_g_h          --- RegularGridInterpolatorによるinterpolating function. この関数に任意の(x_g, y_g, 0)を代入すれば，線形補間によってI_g_hが算出される．ただし，枠外はゼロ．
        | alpha                 --- (x_c, y_c, 0)を原点としたときの，(x_g, y_g, 0)のazimuth. 積分変数 (0 - 2*np.pi). (a2, b2, N2)に対応する．
        | theta                 --- (x_c, y_c, z_c)から(x_g, y_g, z_g)への，みかけ上の天頂角 (rad)．積分変数 (0 - np.pi/2). (a1, b1, N1)に対応する．
    '''
    # (x_g, y_g)をalpha, thetaから計算する．
    r_projected = z_c * np.tan(theta)
    x_g = x_c - r_projected * np.sin(alpha)
    y_g = y_c + r_projected * np.cos(alpha)
    I_g_h = interp_I_g_h(np.array([x_g, y_g, 0]))[0]   

    Pr = cal_path_length_reflected_radiation(x_c, y_c, z_c, W_row, W_path, list_edge_negative_y, list_edge_positive_y, x_g, y_g, 0)
    L_t = Pr * LD

    # O_avを計算する。ただし、反射光計算では、鉛直からのなす角theta (rad)でこれまで議論してきたので、beta (rad)に直す。
    beta = np.pi/2 - theta
    O_av = GF.cal_ellipsoidal_G_from_beta_rad(h_to_v_ratio, beta)
    f = reflect_coef_ground / np.pi * I_g_h * np.exp(-O_av * (1- scatter_coef)**0.5 * L_t) * np.sin(theta) * np.cos(theta)
    distance = np.sqrt((x_c - x_g)**2 + (y_c - y_g)**2 + (z_c - 0)**2)
    if round(distance, 3) < round(Pr, 3):
        print()
        print("反射光の計算で何かがおかしい！")
        print("直線距離 = {0:4.2f}, (x_g, y_g, z_g) = ({1:4.2f}, {2:4.2f}, 0)から(x_c, y_c, z_c) = ({3:4.2f}, {4:4.2f}, {5:4.2f})への反射． \nI_g_h = {6:4.2f}, Pr = {7:4.2f}, theta = {8:4.2f}°, f = {9:4.2f}".format(distance, x_g, y_g, x_c, y_c, z_c, I_g_h, Pr, theta/np.pi * 180, f))

        print()     

    return f

def cal_path_length_reflected_radiation(x_c, y_c, z_c, W_row, W_path, list_edge_negative_y, list_edge_positive_y, x_g, y_g, z_g):
    '''点(x_g, y_g, 0)から点(x_c, y_c, z_c)への反射光強度を計算するにあたって, path lengthを計算する.
    
    "これまでの理解をまとめる No.12, 13, 14"
    "地面からの反射光の計算 No.1, No.2 
    を参照のこと！

    まず，点(x_g, y_g, z_g)が，ハウスの畝-通路群のどこに存在するのかを確認する．
    奥行き方向(y_g)が畝の前面・背面よりもさらに外側に存在する場合，
    (x_g, y_g, z_g)を(x_g_dash, y_g_dash, z_g_dash)に移動させた点からpath lengthを計算する．
    
    '''
    x_edge_negative = list_edge_negative_y[0][0]
    x_edge_positive = list_edge_negative_y[-1][0]
    y_edge_negative = list_edge_negative_y[0][1]
    y_edge_positive = list_edge_positive_y[0][1]

    ################################
    # Prの計算
    # 点(x_g, y_g, z_g)から点(x_c, y_c, z_c)までの距離r および角度beta (x_c, y_c, z_cの"高度")
    r = np.sqrt ((x_c - x_g)**2 + (y_c - y_g)**2 + (z_c - z_g)**2)

    # 計算用の角度 (これまでの理解をまとめる No.13参照)
    if (z_c - z_g == 0):
        alpha_c = np.pi/2
    else:
        alpha_c = np.arctan(abs((x_c - x_g) / (z_c - z_g)))

    beta_c  = np.arctan(abs((y_c - y_g) / np.sqrt((x_c - x_g)**2 + (z_c - z_g)**2)))

    # 奥行き方向(y_g)が畝の前面・背面よりもさらに外側に存在する場合，
    # (x_g, y_g, z_g)を(x_g_dash, y_g_dash, z_g_dash)に移動させた点からpath lengthを計算する．
    # 点(x_g, y_g, z_g)が畝の前面・背面をはみ出してないかどうか
    if (y_edge_negative <= y_g) & (y_g <= y_edge_positive): # はみ出していない
        x_g_dash = x_g
        y_g_dash = y_g
        z_g_dash = z_g
    else: # はみ出している
        if y_g < y_edge_negative: # y_gが奥行き側にはみ出している
            y_g_dash = y_edge_negative
        elif y_edge_positive < y_g: # y_gが手前側にはみ出している
            y_g_dash = y_edge_positive
        vector_g_to_c = np.array([(x_c - x_g), (y_c - y_g), (z_c - z_g)])
        x_g_dash = (vector_g_to_c[0] / vector_g_to_c[1]) * (y_g_dash - y_g) + x_g
        z_g_dash = (vector_g_to_c[2] / vector_g_to_c[1]) * (y_g_dash - y_g) + z_g        

    # 点(x_g_dash, y_g_dash, z_g_dash)と点(x_c, y_c, z_c)との位置関係の確認．

    # 点(x_g_dash, y_g_dash, z_g_dash)が畝の中にあるか，外にあるか．
    # interval_xisinが偶数ならば，(x_g, y_g, z_g)は畝の中に存在する．
    xyz_isin_row, x_isin_row, y_isin_row, z_isin_row, interval_xisin = check_xyz_isin_row(x_g_dash, y_g_dash, z_g_dash, list_edge_negative_y, list_edge_positive_y)
    
    # x_c, y_c, z_cが畝内なのか，それとも通路なのか
    if not xyz_isin_row: # 通路内．x_outside_row_from_leftは"通路"の左端からの距離であることに注意!!
        x = 0
        if interval_xisin == -1: # x_g_dashは両端の畝よりもなお外側に存在する．
            x_outside_row_from_right  = x_edge_negative - x_g_dash # 左端の畝から点[x_g_dash, y_g_dash, z_g_dash]までの距離
            x_outside_row_from_left = x_g_dash - x_edge_positive # 右端の畝から点[x_g_dash, y_g_dash, z_g_dash]までの距離
        else: 
            x_outside_row_from_left  =  x_g_dash - list_edge_negative_y[interval_xisin][0] # 点[x_g_dash, y_g_dash, z_g_dash]が属する通路の左端からの距離
            x_outside_row_from_right = W_path - x_outside_row_from_left # 点[x_g_dash, y_g_dash, z_g_dash]が属する通路の右端からの距離
        
        # x_g_dash < x_cであれば，x_g_dashはx_cよりも左側にある．このときのpath_lengthの計算にはx_outside_row_from_rightを使う．
        # これは通常のpath lengthの計算のとき (x_c, y_c, z_cとx_ps, y_ps, z_psとの比較時)とは逆であることに注意！！
        if x_g_dash <= x_c:
            x_o = x_outside_row_from_right
        else:
            x_o = x_outside_row_from_left

    if xyz_isin_row: # 畝内
        x_o = 0
        x_in_row_from_left =  x_g_dash - list_edge_negative_y[interval_xisin][0] # 点[x_g_dash, y_g_dash, z_g_dash]が属する畝の左端からの距離
        x_in_row_from_right = W_row - x_in_row_from_left # 点[x_g_dash, y_g_dash, z_g_dash]が属する畝の右端からの距離

        # x_g_dash < x_cであれば，x_g_dashはx_cよりも左側にある．このときのpath_lengthの計算にはx_in_row_from_leftを使う．
        # これは通常のpath lengthの計算のとき (x_c, y_c, z_cとx_ps, y_ps, z_psとの比較時)とは逆であることに注意！！
        if x_g_dash <= x_c:
            x = x_in_row_from_left
        else:
            x = x_in_row_from_right
    
    dx = abs(x_c - x_g_dash)
    x_residual = (dx + x - x_o) % (W_path + W_row)
    N_unit = (dx + x - x_o) // (W_path + W_row)
    Pr_dash = (N_unit * W_row - x + x_residual) / np.sin(alpha_c)

    Pr = Pr_dash / np.cos(beta_c)

    if round(r, 3) < round(Pr, 3):
        print()
        print("反射光の計算で何かがおかしい！")
        print("点({0:4.2f}, {1:4.2f}, {2:4.2f})から点({3:4.2f}, {4:4.2f}, {5:4.2f})までの反射光の計算．直線距離={6:4.2f}m, path length = {7:4.2f}m".format(x_g, y_g, z_g, x_c, y_c, z_c, r, Pr))
        print("x_o = {0:4.2f}, x_residual = {1:4.2f}, dx = {2:4.2f}".format(x_o, x_residual, dx))
        print()        

    return Pr

##################################################################
# absorbed radiation
# ある高さの葉群に降り注ぐ光について、すべての項目について計算する (直達光、直達光由来の散乱光、散乱光、地面からの反射光)。
# これらの光には、通常、k_bl * (1 - scatter_coef)がかけられる。
# k_blは葉の傾きを考慮したもの、(1-scatter_coef)は葉の吸収率を考慮したもの。
# ここで葉の吸収率（1 - scatter_coef）を考慮しているので、個葉光合成モデルでは、吸収率を考慮する必要はない。
# なお、さらに、葉群内のmultiple scatteringを考慮する場合、(1 - reflect_coef_canopy)を掛けることになる
#（散乱光、直達光由来の散乱光、地面からの反射光については）。ガチンコの直達光については、(1 - scatter_coef)のまま。
##################################################################

def cal_reflect_coef_canopy(beta, O_av, scatter_coef):
    '''
    葉群の反射係数を計算する。
    入力
        beta            --- 太陽高度．°ではなくradian.直達光はこの方向から差し込む．
        O_av            --- 
        scatter_coef    --- 葉の散乱係数
    '''
    reflect_coef_canopy = 2 * O_av /(O_av + np.sin(beta)) *((1 - np.sqrt(1 - scatter_coef)) / (1 + np.sqrt(1 - scatter_coef)))
    return reflect_coef_canopy

def cal_absorbed_radiation(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, 
                           list_edge_negative_y, list_edge_positive_y, 
                           LD, dV, dA, h_to_v_ratio, I0_beam_h, I0_dif_h, scatter_coef,
                           reflect_coef_ground, interp_I_g_h,  
                           N1, N2, a1, a2, b1, b2, max_iter, acceptable_absolute_error):
    '''
    ある点における光の吸収量を計算して返す(leaf areaあたり).
    この値に, その点におけるdL*f_sunをかけることで，ground areaあたりのsunlit葉の吸収光が求まる。
    また、dL * (1 - f_sun)をかけることで、ground areaあたりのshaded葉の吸収光が求まる。
    
    入力
        beta            --- 太陽高度．°ではなくradian.直達光はこの方向から差し込む．

    出力
        | I_abs_sun_per_LA      --- その微小体積に含まれるsunlit葉の、
                                    sunlit葉面積あたりの吸収光強度 (umol m-2 leaf s-1)

        | I_abs_sh_per_LA       --- その微小体積に含まれるshaded葉の
                                    shaded葉面積あたりの吸収光強度 (umol m-2 leaf s-1)
        
        | I_abs_per_LA          --- その微小体積に含まれる葉の
                                    葉面積あたりの"平均"吸収光強度 (umol m-2 leaf s-1)。
                                    sunlit葉とshaded葉の面積率で、I_abs_sun_per_LAとI_abs_sh_per_LAとを
                                    加重平均している。この値を光合成計算に使ってはいけない！
        
        | I_abs_sun_per_ground  --- その微小体積に含まれるsunlit葉の、
                                    土地面積あたりの葉の吸収光強度 (umol m-2 ground s-1)
        
        | I_abs_sh_per_ground   --- その微小体積に含まれるshaded葉の、
                                    土地面積あたりの葉の吸収光強度 (umol m-2 ground s-1)

        | I_abs_per_ground      --- その微小体積に含まれる葉 (sunlit葉 + shaded葉) の、
                                    土地面積あたりの葉の吸収光強度 (umol m-2 ground s-1)
    '''
    # 太陽高度が0以下のときは、計算しない。
    if beta <= 0:
        I_abs_sun_per_LA        = 0
        I_abs_sh_per_LA         = 0
        I_abs_per_LA            = 0
        I_abs_sun_per_ground    = 0
        I_abs_sh_per_ground     = 0
        I_abs_per_ground        = 0
        f_sun                   = 0
        return pd.Series([I_abs_sun_per_LA, I_abs_sh_per_LA, I_abs_per_LA, I_abs_sun_per_ground, I_abs_sh_per_ground, I_abs_per_ground, f_sun])
    
    O_av = GF.cal_ellipsoidal_G_from_beta_rad(h_to_v_ratio, beta)
    # f_sunおよびf_shを計算する．
    f_sun = cal_beam_fraction(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, LD, O_av)
    f_sh  = 1 - f_sun

    # 直達光の吸収量 (葉面積あたり)。sunlit葉のみが吸収する。
    I_beam_abs = cal_absorbed_beam_radiation(beta, O_av, I0_beam_h, scatter_coef)

    # 直達光由来の散乱光の吸収量（葉面積あたり）。sunlit葉、shaded葉どちらも吸収する。
    I_beam_scattered_abs = cal_absorbed_beam_scattered_radiation(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, LD, h_to_v_ratio, I0_beam_h, scatter_coef)

    # 天空からの散乱光の吸収量（葉面積あたり）。sunlit葉、shaded葉どちらも吸収する。
    I_dif_abs = cal_two_dimensional_integral_by_trapezoid(N1, N2, a1, b1, a2, b2, max_iter, acceptable_absolute_error, 
                                                          cal_absorbed_diffuse_radiation_by_trapezoid, 
                                                          x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, h_to_v_ratio, scatter_coef, LD)

    # 地面からの反射光の吸収量（葉面積あたり）。sunlit葉、shaded葉どちらも吸収する。
    I_ref_abs = cal_two_dimensional_integral_by_trapezoid(N1, N2, a1, b1, a2, b2, max_iter, acceptable_absolute_error, 
                                                          cal_absorbed_reflected_radiation_by_trapezoid, 
                                                          x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, h_to_v_ratio, scatter_coef, 
                                                          reflect_coef_ground, LD, interp_I_g_h)

    # sunlit葉およびshaded葉のleaf areaあたりの吸収光量 (umol m-2 leaf s-1)
    # これらの値に、f_sun * (dV * LD)をかければ、その微小体積の吸収光量 (umol s-1)になる。
    # 個葉光合成速度の計算には、これらの値を用いる。
    I_abs_sun_per_LA = I_beam_abs + I_beam_scattered_abs + I_dif_abs + I_ref_abs 
    I_abs_sh_per_LA  = I_beam_scattered_abs + I_dif_abs + I_ref_abs 

    # その微小体積に含まれる葉の葉面積あたりの"平均"吸収光強度 (umol m-2 leaf s-1)。
    # sunlit葉とshaded葉の面積率で、I_abs_sun_per_LAとI_abs_sh_per_LAとを加重平均している。
    # この値を光合成計算に使ってはいけない！
    I_abs_per_LA = (I_abs_sun_per_LA * f_sun) + (I_abs_sh_per_LA * f_sh)

    # その微小体積に含まれるsunlit葉およびshaded葉の吸収光量 (umol s-1)
    I_abs_sun_of_dV = I_abs_sun_per_LA * f_sun * (dV * LD)
    I_abs_sh_of_dV  = I_abs_sh_per_LA  * (1 - f_sun) * (dV * LD)


    # その微小体積に含まれるsunlit葉およびshaded葉の、ground areaあたりの吸収光量 (umol m-2_ground s-1)
    I_abs_sun_per_ground = I_abs_sun_of_dV / dA
    I_abs_sh_per_ground  = I_abs_sh_of_dV  / dA
    I_abs_per_ground = I_abs_sun_per_ground + I_abs_sh_per_ground

    return pd.Series([I_abs_sun_per_LA, I_abs_sh_per_LA, I_abs_per_LA, I_abs_sun_per_ground, I_abs_sh_per_ground, I_abs_per_ground, f_sun])


def cal_absorbed_beam_radiation(beta, O_av, I0_beam_h, scatter_coef):
    '''
    ある点における直達光の吸収量を計算して返す(leaf areaあたり).
    この値に, その点におけるdL*f_sunをかけることで，ground areaあたりの吸収光が求まる（ただしこのオペレーションはこの関数外で実施する）.

    入力
        I0_beam_h       --- 群落上部における，水平面1m2あたりのPPFD (umol m^-2 s^-1).
        beta            --- 太陽高度．°ではなくradian.
        path_length     --- 任意の点までのpath_length (m).
                            よって，通常のLAI（鉛直下向きに積算したもの）とは異なる．
        O_av            --- 光線と鉛直方向への，葉の投影比率．sphericalの場合は0.5
        scatter_coef    --- scattering coefficient
        LD              --- 畝における葉密度 (m^2 leaf m^-3)
    
    intermediate variables
        L_t             --- 光線がcanopyを貫通した距離を, LAIで表したもの．
                            光線方向に対する垂直面において積算したLAI(m^2 m^-2 beam cross-section)．
    '''
    k_bl = (O_av/np.sin(beta))
    I_beam_abs = k_bl * (1 - scatter_coef) * I0_beam_h
    return I_beam_abs

def cal_absorbed_beam_scattered_radiation(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, LD, h_to_v_ratio, I0_beam_h, scatter_coef):
    '''
    ある点における直達光由来の散乱光の吸収量を計算して返す(leaf areaあたり).
    この値に, その点におけるdL*f_sunをかけることで，ground areaあたりの吸収光が求まる（ただしこのオペレーションはこの関数外で実施する）.

    入力
        I0_beam_h       --- 群落上部における，水平面1m2あたりのPPFD (umol m^-2 s^-1).
        beta            --- 太陽高度．°ではなくradian.
        path_length     --- 任意の点までのpath_length (m).
                            よって，通常のLAI（鉛直下向きに積算したもの）とは異なる．
        h_to_v_ratio    --- ellipsoidal leaf angle distributionの横軸長/縦軸長
        scatter_coef    --- scattering coefficient
        LD              --- 畝における葉密度 (m^2 leaf m^-3)
    
    intermediate variables
        L_t             --- 光線がcanopyを貫通した距離を, LAIで表したもの．
                            光線方向に対する垂直面において積算したLAI(m^2 m^-2 beam cross-section)．
    '''
    
    O_av = GF.cal_ellipsoidal_G_from_beta_rad(h_to_v_ratio, beta)
    k_bl = (O_av/np.sin(beta))
    k    = k_bl * (1 - scatter_coef)**0.5
    Pr = cal_path_length(x_c, y_c, z_c, azm_sun_row_coord, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y)
    L_t = LD * Pr
    reflect_coef_canopy = cal_reflect_coef_canopy(beta, O_av, scatter_coef)      
    #I_beam_scattered_abs = (k_bl * (1 - scatter_coef)) * (I0_beam_h * (np.exp(-O_av * (1 - scatter_coef)**0.5 * L_t) - np.exp(-O_av * L_t)))
    # (直達光および 直達光由来の散乱光)の項から、純粋な直達光分を引き算する。
    I_beam_scattered_abs =  k * (1 - reflect_coef_canopy) * I0_beam_h * np.exp(-O_av * (1 - scatter_coef)**0.5 * L_t) - (k_bl * (1 - scatter_coef)) * I0_beam_h * np.exp(-O_av * L_t)
    return I_beam_scattered_abs
  
def cal_absorbed_diffuse_radiation_by_trapezoid(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, h_to_v_ratio, scatter_coef, LD, azm, beta):
    '''
    ある点における天空からの散乱光の吸収量を計算して返す(leaf areaあたり).
    この値に, その点におけるdL*f_sunをかけることで，ground areaあたりの吸収光が求まる（ただしこのオペレーションはこの関数外で実施する）.
    cal_two_dimensional_integral_by_trapezoidを使う想定．
    引き数の最後 (azm, beta)が，積分の変数． 
    '''
    # Solar_elev = 0であればf = 0とする。
    if beta <= 0:
        f = 0
        return f
    
    O_av = GF.cal_ellipsoidal_G_from_beta_rad(h_to_v_ratio, beta)
    k_bl = (O_av/np.sin(beta))
    k    = k_bl * (1 - scatter_coef)**0.5
    reflect_coef_canopy = cal_reflect_coef_canopy(beta, O_av, scatter_coef)
    Pr = cal_path_length(x_c,y_c,z_c, azm, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y)
    L_t = Pr * LD
    #f = (I0_dif_h/np.pi) * (k_bl * (1 - scatter_coef)) * np.exp(-O_av * (1 - scatter_coef)**0.5 * L_t) * np.sin(beta) * np.cos(beta)
    f = (I0_dif_h/np.pi) * k * (1 - reflect_coef_canopy) * np.exp(-O_av * (1 - scatter_coef)**0.5 * L_t) * np.sin(beta) * np.cos(beta)
    return f
    
def cal_absorbed_reflected_radiation_by_trapezoid(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, h_to_v_ratio, scatter_coef, reflect_coef_ground, LD, interp_I_g_h, alpha, theta):
    '''
    ある点における地面からの反射光の吸収量を計算して返す(leaf areaあたり).
    この値に, その点におけるdL*f_sunをかけることで，ground areaあたりの吸収光が求まる（ただしこのオペレーションはこの関数外で実施する）.
    以下の台形公式関数を使って，地表面(z_g = 0)におけるreflected radiationを計算する.
    cal_two_dimensional_integral_by_trapezoid(N1, N2, a1, b1, a2, b2, max_iter, acceptable_absolute_error, func, *input_for_func)

    まず，(alpha, theta)における(x_g, y_g)を計算して，その点から(x_c, y_c, z_c)への反射光を計算する．
    (x_g, y_g)に降り注ぐ光強度I_g_hは, grid interpolatorの線形補間により計算する. 

    入力
        | I_g_h                 --- (x_g, y_g, z_g)に，水平面に照射する下向きの光の強度．
        | interp_I_g_h          --- RegularGridInterpolatorによるinterpolating function. この関数に任意の(x_g, y_g, 0)を代入すれば，線形補間によってI_g_hが算出される．ただし，枠外はゼロ．
        | alpha                 --- (x_c, y_c, 0)を原点としたときの，(x_g, y_g, 0)のazimuth. 積分変数 (0 - 2*np.pi). (a2, b2, N2)に対応する．
        | theta                 --- (x_c, y_c, z_c)から(x_g, y_g, z_g)への，みかけ上の天頂角 (rad)．積分変数 (0 - np.pi/2). (a1, b1, N1)に対応する．
    '''
    # (x_g, y_g)をalpha, thetaから計算する．
    r_projected = z_c * np.tan(theta)
    x_g = x_c - r_projected * np.sin(alpha)
    y_g = y_c + r_projected * np.cos(alpha)
    I_g_h = interp_I_g_h(np.array([x_g, y_g, 0]))[0]
    
    # 地面に降り注ぐ光がゼロなら、反射光吸収も必然的にゼロ。
    if I_g_h <= 0:
        f = 0
        return f
    # O_avを計算する。ただし、反射光計算では、鉛直からのなす角theta (rad)でこれまで議論してきたので、beta (rad)に直す。
    beta = np.pi/2 - theta
    O_av = GF.cal_ellipsoidal_G_from_beta_rad(h_to_v_ratio, beta)

    reflect_coef_canopy = cal_reflect_coef_canopy(beta, O_av, scatter_coef)
    Pr = cal_path_length_reflected_radiation(x_c, y_c, z_c, W_row, W_path, list_edge_negative_y, list_edge_positive_y, x_g, y_g, 0)
    L_t = Pr * LD
    #f = O_av * (1 - scatter_coef) * reflect_coef_ground / np.pi * I_g_h * np.exp(-O_av * (1- scatter_coef)**0.5 * L_t) * np.sin(theta)
    f = O_av * (1 - scatter_coef)**0.5 * (1 - reflect_coef_canopy) * reflect_coef_ground / np.pi * I_g_h * np.exp(-O_av * (1- scatter_coef)**0.5 * L_t) * np.sin(theta)

    # if theta > np.pi/180 *85:
    #     print("\n(x_c,_y_c, z_c) = ({},{},{})".format(x_c, y_c, z_c))
    #     print("(x_g,_y_g, z_g) = ({},{},{})".format(x_g, y_g, 0))
    #     print("theta = {0:5.4f}, I_g_h = {1:5.4f}".format(theta/np.pi* 180, I_g_h))
    #     print("f = {0:3.2f}".format(f))

    distance = np.sqrt((x_c - x_g)**2 + (y_c - y_g)**2 + (z_c - 0)**2)
    if round(distance, 3) < round(Pr, 3):
        print()
        print("反射光の計算で何かがおかしい！")
        print("直線距離 = {0:4.2f}, (x_g, y_g, z_g) = ({1:4.2f}, {2:4.2f}, 0)から(x_c, y_c, z_c) = ({3:4.2f}, {4:4.2f}, {5:4.2f})への反射． \nI_g_h = {6:4.2f}, Pr = {7:4.2f}, theta = {8:4.2f}°, f = {9:4.2f}".format(distance, x_g, y_g, x_c, y_c, z_c, I_g_h, Pr, theta/np.pi * 180, f))

        print()     

    return f
##################################################################
# 個葉光合成モデル
# leaf_photo_Baldocchiというファイルに移動した。
##################################################################

###################################################
# 重積分（台形公式）
###################################################
def cal_two_dimensional_integral_by_trapezoid(N1, N2, a1, b1, a2, b2, max_iter, acceptable_absolute_error, func, *input_for_func):
    '''
    領域
    x: a2 ~ b2
    y: a1 ~ b1
    で囲まれたf(x, y)を台形公式で積分する．つまり，
    
    b1  b2
    ∫   ∫ f(x,y) dx*dy
    a1  a2

    を計算する．
    まずx (a2, b2, 分割数N2)で積分したのち, 次にy (a1, b1, 分割数N1)で積分する．  
    この関数の中で，*input_for_funcの引数の最後に, x, yを加えて, funcを呼び出す.
    よって, (x, y)は, funcの引数の最後に持ってくること. 

    入力
            | N1              --- xの分割数
            | N2              --- yの分割数
            | 


            
    '''
    def cal_iteration(integral_old, N1, N2, a1, b1, a2, b2, func, *input_for_func):
        '''
        繰り返し計算用のfunction. initialと式形が異なることや，再利用が必要なことから関数とする．
        N1 -> 2*N1
        N2 -> 2*N2
        として計算する．
        入力
            integral_old    --- 前回のiterationで計算した値
            N1              --- xの分割数（前回のiterationで使用した値）
            N2              --- yの分割数（前回のiterationで使用した値）
        '''
        fx_at_a1 = 0
        fx_at_b1 = 0
        for j in range(1, N2 + 1):
            fx_at_a1 += func(*input_for_func, a2+(b2-a2)/(2*N2)*(2*j-1), a1)
            fx_at_b1 += func(*input_for_func, a2+(b2-a2)/(2*N2)*(2*j-1), b1)
        
        fy_at_a2 = 0
        fy_at_b2 = 0
        for k in range(1, N1 + 1):
            fy_at_a2 += func(*input_for_func, a2, a1+(b1-a1)/(2*N1)*(2*k-1))
            fy_at_b2 += func(*input_for_func, b2, a1+(b1-a1)/(2*N1)*(2*k-1))
        
        f_inside = 0
        for k in range(1, N1 + 1):
            for j in range(1, N2 + 1):
                f_inside += func(*input_for_func, a2+(b2-a2)/(2*N2)*(2*j-1), a1+(b1-a1)/(2*N1)*(2*k-1))
        
        fy_rest = 0
        for k in range(1, N1 + 1):
            for j in range(1, N2):
                fy_rest += func(*input_for_func, a2+(b2-a2)/(2*N2)*(2*j), a1+(b1-a1)/(2*N1)*(2*k-1))
        
        fx_rest = 0
        for k in range(1, N1):
            for j in range(1, N2 + 1):
                fx_rest += func(*input_for_func, a2+(b2-a2)/(2*N2)*(2*j-1), a1+(b1-a1)/(2*N1)*(2*k))
        
        integral_new = integral_old/4 + (b1-a1)*(b2-a2)/(4*N1*N2)*((1/2.0)*(fx_at_a1 + fx_at_b1 + fy_at_a2 + fy_at_b2) + f_inside + fx_rest + fy_rest)
        return integral_new
    
    # 初期値を計算．
    fx_at_a1 = 0
    fx_at_b1 = 0
    for j in range(1, N2):
        fx_at_a1 += func(*input_for_func, a2 + (b2-a2)/N2 * j, a1)
        fx_at_b1 += func(*input_for_func, a2 + (b2-a2)/N2 * j, b1)
    
    fy_at_a2 = 0
    fy_at_b2 = 0
    for k in range(1, N1):
        fy_at_a2 += func(*input_for_func, a2, a1 + (b1-a1)/N1 * k)
        fy_at_b2 += func(*input_for_func, b2, a1 + (b1-a1)/N1 * k)
    
    f_inside = 0
    for k in range(1, N1):
        for j in range(1, N2):
            f_inside += func(*input_for_func, a2+(b2-a2)/N2*j, a1+(b1-a1)/N1*k)
    
    integral = (b1 - a1) * (b2 - a2) / (N1 * N2) *((func(*input_for_func, a2, a1) + func(*input_for_func, b2, a1) + func(*input_for_func, a2, b1) + func(*input_for_func, b2, b1)) / 4 + (1/2) * (fx_at_a1 + fx_at_b1 + fy_at_a2 + fy_at_b2) + f_inside)

    # 繰り返し計算．
    for i in range(max_iter):
        #print("{0}回目, Idif = {1:4.2f}".format(i, integral))
        integral_new = cal_iteration(integral, N1, N2, a1, b1, a2, b2, func, *input_for_func)
        absolute_error = abs(integral - integral_new)
        #relative_error = abs(absolute_error / integral)
        
        # 収束判定
        if (absolute_error < acceptable_absolute_error):
            #print("{0}回目のiterationで収束して，integral = {1:3.1f}が計算されました．N1 = {2:3.0f}, N2 = {3:3.0f}".format(i, integral_new, N1, N2))
            break
        # else:
        #     print("{0}回目のiterationで収束していません．integral_old = {1:3.1f}, integral = {2:3.1f}が計算されました．N1 = {3:3.0f}, N2 = {4:3.0f}".format(i, integral, integral_new, N1, N2))
          
        # 値の更新
        integral = integral_new
        N1 = 2 * N1
        N2 = 2 * N2

        if i == max_iter - 1:
            print("収束しませんでした．")
    return integral_new

###################################################
# Depreciated!!!!!!!!!
###################################################
def cal_azimuth_and_elevation_between_a_point_and_center(x_c, y_c, z_c, x_p, y_p, z_p):
    '''
    DEPRECIATED！！！というか使っていない。
    任意の点 (中心点)と, もう一つ別の点 (edgeを想定) との
    なす角を計算する．
    すなわち, 畝座標における
    azimuth (azimuth_c), 
    solar elevation (beta_c)
    を計算する (Gizjen & Goudriaan, 1989 Fig.1を参照).
    
    なお，ここで，"畝座標"とは，北を基準としたazimuthではなく，畝の長辺方向を基準としたazimuth．
    Gijzen and Goudriaan (1989)では，azimuthを北をゼロとして，時計回りに計算している．
    野村の太陽高度計算でも，北をゼロとして時計回りに回転させている．
    回転させた"結果"を，基準座標（畝座標）として用いている．
    
    入力
        x_c, y_c, z_c  --- 中心点
        x_p, y_p, z_p  --- edgeの点
    
    '''
    # x_cを基準として，edgeの座標を計算
    x = x_p - x_c
    y = y_p - y_c
    z = z_p - z_c
    
    # Gizjen and Goudriaan (1989)のFig.1の座標に変換
    if z == 0: # arctanが計算できないが，z ->0の極限を取ると，azimuth_cもbeta_cも90°になる．
        azimuth_c = np.pi/2
        beta_c = np.pi/2
    else:
        azimuth_c     = np.arctan(x/z)
        beta_c        = np.arctan(y/ (z/np.cos(azimuth_c)))
    
    return azimuth_c, beta_c


def cal_diffuse_radiation(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, O_av, scatter_coef, LD, ndiv_azm, ndiv_beta):
    '''
    DEPRECIATED!!!
    地点(x_c, y_c, z_c)におけるdiffuse radiationを計算する．
    天空をndiv_azm * ndiv_betaだけ分割して計算する．
    入力
        ndiv_azm  --- azimuth方向の分割数
        ndiv_beta --- beta方向の分割数
    '''

    delta = np.pi/180 # 90 deg や 0 degなどでは，三角関数の計算値が不安定になる可能性があるので，そこを除外するためにdeltaを足し引きする．
    azm_list = np.linspace(0 + delta, 2*np.pi -delta, ndiv_azm) 
    beta_list = np.linspace(0 + delta, np.pi/2 - delta, ndiv_beta)
    d_azm = (np.pi -delta - (-np.pi + delta)) / ndiv_azm # azmのステップ幅
    d_beta = (np.pi/2 - delta) / ndiv_beta # betaのステップ幅．

    # 微小
    d_A = np.sin(beta_list) * np.cos(beta_list) * d_beta * d_azm

    # 台形公式で積分する．
    F_list = []
    for azm in azm_list:
        f_list = []
        for beta in beta_list:
            Pr = cal_path_length(x_c,y_c,z_c, azm, beta, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y)
            L_t = Pr * LD
            f = np.exp(-O_av * (1 - scatter_coef)**0.5 * L_t) * np.sin(beta) *np.cos(beta)
            f_list.append(f)

        # Fを計算．
        f_list = np.array(f_list)
        F = d_beta * ((f_list[0] + f_list[-1]) / 2 + np.sum(f_list[1:-1]))
        F_list.append(F)
    
    # I_dif_hを計算．
    I_dif_h = (I0_dif_h/np.pi) * d_azm * ((F_list[0] + F_list[-1]) / 2 + np.sum(F_list[1:-1]))
    
    return I_dif_h

def cal_diffuse_radiation_with_convergence(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, O_av, scatter_coef, LD, ndiv_azm, ndiv_beta, acceptable_absolute_error, max_iter = 10):
    '''
    DEPRECIATED!!!
    Acceptable errorより小さくなる または max_iterに達するまで，diffuse radiationの計算刻みを小さくして計算していく．    
    '''
    ndiv_azm_before = ndiv_azm
    ndiv_beta_before = ndiv_beta
    I_dif_h_before = 10000
    error = 10000
    for i in range(max_iter):
        I_dif_h_after = cal_diffuse_radiation(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, O_av, scatter_coef, LD, ndiv_azm_before, ndiv_beta_before)
        print(I_dif_h_before, I_dif_h_after)
        absolute_error = abs(integral - integral_new)
        #relative_error = abs(absolute_error / I_dif_h_before)
        
        # 収束判定
        if (absolute_error < acceptable_absolute_error / 100):
            print("{0}回目のiterationで収束して，I_dif_h = {1:3.1f}が計算されました．ndiv_azm = {2:3.0f}, ndiv_".format(i, I_dif_h_before, ndiv_azm_before, ndiv_beta_before))
            break        

        # 値の更新
        I_dif_h_before = I_dif_h_after
        ndiv_azm_before  = 2 * ndiv_azm_before
        ndiv_beta_before = 2 * ndiv_beta_before

        if i == max_iter - 1:
            print("収束しませんでした．")
    return I_dif_h_before


def cal_reflected_radiation(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, O_av, scatter_coef, reflect_coef_ground, LD, df_I_g_h):
    '''
    DEPRECIATED!!!てか計算失敗(r → 0のときdAが適切に小さくできない問題; つまり、式が厳密でない)
    地面からの反射光を計算する．
    
    入力
        df_I_g_h            --- 地面上の点(x_g, y_g, z_g)へ到達した光強度I_g_hのdataframe. (x_g, y_g, z_g)が占める面積dAも列に含む．
        reflect_coef_ground --- 地面の反射係数
    '''
    
    def cal_reflected_radiation_from_a_point(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, O_av, scatter_coef, reflect_coef_ground, LD, x_g, y_g, z_g, dA, I_g_h):
        '''
        点(x_g, y_g, z_g)から点(x_c, y_c, z_c)への反射光強度を計算する．
        
        "これまでの理解をまとめる No.12, 13, 14"
        "地面からの反射光の計算 No.1, No.2 
        を参照のこと！

        まず，点(x_g, y_g, z_g)が，ハウスの畝-通路群のどこに存在するのかを確認する．
        奥行き方向(y_g)が畝の前面・背面よりもさらに外側に存在する場合，
        (x_g, y_g, z_g)を(x_g_dash, y_g_dash, z_g_dash)に移動させた点からpath lengthを計算する．
        
        '''
        x_edge_negative = list_edge_negative_y[0][0]
        x_edge_positive = list_edge_negative_y[-1][0]
        y_edge_negative = list_edge_negative_y[0][1]
        y_edge_positive = list_edge_positive_y[0][1]

        ################################
        # Prの計算
        # 点(x_g, y_g, z_g)から点(x_c, y_c, z_c)までの距離r および角度beta (x_c, y_c, z_cの"高度")
        r = np.sqrt ((x_c - x_g)**2 + (y_c - y_g)**2 + (z_c - z_g)**2)
        beta = np.arcsin((z_c - z_g) / r)        

        # 計算用の角度 (これまでの理解をまとめる No.13参照)
        if (z_c - z_g == 0):
            alpha_c = np.pi/2
        else:
            alpha_c = np.arctan(abs((x_c - x_g) / (z_c - z_g)))
        beta_c  = np.arctan(abs((y_c - y_g) / np.sqrt((x_c - x_g)**2 + (z_c - z_g)**2)))

        # 奥行き方向(y_g)が畝の前面・背面よりもさらに外側に存在する場合，
        # (x_g, y_g, z_g)を(x_g_dash, y_g_dash, z_g_dash)に移動させた点からpath lengthを計算する．
        # 点(x_g, y_g, z_g)が畝の前面・背面をはみ出してないかどうか
        if (y_edge_negative <= y_g) & (y_g <= y_edge_positive): # はみ出していない
            x_g_dash = x_g
            y_g_dash = y_g
            z_g_dash = z_g
        else: # はみ出している
            if y_g < y_edge_negative: # y_gが奥行き側にはみ出している
                y_g_dash = y_edge_negative
            elif y_edge_positive < y_g: # y_gが手前側にはみ出している
                y_g_dash = y_edge_positive
            vector_g_to_c = np.array([(x_c - x_g), (y_c - y_g), (z_c - z_g)])
            x_g_dash = (vector_g_to_c[0] / vector_g_to_c[1]) * (y_g_dash - y_g) + x_g
            z_g_dash = (vector_g_to_c[2] / vector_g_to_c[1]) * (y_g_dash - y_g) + z_g        
  
        # 点(x_g_dash, y_g_dash, z_g_dash)と点(x_c, y_c, z_c)との位置関係の確認．

        # 点(x_g_dash, y_g_dash, z_g_dash)が畝の中にあるか，外にあるか．
        # interval_xisinが偶数ならば，(x_g, y_g, z_g)は畝の中に存在する．
        xyz_isin_row, x_isin_row, y_isin_row, z_isin_row, interval_xisin = check_xyz_isin_row(x_g_dash, y_g_dash, z_g_dash, list_edge_negative_y, list_edge_positive_y)
        
        # x_c, y_c, z_cが畝内なのか，それとも通路なのか
        if not xyz_isin_row: # 通路内．x_outside_row_from_leftは"通路"の左端からの距離であることに注意!!
            x = 0
            if interval_xisin == -1: # x_g_dashは両端の畝よりもなお外側に存在する．
                x_outside_row_from_right  = x_edge_negative - x_g_dash # 左端の畝から点[x_g_dash, y_g_dash, z_g_dash]までの距離
                x_outside_row_from_left = x_g_dash - x_edge_positive # 右端の畝から点[x_g_dash, y_g_dash, z_g_dash]までの距離
            else: 
                x_outside_row_from_left  =  x_g_dash - list_edge_negative_y[interval_xisin][0] # 点[x_g_dash, y_g_dash, z_g_dash]が属する通路の左端からの距離
                x_outside_row_from_right = W_path - x_outside_row_from_left # 点[x_g_dash, y_g_dash, z_g_dash]が属する通路の右端からの距離
            
            # x_g_dash < x_cであれば，x_g_dashはx_cよりも左側にある．このときのpath_lengthの計算にはx_outside_row_from_rightを使う．
            # これは通常のpath lengthの計算のとき (x_c, y_c, z_cとx_ps, y_ps, z_psとの比較時)とは逆であることに注意！！
            if x_g_dash <= x_c:
                x_o = x_outside_row_from_right
            else:
                x_o = x_outside_row_from_left

        if xyz_isin_row: # 畝内
            x_o = 0
            x_in_row_from_left =  x_g_dash - list_edge_negative_y[interval_xisin][0] # 点[x_g_dash, y_g_dash, z_g_dash]が属する畝の左端からの距離
            x_in_row_from_right = W_row - x_in_row_from_left # 点[x_g_dash, y_g_dash, z_g_dash]が属する畝の右端からの距離

            # x_g_dash < x_cであれば，x_g_dashはx_cよりも左側にある．このときのpath_lengthの計算にはx_in_row_from_leftを使う．
            # これは通常のpath lengthの計算のとき (x_c, y_c, z_cとx_ps, y_ps, z_psとの比較時)とは逆であることに注意！！
            if x_g_dash <= x_c:
                x = x_in_row_from_left
            else:
                x = x_in_row_from_right
        
        dx = abs(x_c - x_g_dash)
        x_residual = (dx + x - x_o) % (W_path + W_row)
        N_unit = (dx + x - x_o) // (W_path + W_row)
        Pr_dash = (N_unit * W_row - x + x_residual) / np.sin(alpha_c)
 
        Pr = Pr_dash / np.cos(beta_c)
        L_t = Pr * LD
        if round(r, 3) < round(Pr, 3):
            print()
            print("反射光の計算で何かがおかしい！")
            print("点({0:4.2f}, {1:4.2f}, {2:4.2f})から点({3:4.2f}, {4:4.2f}, {5:4.2f})までの反射光の計算．直線距離={6:4.2f}m, path length = {7:4.2f}m".format(x_g, y_g, z_g, x_c, y_c, z_c, r, Pr))
            print("x_o = {0:4.2f}, x_residual = {1:4.2f}, dx = {2:4.2f}".format(x_o, x_residual, dx))
            print()        
        # (x_g, y_g, z_g)を中心とした微小面積dAからの，点(x_c, y_c, z_c)への反射光強度 (μmol m-2 s-1)を計算する．
        dI_ref_h = reflect_coef_ground/ (2* np.pi) * dA * I_g_h *np.sin(beta) / (r**2) * np.exp(- O_av * L_t)
        return dI_ref_h

        ################################
    
    df_I_g_h["dI_ref_h"]  = df_I_g_h.swifter.apply(lambda row:cal_reflected_radiation_from_a_point(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, O_av, scatter_coef, reflect_coef_ground, LD, row["x_g"], row["y_g"], row["z_g"], row["dA"], row["I_g_h"]),axis=1)
    I_ref_h = np.sum(df_I_g_h["dI_ref_h"])    
    print("(x_c, y_c, z_c)=({0:4.2f}, {1:4.2f}, {2:4.2f}) における反射光強度 = {3:4.2f} umol m-2 s-1".format(x_c, y_c, z_c, I_ref_h))
    return I_ref_h

# %%
def main(myTime, Solar_elev, azm_sun_row_coord, 
         d_y, t_d, ltt, lng, mrd,
         azm, alpha_c, beta_c,
         Ta, RH, Ca, gb, I0_beam_h, I0_dif_h, 
         H_row, LD,
         cfg):
    
    '''
    再利用のため、とりあえずパラメータはmain()の外に出したが、
    本当はmainの中に入れたい！
    '''
    print("計算開始: {}における計算".format(myTime))
    start = time.time()

    #######################
    # # ハウスに関するパラメータ。これらの値は、群落が成長しても変わらないものとする。
    # 特に、W_rowは、時系列とともに変えることができない。。。。
    W_ground = (cfg.W_row * cfg.n_row) + (cfg.W_path * (cfg.n_row - 1)) + cfg.W_margin * 2
    L_ground = cfg.L_row + cfg.L_margin * 2
    A_ground_in_a = W_ground * L_ground /100

    print("-------------------------------------------")
    print("ハウス情報")
    print("   畝本数 = {0:3.1f}, \n   畝幅 = {1:3.1f} m, \n   通路幅 = {2:3.1f} m, \n   畝長 = {3:3.1f} m, \n   外側の畝の両側の通路幅 = {4:3.1f} m, \n   畝の前後面の外側の通路幅 = {5:3.1f} m, \n".format(cfg.n_row, cfg.W_row, cfg.W_path, cfg.L_row, cfg.W_margin, cfg.L_margin))
    print("   ハウス幅 = {0:3.1f} m, \n   ハウス奥行 = {1:3.1f} m \n   ハウス面積 = {2:3.1f} a".format(W_ground, L_ground, A_ground_in_a))
    print("   azimuth_row = {0:3.1f}; 真北が，畝の真正面から半時計周りに{0:3.1f}°傾いている".format(cfg.azimuth_row))
    print("-------------------------------------------")

    #######################
    # 計算点に関するパラメータ
    Nx_per_row = int(cfg.W_row / cfg.delta_x_row) # rowの中で，光を計算する点を作る際の，x方向への分割数
    Ny_per_row = int(cfg.L_row / cfg.delta_y_row) # rowの中で，光を計算する点を作る際の，y方向への分割数
    Nz_per_row = int(H_row / cfg.delta_z_row) # rowの中で，光を計算する点を作る際の，z方向への分割数

    delta_x_row = cfg.W_row / Nx_per_row
    delta_y_row = cfg.L_row / Ny_per_row
    delta_z_row = H_row / Nz_per_row

    Nx_per_btm = int(W_ground / cfg.delta_x_btm) # 反射光計算のために，ハウスの底面に降り注ぐ光も計算する．そのときの，底の分割数．
    Ny_per_btm = int(L_ground / cfg.delta_y_btm)  # 反射光計算のために，通路に降り注ぐ光も計算する．そのときの，通路のy方向への分割数．

    delta_x_btm = W_ground / Nx_per_btm
    delta_y_btm = L_ground / Ny_per_btm

    #n_points = Nx_per_row * Ny_per_row * Nz_per_row + Nx_per_btm * Ny_per_btm 

    print("-------------------------------------------")
    print("数値計算に関する情報")
    print("   畝の分割数 \n   x方向:{0}, \n   y方向:{1}, \n   z方向:{2}".format(Nx_per_row, Ny_per_row, Nz_per_row))
    print("\n   畝の分割幅 \n   Δx = {0:5.3f} m, \n   Δy = {1:5.3f} m, \n   Δz = {2:5.3f} m".format(delta_x_row, delta_y_row, delta_z_row))
    print("\n   反射光計算のための地面の分割数 \n   x方向:{0}, \n   y方向:{1}".format(Nx_per_btm, Ny_per_btm))
    print("\n   反射光計算のための地面の分割幅 \n   Δx = {0:5.3f} m, \n   Δy = {1:5.3f} m".format(delta_x_btm, delta_y_btm))
    #print("\n 合計計算点数: {0} 点".format(n_points))

    #######################
    list_edge_negative_y, list_edge_positive_y =cal_row_edges(cfg.W_row, H_row, cfg.L_row, cfg.n_row, cfg.W_path, cfg.azimuth_row)
    print("-------------------------------------------")
    print("畝の端の座標:")
    print("x(間口)は{0:3.1f} m から {1:3.1f} mまでの範囲．".format(list_edge_negative_y[0][0], list_edge_negative_y[-1][0]))
    row_xedge_list = [(round(list_edge_negative_y[i][0], 2), round(list_edge_negative_y[i+1][0], 2)) for i in range(0, len(list_edge_negative_y), 2)]
    print("畝のx座標（左，右）")
    print(row_xedge_list)
    print()
    print("y(奥行き方向)は{0:3.1f} m から {1:3.1f} mまでの範囲．".format(list_edge_negative_y[0][1], list_edge_positive_y[0][1]))

    print("-------------------------------------------")


    #######################
    # 散乱光の数値計算に関するパラメータ
    #delta = np.pi/1800 # 90 deg や 0 degなどでは，三角関数の計算値が不安定になる可能性があるので，そこを除外するためにdeltaを足し引きする．
    # a1, b1, N1: diffuse radiationの計算における，betaの端点および分割数．
    #N1 = 2 # まずN1分割から計算する．分割数を2倍ずつ増やして，増やす前後の計算値を比較して，収束判定する．
    a1 = cfg.delta / 1.5
    b1 = np.pi / 2 - cfg.delta

    # a2, b2, N2: diffuse radiationの計算における，azmの端点および分割数
    #N2 = 4 # まずN2分割から計算する．分割数を2倍ずつ増やして，増やす前後の計算値を比較して，収束判定する．
    a2 = cfg.delta / 1.5
    b2 = 2 * np.pi - cfg.delta

    #max_iter = 10 # 散乱光の計算の最大反復数

    print("\n散乱光・反射光の収束判定")
    print("   最大繰り返し計算数:{0} 回, \n   許容誤差 = {1} umol m-2 s-1".format(cfg.max_iter, cfg.acceptable_absolute_error))
    print("-------------------------------------------")

    # 計算点 (df_radiation)
    x_row, y_row, z_row, x_btm, y_btm, z_btm, dV_row, dA_row, dA_btm = create_grid_points_uniform(list_edge_negative_y, list_edge_positive_y, Nx_per_row, Ny_per_row, Nz_per_row, Nx_per_btm, Ny_per_btm, cfg.W_margin, cfg.L_margin)
    df_radiation = pd.DataFrame(np.array([x_row, y_row, z_row, np.ones_like(x_row)*dV_row, np.ones_like(x_row)*dA_row]).reshape(5,-1).T, columns = ["x", "y", "z", "dV", "dA"])
    
    # 反射光の計算
    # まずは地表面へ降り注ぐ日射の計算．
    #x_btm, y_btm, z_btm = create_grid_points_bottom(list_edge_negative_y, list_edge_positive_y, Nx_per_btm, Ny_per_btm, W_margin, L_margin)
    df_I_g_h = pd.DataFrame(np.array(np.broadcast_arrays(x_btm, y_btm, z_btm)).reshape(3,-1).T, columns = ["x", "y", "z"])

    # test
    ###############################################
    # x_c = 0
    # y_c = 0
    # z_c = 1.5

    # Pr = cal_path_length(x_c,y_c,z_c, azm_sun_row_coord, Solar_elev, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y)
    # I_beam_h = cal_beam_radiation(x_c,y_c,z_c, azm_sun_row_coord, Solar_elev, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, LD, O_av, I0_beam_h, scatter_coef)

    # start = timer()
    # I_dif_h = cal_diffuse_radiation_with_convergence(x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, O_av, scatter_coef, LD, N2, N1, acceptable_absolute_error, max_iter = max_iter)
    # end = timer()
    # print("original = {0:4.2f}".format(end - start))

    # start = timer()
    # I_dif_h2 = cal_two_dimensional_integral_by_trapezoid(N1, N2, a1, b1, a2, b2, max_iter, acceptable_absolute_error, cal_diffuse_radiation_by_trapezoid, x_c, y_c, z_c, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, O_av, scatter_coef, LD)
    # end = timer()
    # print("improved = {0:4.2f}".format(end - start))
    df_I_g_h = df_I_g_h.copy()
    df_radiation = df_radiation.copy()
    
    # 出力後に環境データ等がないのは不便なので、df_radiationに書き込む。
    # myTime, Solar_elev, azm_sun_row_coord, Ta, RH, Ca, gb, I0_beam_h, I0_dif_h
    df_radiation["Time"]              = myTime
    df_radiation["Solar_elev"]        = Solar_elev
    df_radiation["azm_sun_row_coord"] = azm_sun_row_coord
    df_radiation["d_y"]               = d_y
    df_radiation["t_d"]               = t_d
    df_radiation["ltt"]               = ltt
    df_radiation["lng"]               = lng
    df_radiation["mrd"]               = mrd
    df_radiation["azm"]               = azm
    df_radiation["alpha_c"]           = alpha_c
    df_radiation["beta_c"]            = beta_c
    df_radiation["Ta"]                = Ta
    df_radiation["RH"]                = RH
    df_radiation["Ca"]                = Ca
    df_radiation["gb"]                = gb
    df_radiation["I0_beam_h"]         = I0_beam_h
    df_radiation["I0_dif_h"]          = I0_dif_h
    df_radiation["H_row"]             = H_row
    df_radiation["LD"]                = LD
    
    # mean_leaf_angleからellipsoidの横軸長/縦軸長を計算する。
    h_to_v_ratio = GF.cal_h_to_v_ratio_from_mean_leaf_angle(cfg.mean_leaf_angle)

    df_I_g_h["I_dif_h"]  = df_I_g_h.swifter.apply(lambda row:cal_two_dimensional_integral_by_trapezoid(cfg.N1, cfg.N2, a1, b1, a2, b2, cfg.max_iter, cfg.acceptable_absolute_error, cal_diffuse_radiation_by_trapezoid, row["x"],row["y"],row["z"],H_row, cfg.W_row, cfg.W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, h_to_v_ratio, cfg.scatter_coef, LD),axis=1)
    df_I_g_h["I_beam_h"] = df_I_g_h.swifter.apply(lambda row:cal_beam_radiation(row["x"],row["y"],row["z"], azm_sun_row_coord, Solar_elev, H_row, cfg.W_row, cfg.W_path, list_edge_negative_y, list_edge_positive_y, LD, h_to_v_ratio, I0_beam_h, cfg.scatter_coef),axis=1)
    df_I_g_h["I_g_h"]    = df_I_g_h["I_beam_h"] + df_I_g_h["I_dif_h"]
    df_I_g_h.rename(columns = {"x":"x_g", "y":"y_g", "z":"z_g"}, inplace = True)

    # interpolating functionをつくる．
    I_g_h = df_I_g_h["I_g_h"].values.reshape(x_btm.shape[0], y_btm.shape[1], z_btm.shape[2])
    interp_I_g_h = RegularGridInterpolator((x_btm.reshape(-1), y_btm.reshape(-1), z_btm.reshape(-1)), I_g_h, fill_value = 0, bounds_error= False)
    
    # 単純な光強度の計算
    ##############################
    # df_radiation["I_dif_h"]  = df_radiation.swifter.apply(lambda row:cal_two_dimensional_integral_by_trapezoid(N1, N2, a1, b1, a2, b2, max_iter, acceptable_absolute_error, cal_diffuse_radiation_by_trapezoid, row["x"],row["y"],row["z"],H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, I0_dif_h, h_to_v_ratio, scatter_coef, LD),axis=1)
    # df_radiation["I_beam_h"] = df_radiation.swifter.apply(lambda row:cal_beam_radiation(row["x"],row["y"],row["z"], azm_sun_row_coord, Solar_elev, H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, LD, h_to_v_ratio, I0_beam_h, scatter_coef),axis=1)
    # df_radiation["I_ref_h"] = df_radiation.swifter.apply(lambda row: cal_two_dimensional_integral_by_trapezoid(N1, N2, a1, b1, a2, b2, max_iter, acceptable_absolute_error, cal_reflected_radiation_by_trapezoid, row["x"],row["y"],row["z"], H_row, W_row, W_path, list_edge_negative_y, list_edge_positive_y, h_to_v_ratio, scatter_coef, reflect_coef_ground, LD, interp_I_g_h),axis=1)
    # df_radiation["I_h_sum"] = df_radiation["I_dif_h"] + df_radiation["I_beam_h"] + df_radiation["I_ref_h"]
    

    # 吸収光量の計算
    ##############################
    df_radiation[["I_abs_sun_per_LA", "I_abs_sh_per_LA", "I_abs_per_LA","I_abs_sun_per_ground", "I_abs_sh_per_ground", "I_abs_per_ground", "f_sun"]] = \
            df_radiation.swifter.apply(lambda row: cal_absorbed_radiation(
                row["x"], row["y"], row["z"],
                azm_sun_row_coord, Solar_elev,
                H_row, cfg.W_row, cfg.W_path, list_edge_negative_y, list_edge_positive_y,
                LD, row["dV"], row["dA"], h_to_v_ratio, I0_beam_h, I0_dif_h,
                cfg.scatter_coef, cfg.reflect_coef_ground, interp_I_g_h,
                cfg.N1, cfg.N2, a1, a2, b1, b2, cfg.max_iter, cfg.acceptable_absolute_error
                ), axis =1)
    # 注意：個葉光合成速度の計算にはI_abs_sun_per_LA, I_abs_sh_per_LAを使用する。
    # これらの個葉光合成速度に、f_sun * (dV * LD)および(1 - f_sun) * (dV * LD)を掛けて、足し合わせたものが、
    # その微小体積の群落光合成速度 (umol s-1)になる。これを、その微小体積の面積dAで割れば、
    # 土地面積あたりの群落光合成速度 (umol m-2 ground s-1)になる。
    # 個葉光合成速度の計算
    ##############################
    Vcmax, Jmax, Rd, Kc, Ko, Gamma_star = LP.cal_params_at_TL(Ta, cfg.R_gas,
                                                              cfg.Vcmax_25, cfg.C_Vcmax, cfg.DH_Vcmax,
                                                              cfg.Jmax_25, cfg.C_Jmax, cfg.DHa_Jmax, cfg.DHd_Jmax, cfg.DS_Jmax,
                                                              cfg.Rd_25, cfg.C_Rd, cfg.DH_Rd,
                                                              cfg.C_Kc, cfg.DH_Kc,
                                                              cfg.C_Ko, cfg.DH_Ko,
                                                              cfg.C_Gamma_star, cfg.DH_Gamma_star)

    #df_params_TL =pd.DataFrame(params_TL.tolist(), columns = ["Vcmax", "Jmax", "Rd", "Kc", "Ko", "Gamma_star"])
    #df_params_TL["TL"] = Ta
    #df_env = pd.DataFrame(np.array(np.meshgrid(Ca_list, Ta_list, RH_list, gb_list)).T.reshape(-1, 4), columns = ["Ca", "TL", "RH", "gb"])
    #df_env2 = df_env.merge(df_params_TL)

    # sunlit葉、shaded葉のJを計算する。
    vect_cal_J_from_Qabs = np.vectorize(LP.cal_J_from_Qabs, excluded = ["Jmax", "Phi_JQ", "Beta_JQ", "Theta_JQ"])
    df_radiation["J_sun"] = vect_cal_J_from_Qabs(Jmax, df_radiation["I_abs_sun_per_LA"], cfg.Phi_JQ, cfg.Beta_JQ, cfg.Theta_JQ)
    df_radiation["J_sh"] = vect_cal_J_from_Qabs(Jmax, df_radiation["I_abs_sh_per_LA"], cfg.Phi_JQ, cfg.Beta_JQ, cfg.Theta_JQ)
    
    # df_radiation["J_sun"] = df_radiation.swifter.apply(lambda row: LP.cal_J_from_Qabs(Jmax, row["I_abs_sun_per_LA"], cfg.Phi_JQ, cfg.Beta_JQ, cfg.Theta_JQ), axis = 1)
    # df_radiation["J_sh"] = df_radiation.swifter.apply(lambda row: LP.cal_J_from_Qabs(Jmax, row["I_abs_sh_per_LA"], cfg.Phi_JQ, cfg.Beta_JQ, cfg.Theta_JQ), axis = 1)

    vect_cal_leaf_photo = np.vectorize(LP.cal_leaf_photo, excluded = ["Vcmax", "Gamma_star", "Kc", "Ko", "Oxy", "Rd", "m", "b_dash", "RH", "gb", "Ca"])
    results_sun = vect_cal_leaf_photo(Vcmax, df_radiation["J_sun"], Gamma_star, Kc, Ko, cfg.Oxy, Rd, cfg.m, cfg.b_dash, RH, gb, Ca)
    df_results_sun = pd.DataFrame(results_sun, index = ["Ac_sun", "gs_c_sun", "Ci_c_sun", "Aj_sun", "gs_j_sun", "Ci_j_sun"]).T
    #results_sun = df_radiation.swifter.apply(lambda row: LP.cal_leaf_photo(Vcmax, row["J_sun"], Gamma_star, Kc, Ko, cfg.Oxy, Rd, cfg.m, cfg.b_dash, RH, gb, Ca), axis = 1)
    #df_results_sun = pd.DataFrame(results_sun.tolist(), columns = ["Ac_sun", "gs_c_sun", "Ci_c_sun", "Aj_sun", "gs_j_sun", "Ci_j_sun"])

    results_sh = vect_cal_leaf_photo(Vcmax, df_radiation["J_sh"], Gamma_star, Kc, Ko, cfg.Oxy, Rd, cfg.m, cfg.b_dash, RH, gb, Ca)
    df_results_sh = pd.DataFrame(results_sh, index = ["Ac_sh", "gs_c_sh", "Ci_c_sh", "Aj_sh", "gs_j_sh", "Ci_j_sh"]).T
    #results_sh = df_radiation.swifter.apply(lambda row: LP.cal_leaf_photo(Vcmax, row["J_sh"], Gamma_star, Kc, Ko, cfg.Oxy, Rd, cfg.m, cfg.b_dash, RH, gb, Ca), axis = 1)
    #df_results_sh = pd.DataFrame(results_sh.tolist(), columns = ["Ac_sh", "gs_c_sh", "Ci_c_sh", "Aj_sh", "gs_j_sh", "Ci_j_sh"])

    df_results = pd.concat([df_radiation, df_results_sun, df_results_sh], axis = 1)

    df_results["A_sun"] = df_results["Ac_sun"].where(df_results["Aj_sun"] > df_results["Ac_sun"], df_results["Aj_sun"])
    df_results["gs_sun"] = df_results["gs_c_sun"].where(df_results["Aj_sun"] > df_results["Ac_sun"], df_results["gs_j_sun"])
    df_results["Ci_sun"] = df_results["Ci_c_sun"].where(df_results["Aj_sun"] > df_results["Ac_sun"], df_results["Ci_j_sun"])
    df_results["Limited_sun"] = np.where(df_results["Ac_sun"] > df_results["Aj_sun"], "RuBP-limited", "Rubisco-limited")

    df_results["A_sh"] = df_results["Ac_sh"].where(df_results["Aj_sh"] > df_results["Ac_sh"], df_results["Aj_sh"])
    df_results["gs_sh"] = df_results["gs_c_sh"].where(df_results["Aj_sh"] > df_results["Ac_sh"], df_results["gs_j_sh"])
    df_results["Ci_sh"] = df_results["Ci_c_sh"].where(df_results["Aj_sh"] > df_results["Ac_sh"], df_results["Ci_j_sh"])
    df_results["Limited_sh"] = np.where(df_results["Ac_sh"] > df_results["Aj_sh"], "RuBP-limited", "Rubisco-limited")

    # 葉面積あたりの光合成速度 (f_sunを考慮している；1m2の葉のうち、一部はsunlit, 一部はshadedとして計算している)
    df_results["A_per_LA"] = df_results["A_sun"] * df_results["f_sun"] + df_results["A_sh"] * (1 - df_results["f_sun"])


    # 保存
    wdir_child = os.path.join(cfg.wdir, "output")
    wfile_name  = "" + myTime.strftime("%y%m%d_%H%M_out") + ".feather"
    #myTime2 = pd.to_datetime(str(myTime)) # np.vectorize(main)としたとき 
    #wfile_name  = "" + myTime2.strftime("%y%m%d_%H%M_out") + ".feather"
    wfile_path = os.path.join(wdir_child, wfile_name)
    feather.write_feather(df_results, wfile_path)
    end = time.time()
    print("計算終了: {0}の計算。かかった時間: {1:4.2f} s".format(myTime, end - start))
    return df_results

# %%
def preprocess_for_main(rfile):
    ######################################
    #############
    # ここからメインプログラム
    ###################################################
    # パラメータの読み込み
    with open(rfile, "r") as file:
        d_config = yaml.safe_load(file)
    #print(d_config)
    cfg = SimpleNamespace(**d_config)
    # for key, val in dict_config.items():
    #     exec(key + "=val")
        #print(key, val)
    
    ###################################################
    # 環境データの読み込み
    #rpath = r"/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/env_sample_one_row.csv"
    df_env = pd.read_csv(cfg.rpath_env, delimiter=',',comment='#',parse_dates=['Time'],index_col="Time")

    # 既に計算済みの場合、計算をスキップする。
    wdir_child = os.path.join(cfg.wdir, "output")
    if os.path.exists(wdir_child):
        # 古いcsvファイルがoutputフォルダに残っている場合は消去する。
        oldcsvpath = os.path.join(wdir_child, "dirnal.csv")
        if os.path.exists(oldcsvpath):
            os.remove(oldcsvpath)
        wfile_name_list = os.listdir(wdir_child)
        wfile_name_list = [i for i in wfile_name_list if 'feather' in i]
        wtime_list = pd.to_datetime(wfile_name_list, format = "%y%m%d_%H%M_out.feather")
        df_env = df_env.loc[~df_env.index.isin(wtime_list)]
        if df_env.empty:
            print("df_env is empty! This is probably because all the output feather files have alreadhy existed.")
            print("Delete the output directory to avoid this error." )
            sys.exit("Program stopped.")

    else: #outputフォルダがない場合は作成する。
        os.makedirs(wdir_child)

    ###################################################
    # 個体群の構造データの読み込み
    df_geo = pd.read_csv(cfg.rpath_geo, delimiter = ',', comment = '#', parse_dates= ["Time"], index_col = "Time")
    
    # df_envとくっつける。ただし、df_envの時刻優先。
    time_start = df_env.index[0]
    time_end = df_env.index[-1]

    df_env = pd.concat([df_env, df_geo], axis = 1)
    df_env = df_env.interpolate()
    df_env = df_env.loc[(df_env.index >= time_start) & (df_env.index <= time_end)]
    #print(df_env)

    ###################################################
    # 太陽の位置、畝の角度などの計算
    # Calculate solar elevation
    df_env["d_y"]=df_env.index.dayofyear.values
    # 30分おきだとステップが大きすぎて太陽高度の誤差がでるので，最後の項を追加．
    df_env["t_d"]=df_env.index.hour.values + df_env.index.minute.values/60 + df_env.index.second.values/(60*60)

    df_env.reset_index(inplace = True)
    df_env["ltt"]=cfg.latitude/180*np.pi
    df_env["lng"]=cfg.longitude/180*np.pi
    df_env["mrd"]=cfg.meridian/180*np.pi
    df_result=df_env[["ltt","lng","mrd","d_y","t_d"]].swifter.apply(lambda row:cal_solar_elevation(*row), axis=1)
    df_result.columns=["Solar_elev","azm","tsr","tss","dl","lst"] # azmは，北を0として，時計回りに（東へ）回転するときの角度を示す．matplotlibでは左手座標にするために，最後にax.invert_yaxis()する．
    df_env=pd.concat([df_env,df_result],axis=1)  

    # 畝座標における太陽の角度（alpha_c, beta_c）を計算。azmith_row_radは、北向きを基準にしたときに、畝が半時計回り（西方向）に何度傾いているかを示す。
    # azm_sun_row_coordは、畝座標系における太陽の方位角を示す。0°は畝の長辺方向。
    azimuth_row_rad = cfg.azimuth_row/180 * np.pi
    df_env["azm_sun_row_coord"] = df_env["azm"] + azimuth_row_rad
    df_env["azm_sun_row_coord"] = df_env["azm_sun_row_coord"].where(df_env["azm_sun_row_coord"]<=2*np.pi, df_env["azm_sun_row_coord"] - 2*np.pi)
    #print(df_env["azm_sun_row_coord"])
    df_env[["alpha_c", "beta_c"]] = df_env[["azm_sun_row_coord", "Solar_elev"]].swifter.apply(lambda x: cal_solar_position_relative_to_row(*x),axis=1)


    ###################################################
    # パラメータリスト
    ###################################################   

    # # #######################
    # # # 作物群落に関するパラメータ
    # LAI = H_row * cfg.W_row * cfg.L_row * cfg.n_row * LD / (W_ground * L_ground) 
    # print("-------------------------------------------")
    # print("作物のパラメータ")
    # print("   葉面積密度LD = {0:3.1f} m^2 m^-3, \n   LAI = {1:3.1f} m^2 m^-2  \n".format(LD, LAI))
    # print("-------------------------------------------")
    
    # #######################
    # # 個葉光合成に関するパラメータ
    print("-------------------------------------------")
    print("個葉光合成に関するパラメータ")
    print("   Vcmax_25 = {0:3.1f} umol m-2 s-1, \n   C_Vcmax = {1:3.0f}, \n   Jmax_25 = {2:3.1f}, \n   C_Jmax = {3:3.0f}, \n   DHa_Jmax = {4:3.0f}, \n   DHd_Jmax = {5:3.0f}, \n   DS_Jmax = {6:3.0f}".format(cfg.Vcmax_25, cfg.C_Vcmax, cfg.DH_Vcmax, cfg.Jmax_25, cfg.C_Jmax, cfg.DHa_Jmax, cfg.DHd_Jmax, cfg.DS_Jmax))
    print("   Rd_25 = {0:3.1f} umol m-2 s-1, \n   C_Rd = {1:3.0f}, \n   DH_Rd = {2:3.0f}, \n   m = {3:3.1f}, \n   b_dash = {4:4.3f}".format(cfg.Rd_25, cfg.C_Rd, cfg.DH_Rd, cfg.m, cfg.b_dash))
    print("-------------------------------------------")
    
    #######################
    # 見た目に関するパラメータ
    radius_sun_orbit = cfg.L_row / 2 *1.1 # 太陽軌道の見かけの半径
    
    # 太陽の位置を計算 (太陽軌道の半径をradius_sun_orbitとする)
    df_env = cal_solar_position(df_env, radius_sun_orbit, cfg.azimuth_row)

    #######################
    # シミュレーション用の光強度
    if cfg.radiation_mode == "Rs_out":
        # ハウス外の日射 (W m-2)を使用するとき；気象庁データを利用するとき
        df_env[["I0_beam_h_out", "I0_dif_h_out"]]   = df_env.swifter.apply(lambda row: cal_outside_diffuse_radiation(row["Rs"], row["d_y"], row["Solar_elev"], S_sc = 1370), axis = 1)
        df_env[["I0_beam_h", "I0_dif_h"]]           = df_env.swifter.apply(lambda row: cal_inside_radiation(row["I0_beam_h_out"], row["I0_dif_h_out"], cfg.transmission_coef_cover, cfg.transmission_coef_structure, cfg.beam_to_dif_conversion_ratio_cover), axis = 1)
        #print(df_env[["I0_beam_h", "I0_dif_h"]])
    elif cfg.radiation_mode == "PARi":
        # チャンバー等の実測PARを使用するとき。ただし、すでにbeamとdiffuseとの分離を終えているものとする。
        # df_env["I0_beam_h"] = df_env["PARi"] * 0.8
        # df_env["I0_dif_h"]  = df_env["PARi"] * 0.2
        print("PARiモード。(I0_beam_h, I0_dif_h) を基に計算を行う。")
    
    #######################
    # RHは0 - 100%なので、0 - 1.0に直す
    df_env["RH"] = df_env["RH"]/100

    # dfs_results = df_env.swifter.apply(lambda row: main(row["Time"], row["Solar_elev"], row["azm_sun_row_coord"],
    #                                                     row["d_y"], row["t_d"], row["ltt"], row["lng"], row["mrd"],
    #                                                     row["azm"], row["alpha_c"], row["beta_c"],
    #                                                     row["Ta"], row["RH"], row["Ca"], row["gb"], 
    #                                                     row["I0_beam_h"], row["I0_dif_h"], df_radiation, df_I_g_h,
    #                                                     a1, b1, a2, b2,
    #                                                     list_edge_negative_y, list_edge_positive_y,
    #                                                     x_row, y_row, z_row, x_btm, y_btm, z_btm, dV_row, dA_row, dA_btm,
    #                                                     cfg), axis = 1)
    
    # vect_main = np.vectorize(main, excluded=["cfg"])
    # dfs_results = vect_main(df_env["Time"], df_env["Solar_elev"], df_env["azm_sun_row_coord"],
    #                         df_env["d_y"], df_env["t_d"], df_env["ltt"], df_env["lng"], df_env["mrd"],
    #                         df_env["azm"], df_env["alpha_c"], df_env["beta_c"],
    #                         df_env["Ta"], df_env["RH"], df_env["Ca"], df_env["gb"],
    #                         df_env["I0_beam_h"], df_env["I0_dif_h"],
    #                         df_env["H_row"], df_env["LD"],
    #                         cfg = cfg)

    # dfs_results = df_env.swifter.force_parallel(enable=True).apply(lambda row: main(row["Time"], row["Solar_elev"], row["azm_sun_row_coord"],
    #                                                     row["d_y"], row["t_d"], row["ltt"], row["lng"], row["mrd"],
    #                                                     row["azm"], row["alpha_c"], row["beta_c"],
    #                                                     row["Ta"], row["RH"], row["Ca"], row["gb"], 
    #                                                     row["I0_beam_h"], row["I0_dif_h"], 
    #                                                     row["H_row"], row["LD"],                                                        
    #                                                     cfg), axis = 1)

    dfs_results = df_env[["Time", "Solar_elev", "azm_sun_row_coord",
                          "d_y", "t_d", "ltt", "lng", "mrd",
                          "azm", "alpha_c", "beta_c",
                          "Ta", "RH", "Ca", "gb",
                          "I0_beam_h", "I0_dif_h", 
                          "H_row", "LD"]].swifter.force_parallel(enable=True).apply(lambda row: main(*row, cfg =cfg), axis = 1)


# %%
if __name__ == '__main__':
    # yamlファイル。yamlファイル内のpathも変更すること！！！！
    rfile = "/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/v3構築用/parameter_list_v3.yml"
    #rfile  = "/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/大学内トマト個体群でのモデル検証/高知大学2021_教育用ハウス/parameter_list_v2.yml"
    start_all = time.time()
    print("#################### 計算開始 ####################")
    preprocess_for_main(rfile)
    end_all = time.time()
    print("#################### 計算終了。かかった時間: {0:4.2f} s ####################".format(end_all - start_all))

# %%
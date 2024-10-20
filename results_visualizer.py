# %%
import os
import numpy as np
import pandas as pd
import yaml
import pyarrow.feather as feather
from scipy.interpolate import Rbf # 外挿・内挿のための関数．
from types import SimpleNamespace # dictだと面倒なので

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

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
import canopy_photo_greenhouse_v2 as CP
import canopy_geometry as CG

pd.set_option('display.max_columns', None)

# %%

##################################################################
# 図示関係の関数
##################################################################
def get_directional_axes_for_visualization(r, azimuth_row):
    '''
    E-W, N-S軸を描画する際の、E, W, N, Sの地点の座標を得る。
    軸の長さはrとする。
    '''
    azimuth_row_rad = azimuth_row/180*np.pi
    west = np.array([r, 0])
    east = np.array([-r, 0])
    north = np.array([0, r])
    south = np.array([0, -r])

    #回転．
    azimuth_row_rad = azimuth_row/180*np.pi
    north_row_coord = np.array([north[0]*np.cos(azimuth_row_rad) - north[1]*np.sin(azimuth_row_rad), north[1]*np.cos(azimuth_row_rad) + north[0]*np.sin(azimuth_row_rad)])
    south_row_coord = np.array([south[0]*np.cos(azimuth_row_rad) - south[1]*np.sin(azimuth_row_rad), south[1]*np.cos(azimuth_row_rad) + south[0]*np.sin(azimuth_row_rad)])
    east_row_coord = np.array([east[0]*np.cos(azimuth_row_rad) - east[1]*np.sin(azimuth_row_rad), east[1]*np.cos(azimuth_row_rad) + east[0]*np.sin(azimuth_row_rad)])
    west_row_coord = np.array([west[0]*np.cos(azimuth_row_rad) - west[1]*np.sin(azimuth_row_rad), west[1]*np.cos(azimuth_row_rad) + west[0]*np.sin(azimuth_row_rad)])
    
    return north_row_coord, south_row_coord, east_row_coord, west_row_coord


def create_cross_section_x_z(df_radiation, y, list_edge_negative_y, to_be_visualized, vmin, vmax, wdir = False, show_fig = True):
    '''
    断面のヒートマップを作成する。

    入力
        df_radiation            --- dataframe
        y                       --- 断面の位置
        list_edge_negative_y    --- 畝の端点の座標のリスト。畝ごとに可視化するのに必要。
        to_be_visualized        --- 可視化する項目
        vmax                    --- カラーバーの最大値

    '''
    df_radiation_c = df_radiation.copy()    
    y_list = df_radiation["y"].drop_duplicates()
    #print(y_list)
    y = y_list.iloc[(y_list - y).abs().argsort()[0]]
    #print("y = {} m".format(y))

    df_dummy = df_radiation_c.loc[df_radiation_c["y"] == y]

    ix = np.shape(list_edge_negative_y)[0]
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 5))
    for i in np.arange(0, ix, step=2):
        df_dummy2 = df_dummy.loc[(list_edge_negative_y[i][0]<= df_dummy["x"]) & (df_dummy["x"] <= list_edge_negative_y[i+1][0])]
        #dummy_pv = df_dummy2.pivot(index = "x", columns = "z", values = "I_sum_h").T
        dummy_pv = df_dummy2.pivot(index = "x", columns = "z", values = to_be_visualized).T
        
        x, z = np.meshgrid(dummy_pv.index, dummy_pv.columns)
        vmax = vmax
        mappable = plt.contourf(z, x, dummy_pv.T, cmap = "jet", levels=600, norm=Normalize(vmin = vmin, vmax = vmax))
        
    plt.gca().set_aspect('equal')
    mappable.set_clim(vmin = vmin, vmax= vmax)
    sm = ScalarMappable(cmap="jet", norm=plt.Normalize(vmin, vmax))
    plt.colorbar(sm, orientation = "horizontal", ax = ax, shrink = 0.5)
    plt.title(df_radiation_c["Time"][0].strftime("%Y-%m-%d_%H:%M"), fontsize = 12)
    if wdir:
        wfile_name = df_radiation_c["Time"][0].strftime("%y%m%d_%H%M_xz_cross_sec") + "_" + to_be_visualized
        wfile_png  = os.path.join(wdir, wfile_name + ".png")
        plt.savefig(wfile_png)
    #plt.colorbar(mappable, ax = ax, orientation = "vertical")
    #plt.imshow(dummy_pv, cmap= "jet", interpolation = "gaussian", aspect = "equal", origin = "lower")
    if show_fig:
        plt.show()
    fig.clear()

def create_plan_heat_map_x_y_A_per_ground(df_results, list_edge_negative_y, vmin, vmax, to_be_visualized = "A_per_ground", chamber_pos_list = [], gh_pos = [], wdir = False, show_fig = True):
    '''
    平面のヒートマップを作成する。

    入力
        df_results              --- dataframe
        list_edge_negative_y    --- 畝の端点の座標のリスト。畝ごとに可視化するのに必要。
        to_be_visualized        --- 可視化する項目
        vmax                    --- カラーバーの最大値
        chamber_pos_list        --- chamberの位置のリスト。複数チャンバーに対応。[ [[x,x+dx], [y, y+dy]], ....]の形とする。
        gh_pos                  --- greenhouseの位置のリスト。[[x,x+dx], [y, y+dy]]の形とする。

    '''
    def draw_rectangle(pos_list):
        x_list = pos_list[0]
        y_list = pos_list[1]
        
        x0 = x_list[0]
        x1 = x_list[1]
        y0 = y_list[0]
        y1 = y_list[1]
        ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth = 1, edgecolor = "k", facecolor = "none")) #

    ix = np.shape(list_edge_negative_y)[0]
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(10, 10))
    for i in np.arange(0, ix, step=2):
        df_dummy2 = df_results.loc[(list_edge_negative_y[i][0]<= df_results["x"]) & (df_results["x"] <= list_edge_negative_y[i+1][0])].copy()
        df_dummy2["A_per_ground"] = df_dummy2["dV"] * df_dummy2["LD"] * df_dummy2["A_per_LA"] / df_dummy2["dA"]
        df_dummy3 = pd.pivot_table(df_dummy2, values = [to_be_visualized], index = ["Time", "x", "y"], columns = "z").sum(axis = 1).reset_index()
        dummy_pv = df_dummy3.pivot_table(index = "x", columns = "y", values = 0).T

        x, z = np.meshgrid(dummy_pv.index, dummy_pv.columns)
        vmax = vmax
        mappable = plt.contourf(z, x, dummy_pv.T, cmap = "jet", levels=600, norm=Normalize(vmin = vmin, vmax = vmax))
    plt.gca().set_aspect('equal')
    mappable.set_clim(vmin = vmin, vmax= vmax)
    sm = ScalarMappable(cmap="jet", norm=plt.Normalize(vmin, vmax))
    cbar = plt.colorbar(sm, orientation = "vertical", ax = ax, shrink = 0.5)
    cbar.ax.set_ylabel('$\mathit{A}$$\mathrm{_c}$ ($\mathrm{\mu mol \; m^{-2}_{ground} \; s^{-1}}$)', rotation=270, labelpad=25)
    plt.title(df_results["Time"][0].strftime("%Y-%m-%d_%H:%M"), fontsize = 12)
    
    # チャンバーやハウスの境界。
    if chamber_pos_list:
        for chamber_pos in chamber_pos_list:
            draw_rectangle(chamber_pos)
    if gh_pos:
        draw_rectangle(gh_pos)      
    
    ax.axis('off')
    ax.invert_yaxis()
    
    if wdir:
        wfile_name = df_results["Time"][0].strftime("%y%m%d_%H%M_xz_plan") + "_" + to_be_visualized
        wfile_png  = os.path.join(wdir, wfile_name + ".png")
        plt.savefig(wfile_png)
    if show_fig:
        plt.show()
    fig.clear()

def plot_3d_photo(df_results, to_be_visualized, list_edge_negative_y, list_edge_positive_y, H_row, vmin, vmax, I0_beam_h, I0_dif_h, A_per_ground, LAI, cfg, wdir = False, show_fig = True):
    #############################
    # 断面図
    df_input = df_results.copy()

    #############################
    # 三次元
    fig = px.scatter_3d(df_input, x='x', y='y',z='z', color = to_be_visualized, 
                        color_continuous_scale=px.colors.sequential.Jet, opacity=0.4,
                        range_color = [vmin, vmax])
    fig.update_traces(marker_size = 2)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
    )

    # ##########################
    # 太陽の方向からの矢印の描画
    # メッシュ作成
    ###########
    # 直達光
    x, y, z = np.meshgrid(np.linspace(list_edge_negative_y[0][0], list_edge_negative_y[-1][0], 4),
                        np.linspace(list_edge_negative_y[0][1], list_edge_positive_y[0][1], 6),
                        np.array([H_row + 2.]))
    x = x.reshape([-1])
    y = y.reshape([-1])
    z = z.reshape([-1])

    # 太陽の方向のベクトル
    u = - df_input["x_sun_row_coord"][0]
    v = - df_input["y_sun_row_coord"][0]
    w = - df_input["H_sun"][0]

    norm = np.sqrt(u**2 + v **2 + w ** 2)
    u = u/norm
    v = v/norm
    w = w/norm
    #print("(u,v,w) = ({0:3.2f}, {1:3.2f}, {2:3.2f})".format(u, v, w))
    #print("normalize後のnorm = {}".format(np.sqrt(u**2 + v **2 + w ** 2)))
    u = np.ones_like(x)*u
    v = np.ones_like(x)*v
    w = np.ones_like(x)*w
    if I0_beam_h > 0:
        sizeref = I0_beam_h/1000
    else:
        sizeref = 0.001
    for i, xx in enumerate(x):
        fig.add_trace(go.Cone(x = [x[i]], y = [y[i]], z = [z[i]], u = [u[i]], v = [v[i]], w = [w[i]], 
                            sizemode = "absolute", sizeref = sizeref, anchor= "tip", 
                            opacity = 0.8, colorscale = "reds", showscale = False))

    ###########
    # 散乱光
    np.random.seed(0)
    x, y, z = np.meshgrid(np.linspace(list_edge_negative_y[0][0]*0.9, list_edge_negative_y[-1][0]*0.9, 4),
                    np.linspace(list_edge_negative_y[0][1]*0.9, list_edge_positive_y[0][1]*0.9, 6),
                    np.array([H_row + 2.]))
    x = x.reshape([-1])
    y = y.reshape([-1])
    z = z.reshape([-1])

    # xyz_min = [list_edge_negative_y[0][0], list_edge_negative_y[0][1], H_row + 2]
    # xyz_max = [list_edge_negative_y[-1][0], list_edge_positive_y[0][1], H_row + 2.5]
    # xyz = np.random.uniform(low = xyz_min, high = xyz_max, size = (20, 3))

    # x = np.array([row[0] for row in xyz])
    # y = np.array([row[1] for row in xyz])
    # z = np.array([row[2] for row in xyz])

    # 散乱光の方向ベクトル
    if I0_dif_h > 0:
        sizeref = I0_dif_h/1000
    else:
        sizeref = 0.001
    
    for i, xx in enumerate(x):
        np.random.seed(i)
        u = np.random.uniform(-1, 1)
        v = np.random.uniform(-1, 1)
        w = np.random.uniform(-1, 0)
        norm = np.sqrt(u**2 + v **2 + w ** 2)
        u = u/norm
        v = v/norm
        w = w/norm
        #print("(u,v,w) = ({0:3.2f}, {1:3.2f}, {2:3.2f})".format(u, v, w))
        #print("normalize後のnorm = {}".format(np.sqrt(u**2 + v **2 + w ** 2)))
        u = np.ones_like(x)*u
        v = np.ones_like(x)*v
        w = np.ones_like(x)*w
        fig.add_trace(go.Cone(x = [x[i]], y = [y[i]], z = [z[i]], u = [u[i]], v = [v[i]], w = [w[i]], 
                            sizemode = "absolute", sizeref = sizeref, anchor= "tip", 
                            opacity = 0.8, colorscale = "blues", showscale = False))
        


    r = 1.2 * ((-1) *list_edge_negative_y[0][1])
    north_row_coord, south_row_coord, east_row_coord, west_row_coord = get_directional_axes_for_visualization(r, cfg.azimuth_row)

    fig.add_trace(go.Scatter3d(x = [north_row_coord[0], south_row_coord[0]], y = [north_row_coord[1], south_row_coord[1]], z = [0,0], mode = "lines",
                            line=dict(color='black',width=2), showlegend= False))
    fig.add_trace(go.Scatter3d(x = [east_row_coord[0], west_row_coord[0]], y = [east_row_coord[1], west_row_coord[1]], z = [0,0], mode = "lines",
                            line=dict(color='black',width=2), showlegend= False))

    fig.update_layout(scene = dict(annotations = [dict(x = north_row_coord[0], y = north_row_coord[1], z = 0, ax = 10, ay = -10, text = "N", showarrow = False)]))
    fig.update_scenes(aspectmode = "data")
    fig.update_layout(scene={
        'xaxis': {'autorange': 'reversed'}, # reverse automatically
        })
    fig.update_layout(
        title = r"{0}, <br>太陽高度 = {1:3.0f}°, 方位角 = {2:3.0f}°,<br>直達光 = {3:3.0f}, 散乱光 = {4:3.0f}, <br>LAI = {5:3.1f} m<sup>2</sup> m<sup>-2</sup>, <br>群落光合成速度 = {6:3.1f} μmol m<sup>-2</sup><sub>ground</sub> s<sup>-1</sup>".format(
            df_input["Time"][0].strftime("%Y-%m-%d %H時%M分"), 
            df_input["Solar_elev"][0] / np.pi * 180,
            df_input["azm"][0] / np.pi * 180,
            I0_beam_h,
            I0_dif_h,
            LAI,
            A_per_ground,
        )
    )

    # Default parameters which are used when `layout.scene.camera` is not provided
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.2, y=1.8, z=0.2)
    )
    fig.update_layout(scene_camera=camera)


    if wdir:
        wfile_name = df_input["Time"][0].strftime("%y%m%d_%H%M")+ "_" + to_be_visualized
        wfile_html = os.path.join(wdir, wfile_name + ".html")
        wfile_png  = os.path.join(wdir, wfile_name + ".png")
        fig.write_html(wfile_html)
        fig.write_image(wfile_png)
    if show_fig:
        fig.show()
    #$\mathrm{{\mu mol \; m^{{-2}}_{{ground}} \; s^{{-1}}}}$


def cal_canopy_photo_per_rows(df_results, list_edge_negative_y, cfg):
    '''
    畝ごとに光合成速度を計算する。ただし、両端の畝も、同一の土地面積としている。

    入力
        df_results            --- dataframe
        list_edge_negative_y    --- 畝の端点の座標のリスト。畝ごとに可視化するのに必要。
    '''
    df_dummy = df_results.copy()
    ix = np.shape(list_edge_negative_y)[0]
    list_A_per_ground_row_wise = []
    for i in np.arange(0, ix, step=2):
        df_dummy2 = df_dummy.loc[(list_edge_negative_y[i][0]<= df_dummy["x"]) & (df_dummy["x"] <= list_edge_negative_y[i+1][0])]
        A_per_ground_row_wise = (df_dummy2["dV"] * df_dummy2["LD"] * df_dummy2["A_per_LA"]).sum() / ((cfg.W_row + cfg.W_path) * (cfg.L_row + 2 * cfg.L_margin))
        list_A_per_ground_row_wise.append(A_per_ground_row_wise)
    return list_A_per_ground_row_wise

# def cal_canopy_photo_plan_view(df_results):
#     '''
#     点(x,y)の光合成速度を鉛直方向に積算する。

#     入力
#         df_results          --- dataframe
#     '''
#     df_dummy = df_results.copy()



def visualize_chamber_location(df_CG, cfg, x, dx, y, dy):
    '''
    長方形に囲まれた区画を可視化する。
    長方形は、(x,y)を始点として、(x+dx, y+dy)の範囲を切り取る。
    '''
    x_edge_n = x
    x_edge_p = x + dx
    y_edge_n = y
    y_edge_p = y + dy
    ax = CG.visualize_greenhouse(df_CG, cfg)
    CG.visualize_greenhouse_boundary(x_edge_p, x_edge_n, y_edge_p, y_edge_n, ax)
    ax.invert_yaxis()


def visualize_multiple_chamber_locations(df_CG, cfg, chamber_pos_list):
    '''
    長方形に囲まれた区画を可視化する。
    長方形は、(x,y)を始点として、(x+dx, y+dy)の範囲を切り取る。
    '''
    ax = CG.visualize_greenhouse(df_CG, cfg)
    for chamber_pos in chamber_pos_list:
        x_edge_n = chamber_pos[0][0]
        x_edge_p = chamber_pos[0][1]
        y_edge_n = chamber_pos[1][0]
        y_edge_p = chamber_pos[1][1]
        CG.visualize_greenhouse_boundary(x_edge_p, x_edge_n, y_edge_p, y_edge_n, ax)
    ax.invert_yaxis()

def cal_canopy_photo_in_area(df_results, x, dx, y, dy, delta_x_row, delta_y_row, list_edge_negative_y, list_edge_positive_y):
    '''
    長方形に囲まれた区画の個体群光合成速度を計算する。
    長方形は、(x,y)を始点として、(x+dx, y+dy)の範囲を切り取る。
        x 〜 x + dx
        y 〜 y + dy
    の面積あたりの光合成速度を算出する。

    ただし、長方形のボーダーが畝の中に存在する場合、ボーダー近傍の格子点 + delta_x_row / 2などを
    長方形の新ボーダーとする。
    '''
    x_edge_n = x
    x_edge_p = x + dx
    y_edge_n = y
    y_edge_p = y + dy

    df_dummy = df_results.copy()
    df_dummy2 = df_dummy.loc[(df_dummy["x"] >= x_edge_n) & 
                             (df_dummy["x"] <= x_edge_p) &
                             (df_dummy["y"] >= y_edge_n) &
                             (df_dummy["y"] <= y_edge_p)
                             ]
    # area内の個体群光合成速度 (μmol m-2)
    A = (df_dummy2["dV"] * df_dummy2["LD"] * df_dummy2["A_per_LA"]).sum()
    A = (df_dummy2["dV"] * df_dummy2["LD"]  * df_dummy2["A_per_LA"]).sum()
    I_abs = (df_dummy2["dV"] * df_dummy2["LD"] * df_dummy2["I_abs_per_LA"]).sum()



    # 土地あたり個体群光合成速度を計算するにあたって、"土地"が畝をはみ出すかどうか。
    # まずx軸方向。
    ix = len(list_edge_negative_y)
    x_edge_n_isin_row = False
    x_edge_p_isin_row = False
    for i in range(0, ix, 2):
        if (list_edge_negative_y[i][0] <= x_edge_n - delta_x_row / 2) & (x_edge_n - delta_x_row / 2 <= list_edge_negative_y[i+1][0]):
            x_edge_n_isin_row = True
        if (list_edge_negative_y[i][0] <= x_edge_p + delta_x_row / 2) & (x_edge_p + delta_x_row / 2 <= list_edge_negative_y[i+1][0]):
            x_edge_p_isin_row = True
    
    # y軸方向
    if (list_edge_negative_y[0][1] <= y_edge_n - delta_y_row / 2) & (y_edge_n - delta_y_row / 2 <= list_edge_positive_y[0][1]):
        y_edge_n_isin_row = True
    else:
        y_edge_n_isin_row = False
    if (list_edge_negative_y[0][1] <= y_edge_p + delta_y_row / 2) & (y_edge_p + delta_y_row / 2 <= list_edge_positive_y[0][1]):
        y_edge_p_isin_row = True
    else:
        y_edge_p_isin_row = False
    
    # 畝の外にareaのborderがあれば、問題なし。
    # 畝の中にareaのborderがある場合、area内の光合成速度を若干補正する必要がある。
    if x_edge_n_isin_row:
        x_edge_n_rev = df_dummy2["x"].min() - delta_x_row / 2
    else:
        x_edge_n_rev = x_edge_n

    if x_edge_p_isin_row:
        x_edge_p_rev = df_dummy2["x"].max() + delta_x_row / 2
    else:
        x_edge_p_rev = x_edge_p

    if y_edge_n_isin_row:
        y_edge_n_rev = df_dummy2["y"].min() - delta_y_row / 2
    else:
        y_edge_n_rev = y_edge_n
    
    if y_edge_p_isin_row:
        y_edge_p_rev = df_dummy2["y"].max() + delta_y_row / 2
    else:
        y_edge_p_rev = y_edge_p
    
    area = (x_edge_p_rev - x_edge_n_rev) * (y_edge_p_rev - y_edge_n_rev)
    A_per_ground_in_area        = A / area
    I_abs_per_ground_in_area    = I_abs / area

    # print("旧edgeは、x: {0:3.2f} 〜 {1:3.2f}　および　y: {2:3.2f} 〜 {3:3.2f}。".format(x_edge_n, x_edge_p, y_edge_n, y_edge_p))
    # print("新edgeは、x: {0:3.2f} 〜 {1:3.2f}　および　y: {2:3.2f} 〜 {3:3.2f}。".format(x_edge_n_rev, x_edge_p_rev, y_edge_n_rev, y_edge_p_rev))
    # print("A_per_ground = {0:4.2f} umol m-2 s-1".format(A_per_ground_in_area))
    return A_per_ground_in_area, I_abs_per_ground_in_area

def cal_and_visualize_canopy_photo_in_area(dfs_list, df_CG, cfg, x, dx, y, dy, delta_x_row, delta_y_row, list_edge_negative_y, list_edge_positive_y):
    '''
    ハウス内のある区画の個体群光合成速度（単位面積あたり）を計算するためのwrapper関数。
    詳しいことは、各関数を確認すること。
    
    dfs_list    --- 各時間の計算結果。
    df_CG       --- Canopy geometryのdataframe。

    '''
    visualize_chamber_location(df_CG, cfg, x, dx, y, dy)
    list_A_per_ground_in_area = []
    for dummy in dfs_list:
        A_per_ground_in_area,  I_abs_per_ground = cal_canopy_photo_in_area(dummy, x, dx, y, dy, delta_x_row, delta_y_row, list_edge_negative_y, list_edge_positive_y)
        list_dummy = [dummy["Time"][0], dummy["I0_beam_h"][0], dummy["I0_dif_h"][0], I_abs_per_ground, A_per_ground_in_area, dummy["x_sun_row_coord"][0], dummy["y_sun_row_coord"][0], dummy["H_sun"][0]]
        list_A_per_ground_in_area.append(list_dummy)

    columns_in_area  = ["Time", "I0_beam_h", "I0_dif_h", "I_abs_per_ground_in_area", "A_per_ground_in_area", "x_sun_row_coord", "y_sun_row_coord", "H_sun"]
    df_diurnal_in_area = pd.DataFrame(data = list_A_per_ground_in_area, columns = columns_in_area)
    df_diurnal_in_area["I0_h"] = df_diurnal_in_area["I0_beam_h"] + df_diurnal_in_area["I0_dif_h"]
    df_diurnal_in_area.set_index("Time", inplace= True)
    return df_diurnal_in_area


def read_feather(rdir, read_csv = True, output_fig = False, show_fig = True, to_be_visualized = "A_per_LA", vmax = 20, vmin = -5):
    '''
    '''
    ###################################################
    # パラメータの読み込み
    # ファイルに出力するかどうか
    output_csv = True

    rpath = os.path.join(rdir, "parameter_list_v2.yml")

    with open(rpath, "r") as file:
        d_config = yaml.safe_load(file)
    #print(d_config)
    cfg = SimpleNamespace(**d_config)

    ###################################################
    # 計算結果の読み込み
    rdir_results = os.path.join(rdir, "output")
    rfile_results_list = os.listdir(rdir_results)
    rfile_results_list = sorted(rfile_results_list)

    if read_csv:
        rpath_csv = os.path.join(rdir_results, "dirnal.csv")
        if os.path.exists(rpath_csv):
            df_diurnal = pd.read_csv(rpath_csv)
            df_diurnal["Time"] =pd.to_datetime(df_diurnal["Time"])
            df_diurnal.set_index("Time", inplace = True)
            return df_diurnal, np.nan, np.nan, np.nan

    ###################################################
    # 個体群構造の読み込み
    rpath_CG = os.path.join(rdir, "canopy_geometry.csv")
    df_CG = CG.process_csv(rpath_CG, cfg)


    ###################################################
    # 出力先
    if output_fig:
        wdir = os.path.join(rdir_results, "figs")
        if not os.path.exists(wdir):
            os.makedirs(wdir)
    else:
        wdir = False

    ###################################################
    # パラメータリスト
    ###################################################   
    #######################
    # # ハウスに関するパラメータ
    W_ground = (cfg.W_row * cfg.n_row) + (cfg.W_path * (cfg.n_row - 1)) + cfg.W_margin * 2
    L_ground = cfg.L_row + cfg.L_margin * 2
    A_ground_in_a = W_ground * L_ground /100

    # print("-------------------------------------------")
    # print("ハウス情報")
    # print("   畝本数 = {0:3.1f}, \n   畝幅 = {1:3.1f} m, \n   通路幅 = {2:3.1f} m, \n   畝長 = {3:3.1f} m, \n   外側の畝の両側の通路幅 = {4:3.1f} m, \n   畝の前後面の外側の通路幅 = {5:3.1f} m, \n".format(cfg.n_row, cfg.W_row, cfg.W_path, cfg.L_row, cfg.W_margin, cfg.L_margin))
    # print("   ハウス幅 = {0:3.1f} m, \n   ハウス奥行 = {1:3.1f} m \n   ハウス面積 = {2:3.1f} a".format(W_ground, L_ground, A_ground_in_a))
    # print("   azimuth_row = {0:3.1f}; 真北が，畝の真正面から半時計周りに{0:3.1f}°傾いている".format(cfg.azimuth_row))
    # print("-------------------------------------------")

    # # #######################
    # # # 個葉光合成に関するパラメータ
    # print("-------------------------------------------")
    # print("個葉光合成に関するパラメータ")
    # print("   Vcmax_25 = {0:3.1f} umol m-2 s-1, \n   C_Vcmax = {1:3.0f}, \n   Jmax_25 = {2:3.1f}, \n   C_Jmax = {3:3.0f}, \n   DHa_Jmax = {4:3.0f}, \n   DHd_Jmax = {5:3.0f}, \n   DS_Jmax = {6:3.0f}".format(cfg.Vcmax_25, cfg.C_Vcmax, cfg.DH_Vcmax, cfg.Jmax_25, cfg.C_Jmax, cfg.DHa_Jmax, cfg.DHd_Jmax, cfg.DS_Jmax))
    # print("   Rd_25 = {0:3.1f} umol m-2 s-1, \n   C_Rd = {1:3.0f}, \n   DH_Rd = {2:3.0f}, \n   m = {3:3.1f}, \n   b_dash = {4:4.3f}".format(cfg.Rd_25, cfg.C_Rd, cfg.DH_Rd, cfg.m, cfg.b_dash))
    # print("-------------------------------------------")

    #######################
    # 計算点に関するパラメータ
    Nx_per_row = int(cfg.W_row / cfg.delta_x_row) # rowの中で，光を計算する点を作る際の，x方向への分割数
    Ny_per_row = int(cfg.L_row / cfg.delta_y_row) # rowの中で，光を計算する点を作る際の，y方向への分割数

    delta_x_row = cfg.W_row / Nx_per_row
    delta_y_row = cfg.L_row / Ny_per_row

    Nx_per_btm = int(W_ground / 0.5) # 反射光計算のために，ハウスの底面に降り注ぐ光も計算する．そのときの，底の分割数．
    Ny_per_btm = int(L_ground / 0.5)  # 反射光計算のために，通路に降り注ぐ光も計算する．そのときの，通路のy方向への分割数．

    delta_x_btm = W_ground / Nx_per_btm
    delta_y_btm = L_ground / Ny_per_btm


    # print("-------------------------------------------")
    # print("数値計算に関する情報")
    # print("   畝の分割数 \n   x方向:{0}, \n   y方向:{1}".format(Nx_per_row, Ny_per_row))
    # print("\n   畝の分割幅 \n   Δx = {0:5.3f} m, \n   Δy = {1:5.3f} m".format(delta_x_row, delta_y_row))
    # print("\n   反射光計算のための地面の分割数 \n   x方向:{0}, \n   y方向:{1}".format(Nx_per_btm, Ny_per_btm))
    # print("\n   反射光計算のための地面の分割幅 \n   Δx = {0:5.3f} m, \n   Δy = {1:5.3f} m".format(delta_x_btm, delta_y_btm))

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

    # print("\n散乱光・反射光の収束判定")
    # print("   最大繰り返し計算数:{0} 回, \n   許容絶対誤差 = {1} umol m-2 s-1".format(cfg.max_iter, cfg.acceptable_absolute_error))
    # print("-------------------------------------------")

    # 畝の端の座標を計算
    #######################
    # 見た目に関するパラメータ
    radius_sun_orbit = cfg.L_row / 2 *1.1 # 太陽軌道の見かけの半径


    # 葉面積あたりの光合成速度 (f_sunを考慮している；1m2の葉のうち、一部はsunlit, 一部はshadedとして計算している)
    # to_be_visualized = "A_per_LA"
    #to_be_visualized = "I_abs_per_LA"
    dfs_list = []
    I_A_list = []
    for rfile_results in rfile_results_list[:]:
        if rfile_results.split(".")[-1] == "feather":
            rfile_path = os.path.join(rdir_results, rfile_results)
            df_results = feather.read_feather(rfile_path)
            df_results["f_sh"] = 1 - df_results["f_sun"]
            dfs_list.append(df_results)

            # LD, H_rowをdf_resultsから抽出する。
            LD      = df_results["LD"].iloc[0]
            H_row   = df_results["H_row"].iloc[0]

            # #######################
            # LD, H_row なしで計算できなかったパラメータを計算・表示する。
            # # 作物群落に関するパラメータ
            LAI = H_row * cfg.W_row * cfg.L_row * cfg.n_row * LD / (W_ground * L_ground) 
            Nz_per_row = int(H_row / cfg.delta_z_row) # rowの中で，光を計算する点を作る際の，z方向への分割数
            delta_z_row = H_row / Nz_per_row
            n_points = Nx_per_row * Ny_per_row * Nz_per_row + Nx_per_btm * Ny_per_btm 
            list_edge_negative_y, list_edge_positive_y = CP.cal_row_edges(cfg.W_row, H_row, cfg.L_row, cfg.n_row, cfg.W_path, cfg.azimuth_row)

            # print("-------------------------------------------")
            # print("H_row, LDに関するパラメータ")
            # print("   葉面積密度LD = {0:3.1f} m^2 m^-3, \n   LAI = {1:3.1f} m^2 m^-2  \n".format(LD, LAI))
            # print("   畝の分割数 \n   z方向:{0}".format(Nz_per_row))
            # print("   畝の分割幅 \n   Δz = {0:5.3f} m".format(delta_z_row))
            # print("   合計計算点数: {0} 点".format(n_points))
            # print("-------------------------------------------")
            # print("畝の端の座標:")
            # print("x(間口)は{0:3.1f} m から {1:3.1f} mまでの範囲．".format(list_edge_negative_y[0][0], list_edge_negative_y[-1][0]))
            # row_xedge_list = [(round(list_edge_negative_y[i][0], 2), round(list_edge_negative_y[i+1][0], 2)) for i in np.arange(0, len(list_edge_negative_y), step = 2)]
            # print("畝のx座標（左，右）")
            # print(row_xedge_list)
            # print()
            # print("y(奥行き方向)は{0:3.1f} m から {1:3.1f} mまでの範囲．".format(list_edge_negative_y[0][1], list_edge_positive_y[0][1]))
            # print("-------------------------------------------")

            # 太陽の位置を計算 (太陽軌道の半径をradius_sun_orbitとする)
            df_results = CP.cal_solar_position(df_results, radius_sun_orbit, cfg.azimuth_row)
            #######################

            y_list = df_results["y"].drop_duplicates()
            y = y_list.iloc[int(len(y_list)/2)]
            A_per_ground = (df_results["dV"] * LD * df_results["A_per_LA"]).sum() / (A_ground_in_a * 100)
            I_abs_per_ground = (df_results["dV"] * LD * df_results["I_abs_per_LA"]).sum() / (A_ground_in_a * 100)
            
            # 畝ごとの光合成速度
            list_A_per_ground_row_wise = cal_canopy_photo_per_rows(df_results, list_edge_negative_y, cfg)

            # 諸々をまとめたリスト
            list_dummy = [df_results["Time"][0], df_results["I0_beam_h"][0], df_results["I0_dif_h"][0], 
                          I_abs_per_ground, A_per_ground, 
                          df_results["x_sun_row_coord"][0], df_results["y_sun_row_coord"][0], df_results["H_sun"][0],
                          df_results["azm"][0], df_results["azm_sun_row_coord"][0], df_results["Solar_elev"][0],
                          df_results["Ta"][0], df_results["RH"][0], df_results["Ca"][0], df_results["gb"][0]]
            list_dummy.extend(list_A_per_ground_row_wise)
            I_A_list.append(list_dummy)
            

            #vmax = df_results[to_be_visualized].max()
            if output_fig:
                plot_3d_photo(df_results, to_be_visualized, list_edge_negative_y, list_edge_positive_y, H_row, vmin, vmax, df_results["I0_beam_h"][0], df_results["I0_dif_h"][0], A_per_ground, LAI, cfg, wdir, show_fig) #, wdir
                create_cross_section_x_z(df_results, y, list_edge_negative_y, to_be_visualized, vmin, vmax, wdir, show_fig) #, wdir
                create_plan_heat_map_x_y_A_per_ground(df_results, list_edge_negative_y, vmin, vmax, "A_per_ground", wdir, show_fig) #, wdir
                plt.close()
            # print("\nTime = {0}, \nA_per_ground = {1:4.2f} umol m-2 ground s-1".format(df_results["Time"][0], A_per_ground))
            # print("LAI = {0:3.2f}".format(LAI))
            # print("Solar_elev= {0:3.2f}".format(df_results["Solar_elev"][0] / np.pi * 180))
    
    # 日変化をまとめる
    columns             = ["Time", "I0_beam_h", "I0_dif_h", 
                           "I_abs_per_ground", "A_per_ground", 
                           "x_sun_row_coord", "y_sun_row_coord", "H_sun",
                           "azm", "azm_sun_row_coord", "Solar_elev",
                           "Ta", "RH", "Ca", "gb"]
    columns_row_wise    = [f'A_per_ground_row_wise_{i}' for i in range(1, len(list_A_per_ground_row_wise)+1)]
    columns.extend(columns_row_wise)
    df_diurnal = pd.DataFrame(data = I_A_list, columns = columns)
    df_diurnal["I0_h"] = df_diurnal["I0_beam_h"] + df_diurnal["I0_dif_h"]
    A_day =df_diurnal["A_per_ground"].sum() * 3600 /10**6
    A_day_row_wise = df_diurnal[columns_row_wise].sum() * 3600 /10**6
    print("###########################")
    print("日あたりの個体群光合成速度 = {0:3.2f} mol m-2 ground s-1".format(A_day))
    print("###########################")
    print("畝ごとの個体群光合成速度 (mol m-2 ground s-1)")
    print(A_day_row_wise)
    print("###########################")
    xfmt = mdates.DateFormatter('%H時')#%m/%d 
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4, 6))
    ############
    ax1=plt.subplot(2,1,1)
    ax1.set_title("群落上部のPPFD",fontsize=15)
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_ylabel('$\mathrm{PPFD}$  \n ($\mathrm{\mu mol \; m^{-2}_{ground} \; s^{-1}}$)')
    plt.plot(df_diurnal["Time"], df_diurnal["I0_h"])        

    plt.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0,edgecolor='none',
                frameon=False,labelspacing=0.1,handlelength=.5,handleheight=.1,columnspacing=0,fontsize=12) #,prop={"family":"Times New Roman"}

    ############
    ax2=plt.subplot(2,1,2)
    ax2.set_title("群落光合成速度",fontsize=15)
    ax2.xaxis.set_major_formatter(xfmt)
    ax2.set_ylabel('群落光合成速度  \n ($\mathrm{\mu mol \; m^{-2}_{ground} \; s^{-1}}$)')
    plt.plot(df_diurnal["Time"], df_diurnal["A_per_ground"])        

    plt.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0,edgecolor='none',
                frameon=False,labelspacing=0.1,handlelength=.5,handleheight=.1,columnspacing=0,fontsize=12) #,prop={"family":"Times New Roman"}

    ax1.grid('on', which='major', axis='x',linestyle="--",alpha=0.6 )
    ax2.grid('on', which='major', axis='x',linestyle="--",alpha=0.6 )       

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    df_diurnal.set_index("Time", inplace= True)
    if output_csv:
        df_diurnal.to_csv(os.path.join(rdir_results, "dirnal.csv"))
    return df_diurnal, dfs_list, list_edge_negative_y, list_edge_positive_y



# %%
if __name__ == "__main__":

    ###################################################
    # パラメータの読み込み
    rdir = "/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/論文執筆_ハイワイヤー/1.8m_400ppm_clear/"
    # ファイルに出力するかどうか
    output = True

    rpath = os.path.join(rdir, "parameter_list_v2.yml")

    with open(rpath, "r") as file:
        d_config = yaml.safe_load(file)
    #print(d_config)
    cfg = SimpleNamespace(**d_config)

    ###################################################
    # 計算結果の読み込み
    rdir_results = os.path.join(rdir, "output")
    rfile_results_list = os.listdir(rdir_results)
    rfile_results_list = sorted(rfile_results_list)

    ###################################################
    # 個体群構造の読み込み
    rpath_CG = os.path.join(rdir, "canopy_geometry.csv")
    df_CG = CG.process_csv(rpath_CG, cfg)

    ###################################################


    # 出力先
    if output:
        wdir = os.path.join(rdir_results, "figs")
        if not os.path.exists(wdir):
            os.makedirs(wdir)
    else:
        wdir = False

    ###################################################
    # パラメータリスト
    ###################################################   
    #######################
    # # ハウスに関するパラメータ
    W_ground = (cfg.W_row * cfg.n_row) + (cfg.W_path * (cfg.n_row - 1)) + cfg.W_margin * 2
    L_ground = cfg.L_row + cfg.L_margin * 2
    A_ground_in_a = W_ground * L_ground /100

    print("-------------------------------------------")
    print("ハウス情報")
    print("   畝本数 = {0:3.1f}, \n   畝幅 = {1:3.1f} m, \n   通路幅 = {2:3.1f} m, \n   畝長 = {3:3.1f} m, \n   外側の畝の両側の通路幅 = {4:3.1f} m, \n   畝の前後面の外側の通路幅 = {5:3.1f} m, \n".format(cfg.n_row, cfg.W_row, cfg.W_path, cfg.L_row, cfg.W_margin, cfg.L_margin))
    print("   ハウス幅 = {0:3.1f} m, \n   ハウス奥行 = {1:3.1f} m \n   ハウス面積 = {2:3.1f} a".format(W_ground, L_ground, A_ground_in_a))
    print("   azimuth_row = {0:3.1f}; 真北が，畝の真正面から半時計周りに{0:3.1f}°傾いている".format(cfg.azimuth_row))
    print("-------------------------------------------")

    # #######################
    # # 個葉光合成に関するパラメータ
    print("-------------------------------------------")
    print("個葉光合成に関するパラメータ")
    print("   Vcmax_25 = {0:3.1f} umol m-2 s-1, \n   C_Vcmax = {1:3.0f}, \n   Jmax_25 = {2:3.1f}, \n   C_Jmax = {3:3.0f}, \n   DHa_Jmax = {4:3.0f}, \n   DHd_Jmax = {5:3.0f}, \n   DS_Jmax = {6:3.0f}".format(cfg.Vcmax_25, cfg.C_Vcmax, cfg.DH_Vcmax, cfg.Jmax_25, cfg.C_Jmax, cfg.DHa_Jmax, cfg.DHd_Jmax, cfg.DS_Jmax))
    print("   Rd_25 = {0:3.1f} umol m-2 s-1, \n   C_Rd = {1:3.0f}, \n   DH_Rd = {2:3.0f}, \n   m = {3:3.1f}, \n   b_dash = {4:4.3f}".format(cfg.Rd_25, cfg.C_Rd, cfg.DH_Rd, cfg.m, cfg.b_dash))
    print("-------------------------------------------")

    #######################
    # 計算点に関するパラメータ
    Nx_per_row = int(cfg.W_row / cfg.delta_x_row) # rowの中で，光を計算する点を作る際の，x方向への分割数
    Ny_per_row = int(cfg.L_row / cfg.delta_y_row) # rowの中で，光を計算する点を作る際の，y方向への分割数

    delta_x_row = cfg.W_row / Nx_per_row
    delta_y_row = cfg.L_row / Ny_per_row

    Nx_per_btm = int(W_ground / 0.5) # 反射光計算のために，ハウスの底面に降り注ぐ光も計算する．そのときの，底の分割数．
    Ny_per_btm = int(L_ground / 0.5)  # 反射光計算のために，通路に降り注ぐ光も計算する．そのときの，通路のy方向への分割数．

    delta_x_btm = W_ground / Nx_per_btm
    delta_y_btm = L_ground / Ny_per_btm


    print("-------------------------------------------")
    print("数値計算に関する情報")
    print("   畝の分割数 \n   x方向:{0}, \n   y方向:{1}".format(Nx_per_row, Ny_per_row))
    print("\n   畝の分割幅 \n   Δx = {0:5.3f} m, \n   Δy = {1:5.3f} m".format(delta_x_row, delta_y_row))
    print("\n   反射光計算のための地面の分割数 \n   x方向:{0}, \n   y方向:{1}".format(Nx_per_btm, Ny_per_btm))
    print("\n   反射光計算のための地面の分割幅 \n   Δx = {0:5.3f} m, \n   Δy = {1:5.3f} m".format(delta_x_btm, delta_y_btm))

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
    print("   最大繰り返し計算数:{0} 回, \n   許容絶対誤差 = {1} umol m-2 s-1".format(cfg.max_iter, cfg.acceptable_absolute_error))
    print("-------------------------------------------")

    # 畝の端の座標を計算
    #######################
    # 見た目に関するパラメータ
    radius_sun_orbit = cfg.L_row / 2 *1.1 # 太陽軌道の見かけの半径


    # %%
    # 葉面積あたりの光合成速度 (f_sunを考慮している；1m2の葉のうち、一部はsunlit, 一部はshadedとして計算している)
    to_be_visualized = "A_per_LA"
    #to_be_visualized = "I_abs_per_LA"
    vmax = 15 #df_results[to_be_visualized].max()
    vmin = -5
    dfs_list = []
    I_A_list = []
    for rfile_results in rfile_results_list[:24]:
        if rfile_results.split(".")[-1] == "feather":
            rfile_path = os.path.join(rdir_results, rfile_results)
            df_results = feather.read_feather(rfile_path)
            df_results["f_sh"] = 1 - df_results["f_sun"]
            dfs_list.append(df_results)

            # LD, H_rowをdf_resultsから抽出する。
            LD      = df_results["LD"].iloc[0]
            H_row   = df_results["H_row"].iloc[0]

            # #######################
            # LD, H_row なしで計算できなかったパラメータを計算・表示する。
            # # 作物群落に関するパラメータ
            LAI = H_row * cfg.W_row * cfg.L_row * cfg.n_row * LD / (W_ground * L_ground) 
            Nz_per_row = int(H_row / cfg.delta_z_row) # rowの中で，光を計算する点を作る際の，z方向への分割数
            delta_z_row = H_row / Nz_per_row
            n_points = Nx_per_row * Ny_per_row * Nz_per_row + Nx_per_btm * Ny_per_btm 
            list_edge_negative_y, list_edge_positive_y = CP.cal_row_edges(cfg.W_row, H_row, cfg.L_row, cfg.n_row, cfg.W_path, cfg.azimuth_row)

            print("-------------------------------------------")
            print("H_row, LDに関するパラメータ")
            print("   葉面積密度LD = {0:3.1f} m^2 m^-3, \n   LAI = {1:3.1f} m^2 m^-2  \n".format(LD, LAI))
            print("   畝の分割数 \n   z方向:{0}".format(Nz_per_row))
            print("   畝の分割幅 \n   Δz = {0:5.3f} m".format(delta_z_row))
            print("   合計計算点数: {0} 点".format(n_points))
            print("-------------------------------------------")
            print("畝の端の座標:")
            print("x(間口)は{0:3.1f} m から {1:3.1f} mまでの範囲．".format(list_edge_negative_y[0][0], list_edge_negative_y[-1][0]))
            row_xedge_list = [(round(list_edge_negative_y[i][0], 2), round(list_edge_negative_y[i+1][0], 2)) for i in np.arange(0, len(list_edge_negative_y), step = 2)]
            print("畝のx座標（左，右）")
            print(row_xedge_list)
            print()
            print("y(奥行き方向)は{0:3.1f} m から {1:3.1f} mまでの範囲．".format(list_edge_negative_y[0][1], list_edge_positive_y[0][1]))
            print("-------------------------------------------")

            # 太陽の位置を計算 (太陽軌道の半径をradius_sun_orbitとする)
            df_results = CP.cal_solar_position(df_results, radius_sun_orbit, cfg.azimuth_row)
            #######################

            y_list = df_results["y"].drop_duplicates()
            y = y_list.iloc[int(len(y_list)/2)]
            A_per_ground = (df_results["dV"] * LD * df_results["A_per_LA"]).sum() / (A_ground_in_a * 100)
            I_abs_per_ground = (df_results["dV"] * LD * df_results["I_abs_per_LA"]).sum() / (A_ground_in_a * 100)
            
            # 畝ごとの光合成速度
            list_A_per_ground_row_wise = cal_canopy_photo_per_rows(df_results, list_edge_negative_y, cfg)

            # 諸々をまとめたリスト
            list_dummy = [df_results["Time"][0], df_results["I0_beam_h"][0], df_results["I0_dif_h"][0], I_abs_per_ground, A_per_ground, df_results["x_sun_row_coord"][0], df_results["y_sun_row_coord"][0], df_results["H_sun"][0]]
            list_dummy.extend(list_A_per_ground_row_wise)
            I_A_list.append(list_dummy)
            

            #vmax = df_results[to_be_visualized].max()
            plot_3d_photo(df_results, to_be_visualized, list_edge_negative_y, list_edge_positive_y, H_row, vmin, vmax, df_results["I0_beam_h"][0], df_results["I0_dif_h"][0], A_per_ground, LAI, cfg, wdir) #, wdir
            create_cross_section_x_z(df_results, y, list_edge_negative_y, to_be_visualized, vmin, vmax, wdir) #, wdir

            print("\nTime = {0}, \nA_per_ground = {1:4.2f} umol m-2 ground s-1".format(df_results["Time"][0], A_per_ground))
            print("LAI = {0:3.2f}".format(LAI))
            print("Solar_elev= {0:3.2f}".format(df_results["Solar_elev"][0] / np.pi * 180))

    # 日変化をまとめる
    columns             = ["Time", "I0_beam_h", "I0_dif_h", "I_abs_per_ground", "A_per_ground", "x_sun_row_coord", "y_sun_row_coord", "H_sun"]
    columns_row_wise    = [f'A_per_ground_row_wise_{i}' for i in range(1, len(list_A_per_ground_row_wise)+1)]
    columns.extend(columns_row_wise)
    df_diurnal = pd.DataFrame(data = I_A_list, columns = columns)
    df_diurnal["I0_h"] = df_diurnal["I0_beam_h"] + df_diurnal["I0_dif_h"]
    A_day =df_diurnal["A_per_ground"].sum() * 3600 /10**6
    A_day_row_wise = df_diurnal[columns_row_wise].sum() * 3600 /10**6
    print("###########################")
    print("日あたりの個体群光合成速度 = {0:3.2f} mol m-2 ground s-1".format(A_day))
    print("###########################")
    print("畝ごとの個体群光合成速度 (mol m-2 ground s-1)")
    print(A_day_row_wise)
    print("###########################")
    xfmt = mdates.DateFormatter('%H時')#%m/%d 
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4, 6))
    ############
    ax1=plt.subplot(2,1,1)
    ax1.set_title("群落上部のPPFD",fontsize=15)
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_ylabel('$\mathrm{PPFD}$  \n ($\mathrm{\mu mol \; m^{-2}_{ground} \; s^{-1}}$)')
    plt.plot(df_diurnal["Time"], df_diurnal["I0_h"])        

    plt.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0,edgecolor='none',
                frameon=False,labelspacing=0.1,handlelength=.5,handleheight=.1,columnspacing=0,fontsize=12) #,prop={"family":"Times New Roman"}

    ############
    ax2=plt.subplot(2,1,2)
    ax2.set_title("群落光合成速度",fontsize=15)
    ax2.xaxis.set_major_formatter(xfmt)
    ax2.set_ylabel('群落光合成速度  \n ($\mathrm{\mu mol \; m^{-2}_{ground} \; s^{-1}}$)')
    plt.plot(df_diurnal["Time"], df_diurnal["A_per_ground"])        

    plt.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0,edgecolor='none',
                frameon=False,labelspacing=0.1,handlelength=.5,handleheight=.1,columnspacing=0,fontsize=12) #,prop={"family":"Times New Roman"}

    ax1.grid('on', which='major', axis='x',linestyle="--",alpha=0.6 )
    ax2.grid('on', which='major', axis='x',linestyle="--",alpha=0.6 )       

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    df_results.head()
    # %%
    # x   = 0.0
    # dx  = 1.8
    # y   = -18 - 0.6
    # dy  = 1.2

    x   = 0.0
    dx  = 1.8
    y   = -25
    dy  = 50

    df_diurnal_in_area = cal_and_visualize_canopy_photo_in_area(dfs_list, df_CG, cfg, x, dx, y, dy, delta_x_row, delta_y_row, list_edge_negative_y, list_edge_positive_y)
    A_day_in_area =df_diurnal_in_area["A_per_ground_in_area"].sum() * 3600 /10**6
    # %%
    df_results.head()
# %%

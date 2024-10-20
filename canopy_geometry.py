#%%
#%matplotlib notebook
# for creating a responsive plot
#%matplotlib widget

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
#import canopy_photo_greenhouse as CP

pd.set_option('display.max_columns', None)
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.family'] = 'IPAexGothic'


# %%
def cal_LAI(W_row, H_row, LD, L_row, n_row, W_path, W_margin, L_margin):
    '''
    畝幅、畝高、葉面積密度などが与えられている場合に、LAIを計算する。
    '''
    W_ground = (W_row * n_row) + (W_path * (n_row - 1)) + W_margin * 2
    L_ground = L_row + L_margin * 2
    A_ground_in_a = W_ground * L_ground /100
    LAI = H_row * W_row * L_row * n_row * LD / (W_ground * L_ground) 
    return LAI

def cal_LD(W_row, H_row, LAI, L_row, n_row, W_path, W_margin, L_margin):
    '''
    LAI、畝幅、畝高が与えられている場合に、葉面積密度(LD)を計算する。
    '''
    W_ground = (W_row * n_row) + (W_path * (n_row - 1)) + W_margin * 2
    L_ground = L_row + L_margin * 2
    A_ground_in_a = W_ground * L_ground /100
    LD = LAI * (W_ground * L_ground) / (H_row * W_row * L_row * n_row)
    print("W_ground = {}, L_ground = {}, H_row = {}, W_row = {}, L_row = {}, n_row = {}".format(W_ground, L_ground, H_row, W_row, L_row, n_row))

    return LD

def visualize_rows(W_row, H_row, L_row, n_row, W_path, azimuth_row, ax, alpha = 0.01, ps = 0.1, point_interval = 0.1):
    '''
    うねをつくる．畝は，南北方向へ延びる畝をdefaultとしてazimuth_rowだけ，時計回りに回転させている．
    ただし，本プログラムでは，"畝座標"を基準にすることから，見た目や座標はそのまま(x-z平面が畝の正面)．
    原点はほ場の中心．
    
    また，すでに，3D平面が作られていることを前提にする．
    つまり，メインプログラムにおいて，
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    が走っているものとする．

        入力
            W_row  -- 畝幅 (m)
            H_row  -- 畝高 (m)
            L_row  -- 畝長 (m)
            n_row  -- 畝の本数 (m)
            W_path -- 通路幅 (m)
            azimth_row_r -- 畝の向き（°; 北向きから東向きに回転した）
            ps     -- 出力時の点群の大きさ
            alpha  -- 出力時の点群のopacity
            point_interval  -- 点と点との距離 (m)
        出力
    '''

    azimuth_row_rad = (azimuth_row / 180 ) * np.pi

    # うねの座標（端から端まで）
    list_row_coord = []
    for i in np.arange(n_row):
        if n_row % 2 == 0:
            x1_row = (-n_row/2 + i) * (W_row + W_path) + (1.0 / 2.0) * W_path
            x2_row = x1_row + W_row
        elif n_row % 2 != 0:
            x1_row = (-(n_row -1) / 2 + i) * (W_row + W_path) - (1.0 / 2.0) * W_row
            x2_row = x1_row + W_row
        y1_row = - L_row / 2
        y2_row = L_row /2
        list_row_coord.append([x1_row, x2_row, y1_row, y2_row])
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    for z in np.arange(0, H_row, point_interval):
        for row_coord in list_row_coord:
            x_row = np.arange(row_coord[0], row_coord[1], point_interval)
            y_row = np.arange(row_coord[2], row_coord[3], point_interval)

            # mesh gridを作成する (列がx_row, 行がy_rowで構成される行列)
            x_row_m, y_row_m = np.meshgrid(x_row, y_row)

            # # 畝の方向が，南北方向からazimuth_rowだけ回転しているので，それを補正する．
            # x_row_coord = x_row_m *np.cos(azimuth_row_rad) - y_row_m * np.sin(azimuth_row_rad)
            # y_row_coord = y_row_m *np.cos(azimuth_row_rad) + x_row_m * np.sin(azimuth_row_rad)

            # x_row (あるいはy_row)と同じ形の行列をつくる
            z_row = np.full_like(x_row_m, z)

            ax.scatter(x_row_m, y_row_m, z_row, color = "g", s = ps, alpha =alpha)
            

    #ax.set_aspect('equal', adjustable='box')

    # 左手座標系で表示する; y軸の向きをひっくり返す
    # ax.invert_yaxis()


def draw_north_south_axis(r, azimuth_row, ax):
    '''
    南北・東西の軸を描く．また，North arrowを描く．ただし，すでに，3D平面が作られていることを前提にする．
    つまり，メインプログラムにおいて，
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    が走っているものとする．
    また，以下のリンクから，3D arrowのclass群をコピペしているものとする．
    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    
        r -- 太陽公転軌道の半球の半径
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

    dx = (north_row_coord[0] - south_row_coord[0]) /10
    dy = (north_row_coord[1] - south_row_coord[1]) /10

    ax.plot3D([north_row_coord[0], south_row_coord[0]], [north_row_coord[1], south_row_coord[1]], [0,0], color = "k", lw = 0.7)
    ax.plot3D([west_row_coord[0], east_row_coord[0]], [west_row_coord[1], east_row_coord[1]], [0,0], color = "k", lw = 0.7)

    ax.arrow3D(north_row_coord[0]-dx,north_row_coord[1]-dy,0,
            dx,dy,0,
            mutation_scale=20,
            arrowstyle="->",
            #linestyle='dashed',
            color = "k",
            lw = 0.7
            )
    ax.annotate3D('N', (north_row_coord[0],north_row_coord[1], 0), xytext=(5, 5), textcoords='offset points')

def draw_axes(r1, r2, angle, color, ls, str1, str2, offset1, offset2, ax):
    '''
    南北・東西の軸を描く．また，North arrowを描く．ただし，すでに，3D平面が作られていることを前提にする．
    つまり，メインプログラムにおいて，
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    が走っているものとする．
    また，以下のリンクから，3D arrowのclass群をコピペしているものとする．
    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    
        r -- 太陽公転軌道の半球の半径
    '''
    angle_rad = angle/180*np.pi
    west = np.array([r2, 0])
    east = np.array([-r2, 0])
    north = np.array([0, r1])
    south = np.array([0, -r1])

    #回転．
    angle_rad = angle/180*np.pi
    north_row_coord = np.array([north[0]*np.cos(angle_rad) - north[1]*np.sin(angle_rad), north[1]*np.cos(angle_rad) + north[0]*np.sin(angle_rad)])
    south_row_coord = np.array([south[0]*np.cos(angle_rad) - south[1]*np.sin(angle_rad), south[1]*np.cos(angle_rad) + south[0]*np.sin(angle_rad)])
    east_row_coord = np.array([east[0]*np.cos(angle_rad) - east[1]*np.sin(angle_rad), east[1]*np.cos(angle_rad) + east[0]*np.sin(angle_rad)])
    west_row_coord = np.array([west[0]*np.cos(angle_rad) - west[1]*np.sin(angle_rad), west[1]*np.cos(angle_rad) + west[0]*np.sin(angle_rad)])

    dx1 = (north_row_coord[0] - south_row_coord[0]) /10
    dy1 = (north_row_coord[1] - south_row_coord[1]) /10

    dx2 = (west_row_coord[0] - east_row_coord[0]) /10
    dy2 = (west_row_coord[1] - east_row_coord[1]) /10

    ax.plot3D([north_row_coord[0], south_row_coord[0]], [north_row_coord[1], south_row_coord[1]], [0,0], color = color, lw = 0.7, ls = ls)
    ax.plot3D([west_row_coord[0], east_row_coord[0]], [west_row_coord[1], east_row_coord[1]], [0,0], color = color, lw = 0.7, ls = ls)

    ax.arrow3D(north_row_coord[0]-dx1,north_row_coord[1]-dy1,0,
            dx1,dy1,0,
            mutation_scale=20,
            arrowstyle="->",
            linestyle=ls,
            color = color,
            lw = 0.7
            )
    ax.annotate3D(str1, (north_row_coord[0],north_row_coord[1], 0), xytext=offset1, textcoords='offset points', color = color)

    ax.arrow3D(west_row_coord[0]-dx2,west_row_coord[1]-dy2,0,
            dx2,dy2,0,
            mutation_scale=20,
            arrowstyle="->",
            linestyle=ls,
            color = color,
            lw = 0.7
            )
    ax.annotate3D(str2, (west_row_coord[0],west_row_coord[1], 0), xytext=offset2, textcoords='offset points', color = color)

def draw_north_arrow(r, azimuth_row, ax):
    '''
    North arrowを描く．ただし，すでに，3D平面が作られていることを前提にする．
    つまり，メインプログラムにおいて，
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    が走っているものとする．
    また，以下のリンクから，3D arrowのclass群をコピペしているものとする．
    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    
        r -- arrowの長さ
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

    dx = (north_row_coord[0] - south_row_coord[0]) 
    dy = (north_row_coord[1] - south_row_coord[1]) 

    #ax.plot3D([north_row_coord[0], south_row_coord[0]], [north_row_coord[1], south_row_coord[1]], [0,0], color = "k", lw = 0.7)
    #ax.plot3D([west_row_coord[0], east_row_coord[0]], [west_row_coord[1], east_row_coord[1]], [0,0], color = "k", lw = 0.7)

    ax.arrow3D(north_row_coord[0]-dx,north_row_coord[1]-dy,0,
            dx,dy,0,
            mutation_scale=20,
            arrowstyle="wedge",
            #linestyle='dashed',
            color = "k",
            lw = 0.7
            )
    ax.annotate3D('N', (north_row_coord[0],north_row_coord[1], 0), xytext=(5, 5), textcoords='offset points')


##################################################################
# 図示関係の関数
##################################################################
class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

def draw_a_sphere(r, xc, yc, zc, ax):
    '''
    球面を描く．ただし，すでに，3D平面が作られていることを前提にする．
    つまり，メインプログラムにおいて，
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    が走っているものとする．
    '''
    c = 0 # center
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi/2:50j]
    x = r*np.cos(u)*np.sin(v) + xc
    y = r*np.sin(u)*np.sin(v) + yc
    z = r*np.cos(v) + zc
    ax.plot_surface(x-c, y-c, z-c, color= "aqua", alpha=0.05)
    ax.set_aspect('equal', adjustable='box')

def cal_greenhouse_edge(W_row, H_row, L_row, n_row, W_path, W_margin, L_margin):
    W_ground = (W_row * n_row) + (W_path * (n_row - 1)) + W_margin * 2
    L_ground = L_row + L_margin * 2
    x_edge_p = W_ground / 2
    x_edge_n = - x_edge_p
    y_edge_p = L_ground / 2
    y_edge_n = - y_edge_p
    return pd.Series([x_edge_p, x_edge_n, y_edge_p, y_edge_n])

def visualize_greenhouse_boundary(x_edge_p, x_edge_n, y_edge_p, y_edge_n, ax):
    x_in_between = np.linspace(x_edge_n, x_edge_p)
    y_in_between = np.linspace(y_edge_n, y_edge_p)
    xx, yy = np.meshgrid(x_in_between, y_in_between)
    z = np.full_like(xx, 0)
    ax.plot_surface(xx, yy, z, color = "lightgray", alpha = 0.1)

    ax.plot([x_edge_p, x_edge_n], [y_edge_p, y_edge_p], [0,0], color = "k", lw = 0.5)
    ax.plot([x_edge_p, x_edge_n], [y_edge_n, y_edge_n], [0,0], color = "k", lw = 0.5)
    ax.plot([x_edge_p, x_edge_p], [y_edge_p, y_edge_n], [0,0], color = "k", lw = 0.5)
    ax.plot([x_edge_n, x_edge_n], [y_edge_p, y_edge_n], [0,0], color = "k", lw = 0.5)

def adjust_ticks(W_row, H_row, L_row, n_row, W_path, W_margin, L_margin, ax):
    edges = cal_greenhouse_edge(W_row, H_row, L_row, n_row, W_path, W_margin, L_margin)
    ax.set_ylim(edges[3], edges[2])
    ax.set_xlim(edges[1], edges[0])
    ax.set_zlim(0, H_row)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_zticks([])
    ax.set_yticks(np.arange(edges[3],edges[2], step = 6))
    ax.set_xticks(np.arange(edges[1], edges[0], step = 6))
    ax.set_zticks(np.arange(0, H_row))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])


    # Labels
    ax.set_xlabel("W = {0:3.1f}".format(edges[0] * 2))
    ax.set_ylabel("L = {0:3.1f}".format(edges[2] * 2))
    ax.set_zlabel("H = {0:3.1f}".format(H_row))
    
    # Transparent spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Hide tick lines
    ax.tick_params(color= "white")
    
    #ax.grid(False)
    #ax.set_axis_off()

def process_csv(rpath, cfg):
    '''
    canopy geometryが入ったcsvを読み込み、LDやハウスの端点を計算して返す。
    '''

    df = pd.read_csv(rpath)
    df["Time"] = pd.to_datetime(df["Time"])
    df.set_index("Time", inplace= True)
    df["LD"] = df.apply(lambda row: cal_LD(cfg.W_row, row["H_row"], row["LAI"], cfg.L_row, cfg.n_row, cfg.W_path, cfg.W_margin, cfg.L_margin), axis = 1)
    df[["x_edge_p", "x_edge_n", "y_edge_p", "y_edge_n"]] = df.apply(lambda row: cal_greenhouse_edge(cfg.W_row, row["H_row"], cfg.L_row, cfg.n_row, cfg.W_path, cfg.W_margin, cfg.L_margin), axis = 1)    
    fig = plt.figure(figsize=(10,10))

    # グラフで確認。
    xfmt = mdates.DateFormatter('%m/%d')# 
    nrows = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(4, 6))
    ax1=plt.subplot(nrows,1,1)
    ax1.xaxis.set_major_formatter(xfmt)
    plt.plot(df["LAI"])
    ax1.set_ylabel('LAI (m2 m-2)')

    ax2=plt.subplot(nrows,1,2)
    plt.plot(df["LD"])
    ax2.set_ylabel('LD (m2 m-3)')
    ax2.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
    return df

def visualize_greenhouse(df, cfg, elev=30, azim=-85, roll=0, index = -1, alpha = 0.01, ps = 0.1):
    """
    読み込んだgreenhouseの座標をプロットする。
    
    入力
        df      --- canopy geometryのdataframe。
        index   --- dfのどの行を可視化するか（行によって、群落の高さなどが異なるはず）。
        ps      -- 出力時の畝内の点群の大きさ
        alpha   -- 出力時の畝内の点群のopacity
    """
    dummy = df.iloc[index].copy()
    radius_sun_orbit = max(cfg.L_row, cfg.W_row) / 2 *1.1 # 太陽軌道の見かけの半径

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    #東西南北の軸を入れる
    #draw_north_south_axis(radius_sun_orbit/10, cfg.azimuth_row, ax)

    # ハウスの境界を描く
    visualize_greenhouse_boundary(dummy["x_edge_p"], dummy["x_edge_n"], dummy["y_edge_p"], dummy["y_edge_n"], ax)

    # 畝を描く
    visualize_rows(cfg.W_row, dummy["H_row"], cfg.L_row, cfg.n_row, cfg.W_path, cfg.azimuth_row, ax, alpha, ps)

    # North arrowを入れる
    draw_north_arrow(radius_sun_orbit/10, cfg.azimuth_row, ax)

    # tickの調節
    adjust_ticks(cfg.W_row, dummy["H_row"], cfg.L_row, cfg.n_row, cfg.W_path, cfg.W_margin, cfg.L_margin, ax)
    ax.set_aspect('equal', adjustable='box')

    # 球を描く
    #draw_a_sphere(radius_sun_orbit, ax)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    return ax
# %%
if __name__ == "__main__":
    rdir = r"./sample/"
    file_name = "canopy_geometry.csv"
    file_name_config = "parameter_list_v2.yml"
    wfile_name = file_name[:-4]+"_processed.csv"

    file_path = os.path.join(rdir, file_name)
    config_path = os.path.join(rdir, file_name_config)
    # configパラメータの読み込み
    with open(config_path, "r") as file:
        d_config = yaml.safe_load(file)
    print(d_config)
    cfg = SimpleNamespace(**d_config)

    df = process_csv(file_path, cfg)
    df.to_csv(os.path.join(rdir, wfile_name))
    ax = visualize_greenhouse(df, cfg)
    #plt.plot([-10], [-10], "o", ms =10) #(-10, -10)は南東の点。
    ax.invert_yaxis()
# %%
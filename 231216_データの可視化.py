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

# %%

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


def draw_a_sphere(r, ax):
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
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    ax.plot_surface(x-c, y-c, z-c, color= "aqua", alpha=0.1)

def draw_a_north_arrow(r, azimuth_row):
    '''
    DEPRECIATED (使う際には方角について要検討！！)
    North arrowを描く．ただし，すでに，3D平面が作られていることを前提にする．
    つまり，メインプログラムにおいて，
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    が走っているものとする．
    また，以下のリンクから，3D arrowのclass群をコピペしているものとする．
    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    
        r -- 太陽公転軌道の半球の半径
    '''
    # 長さradius_sun_orbit/8のNorth arrowを(radius_sun_orbit, 0, 0)に描く．
    x1_arrow = 0
    x2_arrow = 0
    y1_arrow = r/8 #r/3
    y2_arrow = -r/8

    # 回転
    azimuth_row_rad = azimuth_row/180*np.pi
    x1_arrow_row_coord = x1_arrow*np.cos(azimuth_row_rad) - y1_arrow*np.sin(azimuth_row_rad)
    x2_arrow_row_coord = x2_arrow*np.cos(azimuth_row_rad) - y2_arrow*np.sin(azimuth_row_rad)
    y1_arrow_row_coord = y1_arrow*np.cos(azimuth_row_rad) + x1_arrow*np.sin(azimuth_row_rad)
    y2_arrow_row_coord = y2_arrow*np.cos(azimuth_row_rad) + x2_arrow*np.sin(azimuth_row_rad)

    # 平行移動
    x1_arrow_row_coord = x1_arrow_row_coord + r*0.8
    x2_arrow_row_coord = x2_arrow_row_coord + r*0.8
    y1_arrow_row_coord = y1_arrow_row_coord - r*0.8
    y2_arrow_row_coord = y2_arrow_row_coord - r*0.8

    dx_arrow_row_coord = x2_arrow_row_coord - x1_arrow_row_coord
    dy_arrow_row_coord = y2_arrow_row_coord - y1_arrow_row_coord

    ax.arrow3D(x1_arrow_row_coord,y1_arrow_row_coord,0,
            dx_arrow_row_coord,dy_arrow_row_coord,0,
            mutation_scale=20,
            #arrowstyle="-|>",
            #linestyle='dashed',
            color = "k",
            lw = 1
            )
    ax.annotate3D('N', (x2_arrow_row_coord, y2_arrow_row_coord, 0), xytext=(1, 1), textcoords='offset points')

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

    dx = (north_row_coord[0] - south_row_coord[0]) / 10
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

def visualize_rows(W_row, H_row, L_row, n_row, W_path, azimuth_row, ax):
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

    for z in np.arange(0, H_row, 0.1):
        for row_coord in list_row_coord:
            x_row = np.arange(row_coord[0], row_coord[1], 0.1)
            y_row = np.arange(row_coord[2], row_coord[3], 0.1)

            # mesh gridを作成する (列がx_row, 行がy_rowで構成される行列)
            x_row_m, y_row_m = np.meshgrid(x_row, y_row)

            # # 畝の方向が，南北方向からazimuth_rowだけ回転しているので，それを補正する．
            # x_row_coord = x_row_m *np.cos(azimuth_row_rad) - y_row_m * np.sin(azimuth_row_rad)
            # y_row_coord = y_row_m *np.cos(azimuth_row_rad) + x_row_m * np.sin(azimuth_row_rad)

            # x_row (あるいはy_row)と同じ形の行列をつくる
            z_row = np.full_like(x_row_m, z)

            ax.scatter(x_row_m, y_row_m, z_row, color = "g", s = 0.1, alpha =0.03)
            

    #ax.set_aspect('equal', adjustable='box')

    # 左手座標系で表示する; y軸の向きをひっくり返す
    # ax.invert_yaxis()

def create_cross_section_x_z(df_radiation, y, list_edge_negative_y, to_be_visualized, vmax, wdir):
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
        mappable = plt.contourf(z, x, dummy_pv.T, cmap = "jet", levels=600, norm=Normalize(vmin = 0, vmax = vmax))
        
    plt.gca().set_aspect('equal')
    mappable.set_clim(vmin = 0, vmax= vmax)
    sm = ScalarMappable(cmap="jet", norm=plt.Normalize(0, vmax))
    plt.colorbar(sm, orientation = "horizontal", ax = ax)
    plt.title(df_results["Time"][0].strftime("%Y-%m-%d_%H時%M分"), fontsize = 12)
    if wdir:
        wfile_name = df_results["Time"][0].strftime("%y%m%d_%H%M_xz_cross_sec_out")
        wfile_png  = os.path.join(wdir, wfile_name + ".png")
        plt.savefig(wfile_png)
    #plt.colorbar(mappable, ax = ax, orientation = "vertical")
    #plt.imshow(dummy_pv, cmap= "jet", interpolation = "gaussian", aspect = "equal", origin = "lower")
    plt.show()

# %%
def plot_3d_photo(df_input, to_be_visualized, y, vmax, I0_beam_h, I0_dif_h, A_per_ground, LAI, wdir = False):
    #############################
    # 断面図
    df_input = df_input.copy()

    #############################
    # 三次元
    fig = px.scatter_3d(df_input, x='x', y='y',z='z', color = to_be_visualized, 
                        color_continuous_scale=px.colors.sequential.Jet, opacity=0.4,
                        range_color = [0, vmax])
    fig.update_traces(marker_size = 2)
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
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
    u = - df_results["x_sun_row_coord"][0]
    v = - df_results["y_sun_row_coord"][0]
    w = - df_results["H_sun"][0]

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
    north_row_coord, south_row_coord, east_row_coord, west_row_coord = get_directional_axes_for_visualization(r, azimuth_row)

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
            df_results["Time"][0].strftime("%Y-%m-%d %H時%M分"), 
            df_results["Solar_elev"][0] / np.pi * 180,
            df_results["azm"][0] / np.pi * 180,
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
        eye=dict(x=1.2, y=1.8, z=1.2)
    )
    fig.update_layout(scene_camera=camera)


    if wdir:
        wfile_name = df_results["Time"][0].strftime("%y%m%d_%H%M_out")
        wfile_html = os.path.join(wdir, wfile_name + ".html")
        wfile_png  = os.path.join(wdir, wfile_name + ".png")
        fig.write_html(wfile_html)
        fig.write_image(wfile_png)
    fig.show()
    #$\mathrm{{\mu mol \; m^{{-2}}_{{ground}} \; s^{{-1}}}}$

# %%
###################################################
# パラメータの読み込み
rdir = "/home/koichi/pCloudDrive/01_Research/231007_畝を考慮に入れた群落光合成モデル/test_simulation/畝90度"
rpath = os.path.join(rdir, "parameter_list.yml")
with open(rpath, "r") as file:
    test = yaml.safe_load(file)
for key, val in test.items():
    exec(key + "=val")    
###################################################
# 計算結果の読み込み
rdir_results = os.path.join(rdir, "output")
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

print("-------------------------------------------")
print("ハウス情報")
print("   畝本数 = {0:3.1f}, \n   畝幅 = {1:3.1f} m, \n   通路幅 = {2:3.1f} m, \n   作物高 = {3:3.1f} m, \n   畝長 = {4:3.1f} m, \n   外側の畝の両側の通路幅 = {5:3.1f} m, \n   畝の前後面の外側の通路幅 = {6:3.1f} m, \n".format(n_row, W_row, W_path, H_row, L_row, W_margin, L_margin))
print("   ハウス幅 = {0:3.1f} m, \n   ハウス奥行 = {1:3.1f} m \n   ハウス面積 = {2:3.1f} a".format(W_ground, L_ground, A_ground_in_a))
print("   azimuth_row = {0:3.1f}; 真北が，畝の真正面から半時計周りに{0:3.1f}°傾いている".format(azimuth_row))
print("-------------------------------------------")

# #######################
# # 作物群落に関するパラメータ
LAI = H_row * W_row * L_row * n_row * LD / (W_ground * L_ground) 
print("-------------------------------------------")
print("作物のパラメータ")
print("   葉面積密度LD = {0:3.1f} m^2 m^-3, \n   LAI = {1:3.1f} m^2 m^-2  \n".format(LD, LAI))
print("-------------------------------------------")

# #######################
# # 個葉光合成に関するパラメータ
print("-------------------------------------------")
print("個葉光合成に関するパラメータ")
print("   Vcmax_25 = {0:3.1f} umol m-2 s-1, \n   C_Vcmax = {1:3.0f}, \n   Jmax_25 = {2:3.1f}, \n   C_Jmax = {3:3.0f}, \n   DHa_Jmax = {4:3.0f}, \n   DHd_Jmax = {5:3.0f}, \n   DS_Jmax = {6:3.0f}".format(Vcmax_25, C_Vcmax, DH_Vcmax, Jmax_25, C_Jmax, DHa_Jmax, DHd_Jmax, DS_Jmax))
print("   Rd_25 = {0:3.1f} umol m-2 s-1, \n   C_Rd = {1:3.0f}, \n   DH_Rd = {2:3.0f}, \n   m = {3:3.1f}, \n   b_dash = {4:4.3f}".format(Rd_25, C_Rd, DH_Rd, m, b_dash))
print("-------------------------------------------")

#######################
# 計算点に関するパラメータ
Nx_per_row = int(W_row / delta_x_row) # rowの中で，光を計算する点を作る際の，x方向への分割数
Ny_per_row = int(L_row / delta_y_row) # rowの中で，光を計算する点を作る際の，y方向への分割数
Nz_per_row = int(H_row / delta_z_row) # rowの中で，光を計算する点を作る際の，z方向への分割数

delta_x_row = W_row / Nx_per_row
delta_y_row = L_row / Ny_per_row
delta_z_row = H_row / Nz_per_row

Nx_per_btm = int(W_ground / 0.5) # 反射光計算のために，ハウスの底面に降り注ぐ光も計算する．そのときの，底の分割数．
Ny_per_btm = int(L_ground / 0.5)  # 反射光計算のために，通路に降り注ぐ光も計算する．そのときの，通路のy方向への分割数．

delta_x_btm = W_ground / Nx_per_btm
delta_y_btm = L_ground / Ny_per_btm

n_points = Nx_per_row * Ny_per_row * Nz_per_row + Nx_per_btm * Ny_per_btm 

print("-------------------------------------------")
print("数値計算に関する情報")
print("   畝の分割数 \n   x方向:{0}, \n   y方向:{1}, \n   z方向:{2}".format(Nx_per_row, Ny_per_row, Nz_per_row))
print("\n   畝の分割幅 \n   Δx = {0:5.3f} m, \n   Δy = {1:5.3f} m, \n   Δz = {2:5.3f} m".format(delta_x_row, delta_y_row, delta_z_row))
print("\n   反射光計算のための地面の分割数 \n   x方向:{0}, \n   y方向:{1}".format(Nx_per_btm, Ny_per_btm))
print("\n   反射光計算のための地面の分割幅 \n   Δx = {0:5.3f} m, \n   Δy = {1:5.3f} m".format(delta_x_btm, delta_y_btm))
print("\n 合計計算点数: {0} 点".format(n_points))

#######################
# 散乱光の数値計算に関するパラメータ
#delta = np.pi/1800 # 90 deg や 0 degなどでは，三角関数の計算値が不安定になる可能性があるので，そこを除外するためにdeltaを足し引きする．
# a1, b1, N1: diffuse radiationの計算における，betaの端点および分割数．
#N1 = 2 # まずN1分割から計算する．分割数を2倍ずつ増やして，増やす前後の計算値を比較して，収束判定する．
a1 = delta / 1.5
b1 = np.pi / 2 - delta

# a2, b2, N2: diffuse radiationの計算における，azmの端点および分割数
#N2 = 4 # まずN2分割から計算する．分割数を2倍ずつ増やして，増やす前後の計算値を比較して，収束判定する．
a2 = delta / 1.5
b2 = 2 * np.pi - delta

#max_iter = 10 # 散乱光の計算の最大反復数

print("\n散乱光・反射光の収束判定")
print("   最大繰り返し計算数:{0} 回, \n   許容絶対誤差 = {1} umol m-2 s-1".format(max_iter, acceptable_absolute_error))
print("-------------------------------------------")

# 畝の端の座標を計算
#######################
# 見た目に関するパラメータ
radius_sun_orbit = L_row / 2 *1.1 # 太陽軌道の見かけの半径

list_edge_negative_y, list_edge_positive_y = CP.cal_row_edges(W_row, H_row, L_row, n_row, W_path, azimuth_row)
print("-------------------------------------------")
print("畝の端の座標:")
print("x(間口)は{0:3.1f} m から {1:3.1f} mまでの範囲．".format(list_edge_negative_y[0][0], list_edge_negative_y[-1][0]))
row_xedge_list = [(round(list_edge_negative_y[i][0], 2), round(list_edge_negative_y[i+1][0], 2)) for i in np.arange(0, len(list_edge_negative_y), step = 2)]
print("畝のx座標（左，右）")
print(row_xedge_list)
print()
print("y(奥行き方向)は{0:3.1f} m から {1:3.1f} mまでの範囲．".format(list_edge_negative_y[0][1], list_edge_positive_y[0][1]))

print("-------------------------------------------")

# %%
# 葉面積あたりの光合成速度 (f_sunを考慮している；1m2の葉のうち、一部はsunlit, 一部はshadedとして計算している)
to_be_visualized = "A_per_LA"
#to_be_visualized = "I_abs_per_LA"
vmax = 25 #df_results[to_be_visualized].max()
vmin = -4
dfs_list = []
I_A_list = []
for rfile_results in rfile_results_list:
    if rfile_results.split(".")[-1] == "feather":
        rfile_path = os.path.join(rdir_results, rfile_results)
        df_results = feather.read_feather(rfile_path)
        dfs_list.append(df_results)

        # 太陽の位置を計算 (太陽軌道の半径をradius_sun_orbitとする)
        df_results = CP.cal_solar_position(df_results, radius_sun_orbit, azimuth_row)
        #######################

        y_list = df_results["y"].drop_duplicates()
        y = y_list.iloc[int(len(y_list)/2)]
        A_per_ground = (df_results["dV"] * LD * df_results["A_per_LA"]).sum() / (A_ground_in_a * 100)
        I_abs_per_ground = (df_results["dV"] * LD * df_results["I_abs_per_LA"]).sum() / (A_ground_in_a * 100)
        I_A_list.append([df_results["Time"][0], df_results["I0_beam_h"][0], df_results["I0_dif_h"][0], I_abs_per_ground, A_per_ground, df_results["x_sun_row_coord"][0], df_results["y_sun_row_coord"][0], df_results["H_sun"][0]])
        #vmax = df_results[to_be_visualized].max()
        #plot_3d_photo(df_results, to_be_visualized, vmin, vmax, df_results["I0_beam_h"][0], df_results["I0_dif_h"][0], A_per_ground, LAI, wdir)
        create_cross_section_x_z(df_results, y, list_edge_negative_y, to_be_visualized, vmax, wdir)

        print("\nTime = {0}, \nA_per_ground = {1:4.2f} umol m-2 ground s-1".format(df_results["Time"][0], A_per_ground))
        print("LAI = {0:3.2f}".format(LAI))
        print("Solar_elev= {0:3.2f}".format(df_results["Solar_elev"][0] / np.pi * 180))
# %%
# 日変化をまとめる
df_diurnal = pd.DataFrame(data = I_A_list, columns = ["Time", "I0_beam_h", "I0_dif_h", "I_abs_per_ground", "A_per_ground", "x_sun_row_coord", "y_sun_row_coord", "H_sun"])
df_diurnal["I0_h"] = df_diurnal["I0_beam_h"] + df_diurnal["I0_dif_h"]
A_day =df_diurnal["A_per_ground"].sum() * 3600 /10**6
print("日あたりの個体群光合成速度 = {0:3.2f} mol m-2 ground s-1".format(A_day))

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

# %%
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(projection='3d')

#東西南北の軸を入れる
draw_north_south_axis(radius_sun_orbit, azimuth_row, ax)

# 畝を描く
visualize_rows(W_row, H_row, L_row, n_row, W_path, azimuth_row, ax)

# 球を描く
draw_a_sphere(radius_sun_orbit, ax)

# 太陽の軌道を描く
dummy = df_diurnal.loc[df_diurnal["H_sun"] >= 0]
ax.scatter(dummy["x_sun_row_coord"], dummy["y_sun_row_coord"], dummy["H_sun"], s = 5, color = "red")
ax.plot3D(dummy["x_sun_row_coord"], dummy["y_sun_row_coord"], dummy["H_sun"], color = "red", lw = 2)

##############################################

#ax.view_init(elev=30, azim=45, roll=15)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_zlim(0, radius_sun_orbit)
ax.set_aspect('equal', adjustable='box')
#ax.view_init(elev=10, azim=60, roll=0)
# 左手座標系で表示する; y軸の向きをひっくり返す
# ### グラフの最後に必ず必要!
ax.invert_yaxis()


# # 3D畝および受光強度をポイントで表示
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(projection='3d')
# cm = plt.get_cmap("jet")

# ##########################
# # 東西南北の軸を入れる
# draw_north_south_axis(radius_sun_orbit, azimuth_row, ax)

# ##########################
# # 太陽の方向からの矢印の描画
# # 太陽の方向のベクトル
# u = - df["x_sun_row_coord"][0]
# v = - df["y_sun_row_coord"][0]
# w = - df["H_sun"][0]
# # メッシュ作成
# x, y, z = np.meshgrid(np.linspace(list_edge_negative_y[0][0], list_edge_negative_y[-1][0], 6),
#                     np.linspace(list_edge_negative_y[0][1], list_edge_positive_y[0][1], 6),
#                     np.array([H_row*2.5]))
# ax.quiver(x, y, z, u, v, w, length = 1, normalize = True, color = "r", alpha = 0.3)

# ##########################
# # 実際の光強度の描画
# vmax = I0_beam_h + I0_dif_h
# p = ax.scatter(df_radiation["x"], df_radiation["y"], df_radiation["z"], c = df_radiation["I_abs_per_LA"], s = 0.1, cmap = cm, norm=Normalize(vmin = 0, vmax = vmax))
# p.set_clim(vmin = 0, vmax= vmax)
# sm = ScalarMappable(cmap="jet", norm=plt.Normalize(0, vmax))
# plt.colorbar(sm, orientation = "vertical", ax = ax)

# ##########################
# # その他
# #ax.view_init(elev=30, azim=45, roll=15)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlim(0, radius_sun_orbit)
# #ax.view_init(elev=10, azim=60, roll=0)
# # 左手座標系で表示する; y軸の向きをひっくり返す
# # ### グラフの最後に必ず必要!
# ax.invert_yaxis()
# ax.set_aspect('equal', adjustable='box')

# plt.show()

##########################################################################################
# # 端点が計算できていないので，畝ごとに，プロット用に外挿する（ついでに内挿する）．
# ix = np.shape(list_edge_negative_y)[0] # 畝数 × 2
# ys = np.linspace(list_edge_negative_y[0][1], list_edge_positive_y[0][1])
# zs = np.linspace(0, list_edge_negative_y[0][2])

# xnew_list = []
# ynew_list = []
# znew_list = []
# I_dif_h_new_list = []
# I_beam_h_new_list = []
# I_ref_h_new_list = []
# for i in np.arange(0, ix, step=2):
#     df_dummy = df_radiation.loc[(list_edge_negative_y[i][0]<= df_radiation["x"]) & (df_radiation["x"] <= list_edge_negative_y[i+1][0])]
#     rbf_dif = Rbf(df_dummy["x"], df_dummy["y"], df_dummy["z"], df_dummy["I_dif_h"], function = "multiquadric", smooth =1)
#     rbf_beam = Rbf(df_dummy["x"], df_dummy["y"], df_dummy["z"], df_dummy["I_beam_h"], function = "multiquadric", smooth =1)
#     rbf_ref = Rbf(df_dummy["x"], df_dummy["y"], df_dummy["z"], df_dummy["I_ref_h"], function = "multiquadric", smooth =1)
    
#     xs = np.linspace(list_edge_negative_y[i][0], list_edge_negative_y[i+1][0])
#     xnew, ynew, znew = np.meshgrid(xs, ys, zs)
#     xnew = xnew.flatten()
#     ynew = ynew.flatten()
#     znew = znew.flatten()
#     I_dif_h_new = rbf_dif(xnew, ynew, znew)
#     I_beam_h_new = rbf_beam(xnew, ynew, znew)
#     I_ref_h_new = rbf_ref(xnew, ynew, znew)
#     xnew_list.append(xnew)
#     ynew_list.append(ynew)
#     znew_list.append(znew)
#     I_dif_h_new_list.append(I_dif_h_new)
#     I_beam_h_new_list.append(I_beam_h_new)
#     I_ref_h_new_list.append(I_ref_h_new)

# xnew_list = np.array(xnew_list).flatten()
# ynew_list = np.array(ynew_list).flatten()
# znew_list = np.array(znew_list).flatten()
# I_dif_h_new_list = np.array(I_dif_h_new_list).flatten()
# I_beam_h_new_list = np.array(I_beam_h_new_list).flatten()
# I_ref_h_new_list = np.array(I_ref_h_new_list).flatten()

# df_extrapolated = pd.DataFrame({"x": xnew_list, "y": ynew_list, "z": znew_list, "I_dif_h": I_dif_h_new_list, "I_beam_h": I_beam_h_new_list, "I_ref_h": I_ref_h_new_list})
# df_extrapolated["I_h_sum"] = df_extrapolated["I_dif_h"] + df_extrapolated["I_beam_h"] + df_extrapolated["I_ref_h"]

# %%
dfs_list[5].head()
# %%

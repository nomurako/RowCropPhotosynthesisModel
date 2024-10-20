#%%
import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import matplotlib.ticker as ticker
from scipy.special import beta
import argparse
import sys

plt.rcParams['axes.linewidth']=0.8
plt.rcParams['font.size']=12
#plt.rcParams['font.family'] = 'IPAexGothic'

# %%
def cal_ellipsoidal_G(x, theta):
    '''
    ellipsoidal leaf angle distributionのG-functionを直接計算する。
    Campbell (1986) の kの式と、Campbell (1990)のAの式とを合わせて、
    かつ、k = G/np.cos(theta_rad)より、得られたkにnp.cos(theta_rad)をかけてGに直した。
    何かの論文で、この書き方があったような気がするけど、思い出せない。
    args:
        x     -- ratio of horizontal to vertical.つまり， x = horizontal / vertical.
        theta -- view zenith angle (°).太陽光線の方向．
    '''
    theta_rad = theta/180 * np.pi
    A_ellip = (x+1.774*(x + 1.182)**(-0.733)) / x
    G_ellip = (x**2 + np.tan(theta_rad) **2)**0.5 / (A_ellip * x) * np.cos(theta_rad)
    return G_ellip  

def cal_ellipsoidal_G_from_beta_rad(x, beta_rad):
    '''
    beta_rad (太陽高度; radian表示)からGを計算する
    (こんがらがらないようにするためのwrapper関数)。
    '''
    theta_rad = np.pi/2 - beta_rad
    theta = theta_rad / np.pi *180
    G_ellip = cal_ellipsoidal_G(x, theta)
    return G_ellip

def cal_h_to_v_ratio_from_mean_leaf_angle(mean_leaf_angle):
    '''
    Ellipsoidal leaf angle distributionを使うために、
    平均葉角 (normal to leaf)からx = h/vを計算する。
    平均葉角は、水平葉でゼロとする。

    args:
        mean_leaf_angle --- 平均葉角（°）
    '''
    mean_leaf_angle_rad = mean_leaf_angle * (np.pi/180)
    x = -3 + (9.65/mean_leaf_angle_rad)** (1/1.65)
    return x


def cal_spherical_leaf_angle_dist(theta_L):
    '''
    spherical leaf angle distribution．球面の葉の角度分布関数（G-functionじゃないよ）
    これをtheta_Lで積分して，theta方向に投影すればG-functionになる（G = 0.5が計算できる）．
    args:
        theta -- 葉のnormalのzenith angle.
    '''
    theta_L_rad = theta_L/ 180 * np.pi
    return np.sin(theta_L_rad)

def cal_beta_leaf_angle_dist(theta_L, theta_L_avg, theta_L_var):
    '''
    beta leaf angle distribution．beta関数で表現した葉の角度分布関数．
    args:
        theta_L     -- leaf-angle (degree) at which f is calculated.葉のnormalとzenithとのなす角．0は上向き．
        theta_L_avg -- parameter (mean leaf-angle; 0 - 90)．平均の葉の角度．
        theta_L_var -- parameter (variance of leaf-angle)．葉の角度の分散．22**2とかになる．
        Wang et al. (2007)および研究ノートNo.24を参照．
    '''
    # 角度を0 - 1で表して，それをtと呼ぶことにする．
    t = theta_L /90
    t_avg = theta_L_avg /90
    t_var = theta_L_var /90**2

    t_std_max = t_avg * (1 - t_avg) # tの標準偏差の最大値（らしい．）
    nu = t_avg * (t_std_max / t_var -1)
    mu = (1 - t_avg) * (t_std_max / t_var -1)
    B = beta(mu, nu) # beta関数．
    f = 1/B * (1 - t)**(mu-1) *t**(nu-1) # theta_Lの角度の葉のfrequency.

    # fを0からπ/2で積分すれば，1になるようにする．
    # 今，fを0からπ/2で積分すると， π/2がでてくるようになっている．
    f = f* (2/np.pi) # これで，横軸はradianの確率密度関数になった．
    return f

# # test用．
# theta_dummy = np.arange(0,90,1)
# theta_L_avg =51
# theta_L_var =30**2
# f_test = [cal_beta_leaf_angle_dist(th, theta_L_avg, theta_L_var) for th in theta_dummy] #np.apply_along_axis(cal_G_function, 0, theta_dummy, x=x_ellip)
# f_test = np.array(f_test)
# spherical = cal_spherical_leaf_angle_dist(theta_dummy)
# plt.plot(f_test)
# plt.plot(spherical)

def cal_hj(theta, theta_L, dL):
    '''
    G(theta)を計算する際のintermediate variable.
    これにtheta_Lにおける葉の角度分布のfrequencyを掛けて，theta_L = 0 to π/2まで足し合わせればG(theta)を算出できる．
    解析解がないので，数値積分になる．f*hj*dLを0からπ/2まで積算することになる．
    args:
        theta    -- leafの投影方向．cameraのview zenith angleに相当する．
        theta_L  -- 葉のnormalのzenith angle．つまり，葉の向き．
        dL       -- theta_Lについての積分における刻み幅． 
    '''
    # radianに変換．
    theta_rad = theta / 180 * np.pi
    dL_rad = dL /180 * np.pi
    theta_L_rad = theta_L /180 * np.pi

    #  場合分けに使うthreshold
    thresh = abs(1/(np.tan(theta_rad)*np.tan(theta_L_rad)))
    
    # part of hj
    hj_sub = np.cos(theta_L_rad)*np.cos(theta_rad)* dL_rad
    if thresh > 1:
        hj = hj_sub
        # print("aaa",hj)
    else:
        # integral approximation by f_avg * dx
        cot_mul = 1 / (np.tan(theta_rad)* np.tan(theta_L_rad))
        myTAN = np.tan(np.arccos(cot_mul))
        hj_sub_2 = np.cos(theta_L_rad)* (myTAN - np.arccos(cot_mul))
        hj = hj_sub + 2/np.pi *np.cos(theta_rad) *hj_sub_2 * dL_rad
    return hj

def cal_beta_G(theta, theta_L_avg, theta_L_var, dL=0.1):
    '''
    beta functionによってG(theta)を計算する．
    G(theta)を算出するためには，すべてのtheta_Lで積分する必要がある．その積分をおこなう．
    args:
        theta       -- leafの投影方向．cameraのview zenith angleに相当する．
        theta_L_avg -- theta_L（葉のnormalのzenith angle; 葉の向き）の平均値
        theta_L_var -- theta_Lの分散．
        dL          -- theta_Lについての積分における刻み幅．
        
        例
            horizontal leaves  -- theta_L_avg = 90.00, theta_L_var = 0**2
            Cornus drummondii  -- theta_L_avg = 57.31, theta_L_var = 20.35**2
    '''
    theta_L_dummy = np.arange(0,90, dL)
    G_list = []
    for theta_L in theta_L_dummy:
        hj = cal_hj(theta, theta_L, dL)
        #print(hj)
        fj  = cal_beta_leaf_angle_dist(theta_L, theta_L_avg, theta_L_var)
        Gj = hj*fj
        G_list.append(Gj)
    G = np.array(G_list).sum()
    #print(G)
    return G

#%%
if __name__ == "__main__":
    theta_list = np.linspace(0, 85)
    theta_rad_list = theta_list * (np.pi/180)
    vfunc = np.vectorize(cal_ellipsoidal_G, excluded="x")

    # 平均葉角 (水平葉 = 0)
    for mean_leaf_angle in [0.1, 20, 30, 40, 57.3, 70, 89.9]:
        # xはellipsoidの水平軸を鉛直軸で割ったもの。
        # xが小さいということは、葉が立っている。
        x = cal_h_to_v_ratio_from_mean_leaf_angle(mean_leaf_angle)
        G_ellipsoid =vfunc(theta = theta_list, x = x)    
        plt.plot(theta_list, G_ellipsoid, label = mean_leaf_angle)
    plt.legend()
    plt.xlabel("View zenith angle")
    plt.ylabel("G")

    # kのプロット
    # 平均葉角 (水平葉 = 0)
    for mean_leaf_angle in [0.1, 20, 30, 40, 57.3, 70, 89.9]:
        # xはellipsoidの水平軸を鉛直軸で割ったもの。
        # xが小さいということは、葉が立っている。
        x = cal_h_to_v_ratio_from_mean_leaf_angle(mean_leaf_angle)
        G_ellipsoid =vfunc(theta = theta_list, x = x)    
        plt.plot(theta_list, G_ellipsoid/np.cos(theta_rad_list), label = mean_leaf_angle)
    plt.legend()
    plt.xlabel("View zenith angle")
    plt.ylabel("k")
    plt.ylim(0,3)

    cal_h_to_v_ratio_from_mean_leaf_angle(31.3)
    #%%
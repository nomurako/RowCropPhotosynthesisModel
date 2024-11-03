#%%
import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import matplotlib.ticker as ticker
from scipy.optimize import fsolve
import argparse
import sys

plt.rcParams['axes.linewidth']=0.8
plt.rcParams['font.size']=12
plt.rcParams['font.family'] = 'IPAexGothic'

# %%
#constant
R_gas=8.3144598 # J K-1 mol-1
M_H2O=18.01528 # g mol-1
M_CO2=44.01 # g mol-1
Pa=101325 # Pa
Oxy=210 # mmol mol-1; Oxygen_concentration

# %%
# functions
##################################################################
# 個葉光合成モデル
##################################################################
def cal_temp_function(c,dh,tl,R_gas): # Bernacchi function
    """calculate temperature dependence.

    The Bernacchi function was applied.

    Keyword arguments:
        c       --- 
        dh      --- 
        tl      --- leaf temperature in degC
        R_gas   --- 気体定数
    Returns:
        A parameter at the leaf temperature, tl.
    """       
    return np.exp(c-dh/(R_gas*(tl+273.15)))

def cal_temp_function_rev(tl,f,c,dha,dhd,ds, R_gas): # Bernacchi function
    """calculate temperature dependence.

    The Bernacchi function was applied.

    Keyword arguments:
        c           --- 
        dha         --- 
        dhd         ---
        ds          --- 
        tl          --- leaf temperature in degC
        R_gas       --- 気体定数

    Returns:
        A parameter at the leaf temperature, tl.
    """       
    return f*np.exp(c-dha/(R_gas*(tl+273.15)))/(1+np.exp((ds*(tl+273.15)-dhd)/(R_gas*(tl+273.15))))

def cal_J(Jmax,q,Phi_JQ,Beta_JQ,Theta_JQ, Alpha_L):
    """calculate electron transport rate, J.

    Keyword arguments:
        Jmax        --- Jmax in umol m-2 s-1
        q           --- incident PAR in umol m-2 s-1
        Phi_JQ      --- J-Q曲線の初期勾配 (0.85)
        Beta_JQ     --- PSIIへの分配率 (0.5)
        Theta_JQ    --- J-Q曲線の凸度 (0.7)
        Alpha_L     --- 葉の吸収率 (0.84)

    Returns:
        electron transport rate
    """       
    q2=q * Phi_JQ * Beta_JQ * Alpha_L
    return (q2+Jmax-((q2+Jmax)**2-4*q2*Jmax*Theta_JQ)**0.5)/(2*Theta_JQ)

def cal_J_from_Qabs(Jmax,q_abs,Phi_JQ,Beta_JQ,Theta_JQ):
    """    
    吸収光強度からJを計算する。
    群落光合成モデルから個葉光合成速度を計算するときに使う。
    引数q_absに吸収率はすでに加味されているので、ここではAlpha_Lをかけない

    Keyword arguments:
        Jmax        --- Jmax in umol m-2 s-1
        q_abs       --- 吸収PAR in umol m-2_leaf s-1
        Phi_JQ      --- J-Q曲線の初期勾配 (0.85)
        Beta_JQ     --- PSIIへの分配率 (0.5)
        Theta_JQ    --- J-Q曲線の凸度 (0.7)

    Returns:
        electron transport rate
    """       
    q2=q_abs * Phi_JQ * Beta_JQ # q_absに吸収率はすでに加味されているので、Alpha_Lをかけない。
    return (q2+Jmax-((q2+Jmax)**2-4*q2*Jmax*Theta_JQ)**0.5)/(2*Theta_JQ)

def cal_params_at_TL(TL, R_gas,
                     Vcmax_25, C_Vcmax, DH_Vcmax, 
                     Jmax_25, C_Jmax, DHa_Jmax, DHd_Jmax, DS_Jmax,
                     Rd_25, C_Rd, DH_Rd,
                     C_Kc, DH_Kc,
                     C_Ko, DH_Ko,
                     C_Gamma_star, DH_Gamma_star
                    ):
    '''
    温度依存性パラメータ群を計算する．
    
    入力
        TL      --- 葉温 (degC)

    '''
        
    Vcmax           = Vcmax_25 * cal_temp_function(C_Vcmax, DH_Vcmax, TL, R_gas)
    Jmax            = cal_temp_function_rev(TL, Jmax_25, C_Jmax, DHa_Jmax, DHd_Jmax, DS_Jmax, R_gas)
    Rd              = Rd_25 * cal_temp_function(C_Rd, DH_Rd, TL, R_gas)
    Kc              = cal_temp_function(C_Kc, DH_Kc, TL, R_gas)
    Ko              = cal_temp_function(C_Ko, DH_Ko, TL, R_gas)
    Gamma_star      = cal_temp_function(C_Gamma_star, DH_Gamma_star, TL, R_gas)

    return Vcmax, Jmax, Rd, Kc, Ko, Gamma_star

def cal_leaf_photo (Vcmax, J, Gamma_star, Kc, Ko, Oxy, Rd, m, b_dash, rh, gb, Ca):
    '''
    Baldocchi (1994)およびMasutomi (2023)を使って，個葉光合成速度を計算する．
    gb, gsはCO2に対する値．

    入力
        m           --- Ballモデルの傾き。水蒸気に対する値。 この関数内で1.6で割って、CO2に対する値に直す。
        b_dash      --- Ballモデルの切片。水蒸気に対する値。 この関数内で1.6で割って、CO2に対する値に直す。
        gb          --- CO2に対する葉面境界層コンダクタンス (mol m-2 s-1)
        rh          --- 相対湿度 (-)．パーセントではない．
        Oxy         --- 大気中の酸素濃度．210 mmol mol-1．
        Vcmax, J, Gamma_star, Kc, Rdはその温度における値 (25℃の値ではない)．

    Rubisco律速のとき
        a = Vcmax
        d = Gamma_star
        e = 1
        b = Kc * (1 + Oxy / Ko)    

    
    RuBP再生律速のとき
        a = J
        d = Gamma_star
        e = 4
        b = 8*Gamma_star

    出力
        Ac          --- Rubisco律速のときの純光合成速度
        gs_c        --- Rubisco律速のときの、CO2に対する気孔コンダクタンス
        Ci_c        --- Rubisco律速のときの葉内間隙中のCO2濃度
        Aj          --- RuBP律速のときの純光合成速度
        gs_j        --- RuBP律速のときの、CO2に対する気孔コンダクタンス
        Ci_j        --- RuBP律速のときの葉内間隙中のCO2濃度
        
    '''
    # Rubisco律速のときの光合成速度Acの計算
    
    def cal_cubic(a, d, e, b, m_co2, b_dash_co2): 

        alpha = 1 + b_dash_co2 / gb - m_co2 * rh
        beta  = Ca * (gb * m_co2 * rh - 2*b_dash_co2 - gb)
        gamma = Ca ** 2 * b_dash_co2 * gb
        theta = gb * m_co2 * rh - b_dash_co2

        aa    = 1
        bb    = (e * beta + b * theta - a * alpha + e * alpha * Rd) / (e * alpha)
        cc    = (e* gamma + b * gamma / Ca - a * beta + a * d * theta + e * Rd * beta + Rd * b * theta) / (e * alpha)
        dd    = (- a * gamma + a * d * gamma / Ca + e * Rd * gamma + Rd * b * gamma / Ca)  / (e * alpha)
        
        p     = (- bb**2 + 3 * aa * cc) / (3 * aa **2)
        q     = (2 * bb ** 3 - 9 * aa * bb * cc + 27 * aa**2 * dd) / (27 * aa **3)
        omega = (-1 + np.sqrt(3) * 1j) / 2

        ss = - bb / (3 * aa)

        # tt, uuは(1/3)乗根。負の数の(1/3)乗根は複数の値を取る。
        # つまり、aが負の数のとき、a**(1/3)は、
        # abs(a) ** (1/3)、
        # omega * abs(a) ** (1/3)
        # (omega**2) * abs(a) ** (1/3)
        # のいずれかの値になる。
        # また、a ** 0.5において、"複素数アリ"とpythonに予め分からせるには、 (a + 0j)**0.5とする。そうでなければaが負のときにエラー（nan）になる
        tt = ((3 * np.sqrt(3) * q + (27 * q**2 + 4 * p**3 + 0j)**0.5) / (6 * np.sqrt(3))) ** (1/3.0)
        uu = ((3 * np.sqrt(3) * q - (27 * q**2 + 4 * p**3 + 0j)**0.5) / (6 * np.sqrt(3))) ** (1/3.0)
        x1 = ss - tt - uu
        x2 = ss - tt * omega - uu * omega**2
        x3 = ss - tt * omega**2 - uu * omega

        return x1, x2, x3

    def cal_quadratic(a, d, e, b, b_dash_co2):
        gl = b_dash_co2 * gb / (b_dash_co2 + gb)
        
        aa = 1
        bb = - (gl / e) * (e * Ca + b - e * Rd / gl + a / gl)
        cc = - (gl / e) * (Rd * (e * Ca + b) - a * (Ca - d))

        x1 = (-bb + (bb**2 - 4 * aa * cc)**0.5) / (2 * aa)
        x2 = (-bb - (bb**2 - 4 * aa * cc)**0.5) / (2 * aa)
        return x1, x2

    def cal_gs_from_A(A, m_co2, b_dash_co2, rh, Ca, gb):
        gs = m_co2 * rh / (Ca - A/gb) * A + b_dash_co2
        return gs
    
    def cal_Ci_from_gs(gs, gb, A, Ca):
        gl = gs*gb / (gs + gb)
        Ci = Ca - A/gl
        return Ci
    
    def check_A_gs_Ci_cubic(A, gs, Ci):
        '''
        もしA, gs, Ciが方程式の
        requirementsを満たすならば，Trueを返す．
        '''
        if not (abs(A.imag)> 10**(-10)):
            if A.real >= 0:
                if gs.real > 0:
                    if Ci.real >0:
                        return True
        return False
    ####################################################################################
    # m, bを水蒸気に対する値からCO2に対する値に変換する。
    m_co2 = m/1.6
    b_dash_co2 = b_dash/1.6

    # Rubisco律速
    Ac_list = cal_cubic(Vcmax, Gamma_star, 1, Kc * (1 + Oxy / Ko), m_co2, b_dash_co2)
    gs_c_list = []
    Ci_c_list = []
    for dummy in Ac_list:
        gs_c = cal_gs_from_A(dummy, m_co2, b_dash_co2, rh, Ca, gb)
        Ci_c = cal_Ci_from_gs(gs_c, gb, dummy, Ca)
        gs_c_list.append(gs_c)
        Ci_c_list.append(Ci_c)

    # RuBP再生律速
    Aj_list = cal_cubic(J, Gamma_star, 4, 8*Gamma_star, m_co2, b_dash_co2)
    gs_j_list = []
    Ci_j_list = []
    for dummy in Aj_list:
        gs_j = cal_gs_from_A(dummy, m_co2, b_dash_co2, rh, Ca, gb)
        Ci_j = cal_Ci_from_gs(gs_j, gb, dummy, Ca)
        gs_j_list.append(gs_j)
        Ci_j_list.append(Ci_j)

    A_gs_Ci_c_zip = zip(Ac_list, gs_c_list, Ci_c_list)
    A_gs_Ci_j_zip = zip(Aj_list, gs_j_list, Ci_j_list)
    
    Ac = -10000
    cubic_test_c = []
    for Ac_gs_Ci_c in A_gs_Ci_c_zip:
        if check_A_gs_Ci_cubic(Ac_gs_Ci_c[0], Ac_gs_Ci_c[1], Ac_gs_Ci_c[2]):
            Ac   = Ac_gs_Ci_c[0].real
            gs_c = Ac_gs_Ci_c[1].real
            Ci_c = Ac_gs_Ci_c[2].real
            cubic_test_c.append(True)
        else:
            cubic_test_c.append(False)
    #if not np.any(cubic_test_c):
    #    print(cubic_test_c)
    #     print("Ac", Ac_list)
    #     print("Ca", Ca)
    #     #print("gs", gs_c_list)
    #     #print("Ci", Ci_c_list)

    Aj = -10000
    for Ac_gs_Ci_j in A_gs_Ci_j_zip:
        if check_A_gs_Ci_cubic(Ac_gs_Ci_j[0], Ac_gs_Ci_j[1], Ac_gs_Ci_j[2]):
            Aj   = Ac_gs_Ci_j[0].real
            gs_j = Ac_gs_Ci_j[1].real
            Ci_j = Ac_gs_Ci_j[2].real
    
    # Ac>=0の解がなければ，Ac < 0, gs = 定数 = b_dashの解しか存在しない．
    if Ac == -10000:
        #print("/n2次方程式_Ac")
        Ac_list_quadratic   = cal_quadratic(Vcmax, Gamma_star, 1, Kc * (1 + Oxy / Ko), b_dash_co2)        
        #print(Ac_list_quadratic)
        for dummy in Ac_list_quadratic:
            Ci_dummy = cal_Ci_from_gs(b_dash_co2, gb, dummy, Ca)
            if (Ci_dummy > 0) :
                Ac = dummy.real
                gs_c = b_dash_co2.real
                Ci_c = Ci_dummy.real
                #print("[OK]  ", Ac, gs_c, Ci_c)
            # else: # Ac_list_quadraticのうち、片方が[OK]であれば問題ないので、ここでもし[NOT OK]が出たとしても問題ない
            #     print("[NOT OK]  ", dummy, b_dash_co2, Ci_dummy)
            #     print("このとき、Ac = {0:4.2f}, gs_c = {1:4.2f}, Ci_c = {2:4.2f}".format(Ac, gs_c, Ci_c))
            #     print("Vcmax ={:3.2f}, J ={:3.2f}, Gamma_star ={:3.2f}, Kc = {:3.2f}, Ko ={:3.2f}, Oxy = {:3.2f}, Rd = {:3.2f}, m = {:3.2f}, b_dash = {:3.2f}, rh = {:3.2f}, gb = {:3.2f}, Ca = {:3.2f}".format(Vcmax, J, Gamma_star, Kc, Ko, Oxy, Rd, m, b_dash, rh, gb, Ca))


    if Aj == -10000:
        #print("/n2次方程式_Aj")
        Aj_list_quadratic   = cal_quadratic(J, Gamma_star, 4, 8*Gamma_star, b_dash_co2)
        
        for dummy in Aj_list_quadratic:
            Ci_dummy = cal_Ci_from_gs(b_dash_co2, gb, dummy, Ca)
            if (Ci_dummy > 0) :
                Aj = dummy.real
                gs_j = b_dash_co2.real
                Ci_j = Ci_dummy.real

    return Ac, gs_c, Ci_c, Aj, gs_j, Ci_j                


#%%# test
if __name__ == "__main__":
    #constants and functions related to FvCB model
    Vcmax_25 = 90
    C_Vcmax  = 17.7
    DH_Vcmax = 43694

    Jmax_25  = 200
    C_Jmax   = 25.3
    DHa_Jmax = 62295
    DHd_Jmax = 123553
    DS_Jmax  = 400

    Rd_25 = 1.5
    C_Rd  = 18.7
    DH_Rd = 46390

    Kc_25=404.9
    C_Kc=38.05
    DH_Kc=79.43*10**3

    Ko_25=278.4
    C_Ko=20.30
    DH_Ko=36.38*10**3

    Gamma_star_25=42.75
    C_Gamma_star=19.02
    DH_Gamma_star=37.83*10**3

    m = 10
    b_dash = 0.005

    Phi_JQ   = 0.85
    Beta_JQ  = 0.5
    Alpha_L  = 0.85
    Theta_JQ = 0.7

    Ca_list = np.linspace(550, 555, 2)
    Q_list  = np.linspace(0, 2200)
    Ta_list = np.linspace(27, 30, 2)
    RH_list = np.linspace(0.05,1.0, 10)
    gb_list = np.linspace(0.01,10, 10)

    # 温度依存性パラメータを計算する．
    Ta_series = pd.Series(Ta_list)
    params_TL = Ta_series.apply(lambda Ta: cal_params_at_TL(Ta, R_gas,
                    Vcmax_25, C_Vcmax, DH_Vcmax,
                    Jmax_25, C_Jmax, DHa_Jmax, DHd_Jmax, DS_Jmax,
                    Rd_25, C_Rd, DH_Rd,
                    C_Kc, DH_Kc, 
                    C_Ko, DH_Ko,
                    C_Gamma_star, DH_Gamma_star))
    df_params_TL =pd.DataFrame(params_TL.tolist(), columns = ["Vcmax", "Jmax", "Rd", "Kc", "Ko", "Gamma_star"])
    df_params_TL["TL"] = Ta_series

    # %%
    df_env = pd.DataFrame(np.array(np.meshgrid(Ca_list, Q_list, Ta_list, RH_list, gb_list)).T.reshape(-1, 5), columns = ["Ca", "Q", "TL", "RH", "gb"])
    df_env.head()
    # cal_params_at_TL(30, R_gas,
    #                  Vcmax_25, C_Vcmax, DH_Vcmax,
    #                  Jmax_25, C_Jmax, DHa_Jmax, DHd_Jmax, DS_Jmax,
    #                  Rd_25, C_Rd, DH_Rd,
    #                  C_Kc, DH_Kc, 
    #                  C_Ko, DH_Ko,
    #                  C_Gamma_star, DH_Gamma_star)

    # %%
    df_env2 = df_env.merge(df_params_TL)

    #%%
    df_env2["J"] = df_env2.apply(lambda row: cal_J(row["Jmax"], row["Q"], Phi_JQ, Beta_JQ, Theta_JQ, Alpha_L), axis = 1)

    # %%
    results = df_env2.apply(lambda row: cal_leaf_photo(row["Vcmax"], row["J"], row["Gamma_star"], row["Kc"], row["Ko"], Oxy, row["Rd"], m, b_dash, row["RH"], row["gb"], row["Ca"]), axis = 1)
    df_results = pd.DataFrame(results.tolist(), columns = ["Ac", "gs_c", "Ci_c", "Aj", "gs_j", "Ci_j"])
    df_results = pd.concat([df_env2, df_results], axis = 1)
    df_results

    # %%
    # 確認plot
    Ca = Ca_list[1]
    Q = Q_list[1]
    Ta = Ta_list[1]
    gb = gb_list[-1]
    RH = RH_list[-1]
    dummy = df_results.loc[(df_results["RH"] == RH) & (df_results["Ca"] == Ca) 
                        & (df_results["TL"] == Ta) & (df_results["gb"] == gb)]
    print("Ca ={}, Q = {}, Ta = {}, gb = {}, RH = {}".format(Ca, Q, Ta, gb, RH))
    plt.plot(dummy["Q"], dummy["Ac"])
    plt.plot(dummy["Q"], dummy["Aj"])

    # %%
        
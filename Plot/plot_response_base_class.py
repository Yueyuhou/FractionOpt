import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import math
import pathlib
from pathlib import Path
from Utils.readAndWrite.read_config_file import get_config_info
import pandas as pd
from RT_env.envs.ResponseModelClass import ResponseModelBase, LogisticPSIResponseModel, LogisticPSIResponseModelEnv3
import seaborn as sns
from scipy import stats
from itertools import permutations, combinations

pd.set_option('display.max_columns', None)
sns.set_context("paper")


def rename_df_and_save(df, save_path=None):
    paras_name = ['patient_id', 'convRT_ptv_BED', 'adapRT_ptv_BED', "convRT_oar_BED", 'convRT_cutoff_OAR_BED',
                  'adapRT_oar_BED', 'alpha', 'lambda', 'PSI', 'tumor_V0',
                  'conRT_final_response', 'adaRT_final_response']

    dose_name = [i for i in range(len(df.columns) - len(paras_name))]
    value = paras_name + dose_name
    keys = df.columns
    new_col = dict(zip(keys, value))
    df.rename(columns=new_col, inplace=True)

    if save_path is not None:
        df.to_excel(save_path, index=False)
    return df


def count_improved_survival_ratio(df, cutoff_value):
    # The first quadrant: improved survival rate for both adaRT and conRT
    condition = (df['adaRT_final_response'] <= cutoff_value) & (df['conRT_final_response'] <= cutoff_value)
    freq_1 = df[condition].shape[0] / df.shape[0] * 100

    # The second quadrant: improved survival rate for conRT
    condition = (df['adaRT_final_response'] > cutoff_value) & (df['conRT_final_response'] <= cutoff_value)
    freq_2 = df[condition].shape[0] / df.shape[0] * 100

    # The third quadrant: There is no improved survival rate for both RT.
    condition = (df['adaRT_final_response'] > cutoff_value) & (df['conRT_final_response'] > cutoff_value)
    freq_3 = df[condition].shape[0] / df.shape[0] * 100

    adaRT = df.loc[condition, 'adaRT_final_response']
    adaRT_mean = adaRT.mean()
    adaRT_std = adaRT.std()

    conRT = df.loc[condition, 'conRT_final_response']
    conRT_mean = conRT.mean()
    conRT_std = conRT.std()
    print("final response: ")
    print("adaRT_mean, adaRT_std, conRT_mean, conRT_std: ", [adaRT_mean, adaRT_std, conRT_mean, conRT_std])
    # 执行t检验
    # 正态性检验
    s, p = stats.shapiro(adaRT.values - conRT.values)
    print("Shapiro-Wilk Test: ", s, p)
    t_statistic, p_value = stats.ttest_rel(adaRT.values, conRT.values)
    print("T-Test: ", t_statistic, p_value)
    # s, p = stats.wilcoxon(adaRT.values-conRT.values)
    # print("Wilcoxon符号秩检验统计量:", s)
    # print("p值:", p)

    # The fourth quadrant: improved survival rate for adaRT
    condition = (df['adaRT_final_response'] <= cutoff_value) & (df['conRT_final_response'] > cutoff_value)
    freq_4 = df[condition].shape[0] / df.shape[0] * 100

    return [freq_1, freq_2, freq_3, freq_4]


def save_fig(img_save_path, suffix=None, patient_idx=None, subfolder=None):
    if not img_save_path.exists():
        img_save_path.mkdir(parents=True)

    if patient_idx is not None:
        pth = img_save_path.joinpath(subfolder)
        if not pth.exists():
            pth.mkdir(parents=True)

        plt.savefig(pth.joinpath(suffix + '_patient_' + str(patient_idx) + '.png'),
                    dpi=600, bbox_inches='tight')
    else:
        plt.savefig(img_save_path.joinpath(suffix + '.pdf'),
                    dpi=600, bbox_inches='tight')
    plt.close()


def read_fraction_scheme_and_write(data_path, save_path=None):
    data_path = Path(data_path)
    # 如果 data_path是.csv文件，那么读取文件
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path, names=range(200))
        df.dropna(axis=1, how='all', inplace=True)

        if save_path is None:
            excel_save_path = Path(r"../Result").joinpath(data_path.stem + '.xlsx')
        else:
            excel_save_path = Path(save_path).joinpath(data_path.stem + '.xlsx')

        df = rename_df_and_save(df, excel_save_path)
    elif data_path.suffix == '.xlsx':
        df = pd.read_excel(data_path)
    else:
        raise ValueError("The data_path should be a .csv or .xlsx file.")

    return df


class TumorResponsePlotBase:
    def __init__(self, data_path, config_info):
        self.data_path = Path(data_path)
        self.config_info = config_info
        self.alpha, self.lmd, self.PSI, self.V0 = None, None, None, None
        self.patient_idx = None
        self.df = None
        self.response_model = None

    def model_initialization(self):
        self.df = read_fraction_scheme_and_write(self.data_path)
        self.response_model = LogisticPSIResponseModel(self.df, self.config_info)

    def parameters_init(self, patient_idx, seed=1, alpha_=None, lmd_=None, PSI_=None, tumor_V0=None):
        alpha, lambda_, PSI, tumor_V0 = \
            self.response_model.initialization(patient_idx, seed, alpha_, lmd_, PSI_, tumor_V0)
        return {"alpha": alpha, "lambda_": lambda_, 'PSI': PSI, 'tumor_V0': tumor_V0}

    def get_conRT_response(self):
        return self.response_model.convRT_response.copy()

    def get_adaRT_response(self):
        return self.response_model.adapRT_response.copy()

    def plot_different_PSI_for_response(self, patient_idx=0, tumor_V0=100., alpha_=None, lmd=None,
                                        lines=200,
                                        img_save_path=None):
        # plot response to conventional RT, with different PSI values
        # alpha = mean_alpha, lmd = mean_lmd

        if alpha_ is None:
            alpha_ = self.config_info['tumor_rt_parameters']['alpha_mean']

        if lmd is None:
            lmd = self.config_info['tumor_rt_parameters']['lambda_mean']

        min_PSI = self.config_info['tumor_rt_parameters']['PSI_min']
        max_PSI = self.config_info['tumor_rt_parameters']['PSI_max']

        PSI_list = np.arange(min_PSI, max_PSI, (max_PSI - min_PSI) / lines)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        colors = cmap((PSI_list - min_PSI) / (np.amax(PSI_list) - np.amin(PSI_list)))

        sns.set_context("paper")
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, psi in enumerate(PSI_list):
            self.parameters_init(patient_idx=patient_idx, seed=1, alpha_=alpha_, lmd_=lmd,
                                 PSI_=psi, tumor_V0=tumor_V0)
            conRT_response = self.get_conRT_response()
            print(len(conRT_response))
            df = pd.DataFrame({"Time": list(range(len(conRT_response))), "Volume":  conRT_response})
            # ax.plot(list(range(len(conRT_response))), conRT_response, c=colors[i])
            ax = sns.lineplot(df, x='Time', y='Volume', c=colors[i], ax=ax)

        # 设置x轴和y轴范围从整数开始
        ax.set_xlim(left=0, right=41)  # 设置x轴从0开始
        ax.set_ylim(bottom=20, top=100)  # 设置y轴从0开始

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_PSI, vmax=max_PSI))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('PSI Value', fontsize=14, fontweight='medium')  #, fontweight='semibold'

        ax.set_xlabel('Time/day', fontsize=14, fontweight='medium')  # fontweight='bold'
        ax.set_ylabel('Relative Volume/%', fontsize=14, fontweight='medium')
        ax.set_title(
            "Tumor Responses to RT with Different PSI Values \n ($\\alpha=0.09/Gy^{-1}$, $\\lambda=0.07/day^{-1}$)",
            pad=20, fontsize=16, fontweight='medium')
        # 增大坐标轴刻度的字号
        ax.tick_params(axis='both', which='major', labelsize=12)
        # ax.set_title("Tumor Responses to RT with Different $\\alpha$ and $\\lambda$ combinations \n (PSI=0.85)",
        #             fontsize=14,
        #             pad=20)

        # draw a horizontal line at y=67
        plt.axhline(y=76.0, color='r', linestyle='--')

        plt.tight_layout()

        if img_save_path is None:
            plt.show()
        else:
            save_fig(img_save_path, 'Revised_different_PSI_for_response')

    def plot_different_alpha_lmd_for_response(self, patient_idx=0, tumor_V0=100., PSI=None, num=10, img_save_path=None):
        # plot response to conventional RT, with different alpha/lambda values
        # PSI = mean_PSI

        if PSI is None:
            PSI = self.config_info['tumor_rt_parameters']['PSI_mean']

        min_alpha = self.config_info['tumor_rt_parameters']['alpha_min']
        max_alpha = self.config_info['tumor_rt_parameters']['alpha_max']

        min_lmd = self.config_info['tumor_rt_parameters']['lambda_min']
        max_lmd = self.config_info['tumor_rt_parameters']['lambda_max']

        alpha_list = np.arange(min_alpha, max_alpha, (max_alpha - min_alpha) / num)
        alpha_list = np.around(alpha_list, decimals=3)
        lmd_list = np.arange(max_lmd, min_lmd, (min_lmd - max_lmd) / num)
        lmd_list = np.around(lmd_list, decimals=3)
        df = pd.DataFrame(data=np.zeros((num, num)), index=lmd_list, columns=alpha_list)

        comb_array1, comb_array2 = np.meshgrid(lmd_list, alpha_list)

        for lmd, alpha in zip(comb_array1.ravel(), comb_array2.ravel()):
            self.parameters_init(patient_idx=patient_idx, seed=1, alpha_=alpha, lmd_=lmd, PSI_=PSI, tumor_V0=tumor_V0)
            conRT_response = self.get_conRT_response()
            df.loc[lmd, alpha] = conRT_response[-1]

        fig, ax = plt.subplots(figsize=(8, 6))
        g = sns.heatmap(df, cmap=sns.color_palette("vlag", as_cmap=True),
                        cbar_kws={'format': mpl.ticker.PercentFormatter(decimals=0, is_latex=True)},
                        annot=True, fmt='.1f', vmin=0, vmax=110, ax=ax)

        g.set_xticks(np.arange(len(alpha_list)))
        g.set_yticks(np.arange(len(lmd_list)))
        g.set_xticklabels(['{:.2e}'.format(val) for val in alpha_list])
        g.set_yticklabels(['{:.2e}'.format(val) for val in lmd_list])

        g.set_xlabel(r'$\alpha /Gy^{-1}$', fontsize=14, fontweight='medium')
        g.set_ylabel(r'$\lambda /day^{-1}$', fontsize=14, fontweight='medium')
        g.set_title("Tumor Responses to RT with Different $\\alpha$ and $\\lambda$ combinations \n (PSI=0.85)",
                    fontsize=16,
                    pad=20, fontweight='medium')

        plt.xticks(rotation=30)
        plt.yticks(rotation=30)
        # 增大坐标轴刻度的字号
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

        if img_save_path is not None:
            save_fig(img_save_path, 'Revised_different_alpha_lmd_for_response')
        else:
            plt.show()

    def plot_3d_model_parameters_2_rsp(self, point_num, save_path=None, save_suffix=None, tumor_V0=100):
        def get_range(config_info, key, point_num, decimals=3):
            min_val = config_info['tumor_rt_parameters'][f'{key}_min']
            max_val = config_info['tumor_rt_parameters'][f'{key}_max']
            val_list = np.arange(min_val, max_val, (max_val - min_val) / point_num)
            return np.around(val_list, decimals)

        alpha_list = get_range(self.config_info, 'alpha', point_num)
        lmd_list = get_range(self.config_info, 'lambda', point_num)
        PSI_list = get_range(self.config_info, 'PSI', point_num)
        # tumor_V0_list = get_range(self.config_info, 'tumor_V0', point_num)

        # 将4个数组内的数字组合成一个4维数组
        # comb_array1, comb_array2, comb_array3, comb_array4 = np.meshgrid(alpha_list, lmd_list, PSI_list, tumor_V0_list)
        comb_array1, comb_array2, comb_array3 = np.meshgrid(alpha_list, lmd_list, PSI_list)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # 绘制散点图
        prediction = []
        cutoff_threshold = self.config_info['tumor_rt_parameters']['cutoff_response']

        for alpha, lmd, PSI in zip(comb_array1.ravel(), comb_array2.ravel(),
                                   comb_array3.ravel()):
            self.parameters_init(patient_idx=0, seed=1, alpha_=alpha, lmd_=lmd, PSI_=PSI, tumor_V0=tumor_V0)
            conRT_response = self.get_conRT_response()
            prediction.append(int(conRT_response[-1] <= cutoff_threshold))

        prediction = np.array(prediction)
        good_subgroup = prediction > 0
        ax.scatter(comb_array2.ravel()[good_subgroup], comb_array1.ravel()[good_subgroup],
                   comb_array3.ravel()[good_subgroup],
                   marker='o', c='tab:orange', alpha=0.6)
        bad_subgroup = prediction < 1
        ax.scatter(comb_array2.ravel()[bad_subgroup], comb_array1.ravel()[bad_subgroup],
                   comb_array3.ravel()[bad_subgroup],
                   marker='o', c='tab:blue', alpha=0.6)

        ax.set_ylabel(r'$\alpha /Gy^{-1}$', fontsize=14, fontweight='medium')
        ax.set_xlabel(r'$\lambda /day^{-1}$', fontsize=14, fontweight='medium')
        ax.set_zlabel('PSI', fontsize=14, fontweight='medium')
        plt.legend(['Low-risk group for LRC', 'High-risk group for LRC'], loc='upper right')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.title(f'Tumor Response to RT under Different Model Parameters (tumor $V_{0}$={tumor_V0} cm$^3$)',
                  fontsize=16, fontweight='medium')
        if save_path is not None:
            save_fig(save_path, suffix=save_suffix)
        else:
            plt.show()


if __name__ == "__main__":
    config_path = Path(r"../Utils/config_env_HN.yml")
    config_info = get_config_info(config_path)
    frac_path_HN = Path(r"../Data/frac_log/ppo_env2/v2_train_ppo_18o_D6a_HN_LRC_60Gy_de-escalation.csv")
    # frac_scheme_path = Path(r"../Data/frac_log/ppo_env3/v3_train_wd_18o6a.csv")

    img_save_path = Path(r"../Result/env2")

    if not img_save_path.exists():
        img_save_path.mkdir(parents=True)

    model = TumorResponsePlotBase(frac_path_HN, config_info)
    model.model_initialization()
    # model.plot_different_PSI_for_response(patient_idx=0, tumor_V0=100., lines=25,
    #                                       img_save_path=img_save_path)
    # model.plot_different_alpha_lmd_for_response(patient_idx=0, tumor_V0=100., num=10,
    #                                             img_save_path=img_save_path)
    model.plot_3d_model_parameters_2_rsp(point_num=15, tumor_V0=100, save_path=img_save_path,
                                        save_suffix='3d_model_rsp_V100')

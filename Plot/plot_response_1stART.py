import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import pathlib
from pathlib import Path
import glob
from Utils.readAndWrite.read_config_file import get_config_info
import pandas as pd
from RT_env.envs.ResponseModelClass import ResponseModelBase, LogisticPSIResponseModel
import seaborn as sns
from scipy import stats
from plot_response_base_class import TumorResponsePlotBase, rename_df_and_save, save_fig, \
    count_improved_survival_ratio, read_fraction_scheme_and_write
import noise_plot
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


pd.set_option('display.max_columns', None)
sns.set_context("paper")


def compute_adapRT_rsp_(df, rsp, time_span, patient_id_list, is_final_rsp):
    final_rsp_list = []

    patient_id_list = patient_id_list if patient_id_list is not None else list(df.index)

    for i in patient_id_list:
        dose_list = df.loc[i, list(range(1, time_span, 1))].dropna().to_numpy()
        _ = rsp.initialization(i, seed=1, alpha_=df.loc[i, 'alpha'], lmd_=df.loc[i, 'lambda'],
                               PSI_=df.loc[i, 'PSI'], tumor_V0=df.loc[i, 'tumor_V0'])
        adapRT_response = rsp.cal_RT_response(dose_list=dose_list.tolist()).copy()
        if is_final_rsp:
            final_rsp_list.append(adapRT_response[-1])
        else:
            final_rsp_list.append(adapRT_response)

    return final_rsp_list


def stat_final_rsp(df_path, config_info):
    df = read_fraction_scheme_and_write(df_path)
    cutoff_rsp = config_info['tumor_rt_parameters']['cutoff_response']
    conRT_final_rsp = df['conRT_final_response']
    adaRT_final_rsp = df['adaRT_final_response']

    num = adaRT_final_rsp[adaRT_final_rsp <= cutoff_rsp].shape[0]
    print("High LRC ratio for ART: ", num / len(adaRT_final_rsp))
    print("conRT_final_rsp describe: ", conRT_final_rsp.describe())
    print("adaRT_final_rsp describe: ", adaRT_final_rsp.describe())


class ARTResponsePlotEnv2(TumorResponsePlotBase):
    def __init__(self, path_dict, config_info_dict):
        super().__init__(path_dict['frac_path_HN'], config_info_dict['HN'])
        self.path_dict = path_dict
        self.config_info_dict = config_info_dict

    @staticmethod
    def model_initialization_env2(data_path, config_info):
        df = read_fraction_scheme_and_write(data_path)
        response_model = LogisticPSIResponseModel(df, config_info)
        return df, response_model

    @staticmethod
    def print_data_description(df, thereshold, info):
        print(info)
        print('超出阈值的个数: ', len(df[df > thereshold]),
              len(df[df > thereshold]) / len(df) * 100)

        print('低于阈值的个数: ', len(df[df < thereshold]),
              len(df[df < thereshold]) / len(df) * 100)

        print(df.describe())

    def stat_BED_info(self, df_1, df_2, cutoff_response):
        convRT_ptv_BED_info_ = df_1['convRT_ptv_BED'].describe()
        convRT_oar_BED_info_ = df_1['convRT_oar_BED'].describe()

        high_survival_adapRT_ptv_BED_info_ = \
            df_1.loc[df_2['conRT_final_response'] <= cutoff_response, 'adapRT_ptv_BED'].describe()
        high_survival_adapRT_oar_BED_info_ = \
            df_1.loc[df_2['conRT_final_response'] <= cutoff_response, 'adapRT_oar_BED'].describe()

        low_survival_adapRT_ptv_BED_info_ = \
            df_1.loc[df_2['conRT_final_response'] > cutoff_response, 'adapRT_ptv_BED'].describe()
        low_survival_adapRT_oar_BED_info_ = \
            df_1.loc[df_2['conRT_final_response'] > cutoff_response, 'adapRT_oar_BED'].describe()

        return convRT_ptv_BED_info_, convRT_oar_BED_info_, \
            high_survival_adapRT_ptv_BED_info_, high_survival_adapRT_oar_BED_info_, \
            low_survival_adapRT_ptv_BED_info_, low_survival_adapRT_oar_BED_info_

    def cumpute_conRT_rsp(self, model_para_path, config_info, patient_id_list=None, is_final_rsp=True):
        df, rsp = self.model_initialization_env2(model_para_path, config_info)
        final_rsp_list = []

        patient_id_list = patient_id_list if patient_id_list is not None else list(df.index)

        for i in patient_id_list:
            _ = rsp.initialization(i, seed=1, alpha_=df.loc[i, 'alpha'], lmd_=df.loc[i, 'lambda'],
                                   PSI_=df.loc[i, 'PSI'], tumor_V0=df.loc[i, 'tumor_V0'])
            conRT_response = rsp.convRT_response.copy()
            if is_final_rsp:
                final_rsp_list.append(conRT_response[-1])
            else:
                final_rsp_list.append(conRT_response)

        return final_rsp_list

    def compute_adapRT_rsp(self, model_para_path, config_info, time_span=41, patient_id_list=None, is_final_rsp=True):
        df, rsp = self.model_initialization_env2(model_para_path, config_info)
        final_rsp_list = compute_adapRT_rsp_(df, rsp, time_span, patient_id_list, is_final_rsp)
        return final_rsp_list

    # NOTE: Plot: Using personal data to plot
    def plot_personal_fraction_scheme(self, patient_idx, img_save_path=None):
        plt.style.use('seaborn-v0_8-deep')

        patient_data = self.df.iloc[patient_idx, :]
        dose_scheme = patient_data[13:].dropna().to_numpy()

        # plot
        fig, ax = plt.subplots(figsize=(14, 6))  # 6.4, 4.8
        ax.bar(np.arange(len(dose_scheme)), dose_scheme, width=0.8, edgecolor="black",
               linewidth=0.7, align='center',
               color='skyblue', alpha=0.7)

        ax.set_xlim([-0.5, len(dose_scheme) - 0.5])
        ax.set_ylim([0, 4.5])

        ax.set_xticks(np.arange(0, dose_scheme.size, 1))
        ax.set_yticks(np.arange(5))

        ax.set_xlabel('Time/day')
        ax.set_ylabel('Fraction Dose/Gy')
        ax.set_title("Dose Fraction Scheme")

        if img_save_path is None:
            plt.show()
        else:
            save_fig(img_save_path, patient_idx=patient_idx, subfolder='dose_scheme')

    # NOTE: Using population data to plot.
    def plot_final_response_comparison_for_all_patients(self, df_, img_save_path=None,
                                                        pic_name='final_response_comparison'):
        cutoff_response = self.config_info['tumor_rt_parameters']['cutoff_response']
        df_['survival_underAdaRT'] = df_['adaRT_final_response'] <= cutoff_response
        df_['survival_underAdaRT'] = df_['survival_underAdaRT'].map({True: 'high', False: 'low'})

        with sns.axes_style("darkgrid"):
            g = sns.JointGrid(data=df_, x="adaRT_final_response", y="conRT_final_response", marginal_ticks=True)

            g.plot_joint(sns.scatterplot, s=100, alpha=.6, hue=df_['survival_underAdaRT'],
                         hue_order=['high', 'low'],
                         style=df_['survival_underAdaRT'], style_order=['high', 'low'])
            g.plot_marginals(sns.histplot, cumulative=False, fill=True, stat='percent', binwidth=5)

            g.refline(x=cutoff_response, y=cutoff_response, color='black', linestyle='dashdot')

        min_val = min(df_['adaRT_final_response'].min(), df_['conRT_final_response'].min()) * 0.9
        max_val = max(df_['adaRT_final_response'].max(), df_['conRT_final_response'].max()) * 1.02

        g.ax_joint.plot([min_val, max_val],
                        [min_val, max_val],
                        color='darkorange', linestyle='--')
        # Add annotation "1" in the bottom right corner
        annotations = ["Ⅰ ", "Ⅱ", "  Ⅲ", "Ⅳ"]
        positions = [(0.3, 0.35), (0.9, 0.35), (0.9, 0.55), (0.3, 0.55)]
        ratio = count_improved_survival_ratio(df_, cutoff_value=cutoff_response)

        for annotation, position, r in zip(annotations, positions, ratio):
            plt.text(position[0], position[1], annotation + ': ' + str(round(r, 1)) + '%',
                     transform=g.ax_joint.transAxes, fontsize=12, ha='right',
                     va='bottom', color='black')
        g.ax_marg_x.axis('on')
        g.ax_marg_y.axis('on')

        g.ax_joint.set_xlim(min_val, max_val)
        g.ax_joint.set_ylim(min_val, max_val)

        g.ax_joint.set_xlabel('Relative GTV volume after adaptive RT /%')
        g.ax_joint.set_ylabel('Relative GTV volume after conventional RT /%')

        g.ax_marg_x.set_ylabel('Percent /%')
        g.ax_marg_y.set_xlabel('Percent /%')

        g.ax_marg_x.yaxis.get_label().set_visible(True)
        g.ax_marg_y.xaxis.get_label().set_visible(True)

        if img_save_path is None:
            plt.show()
        else:
            save_fig(img_save_path, pic_name)

    def plot_cutoff_BED_for_all_patients(self, df, img_save_path=None, pic_name='BED_for_all_patients'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        convRT_BED = [df['convRT_ptv_BED'][0], df['convRT_oar_BED'][0]]

        scatter_colors = [('tab:blue', 'tab:orange'),
                          ('tab:blue', 'tab:orange')]

        scatter_data = [('adapRT_ptv_BED', 'convRT_ptv_BED'),
                        ('adapRT_oar_BED', 'convRT_oar_BED')]

        rsp_data = ['adaRT_final_response', 'conRT_final_response']

        title = ['The BED comparison for GTV', 'The BED comparison for OAR']
        bed_list = [2 * (1 + 0.2) * 30, 0.6 * (1 + 0.06) * 30]  # d * (1 + d / self._ab_tumor)
        j = 0
        for ax, colors, data_pair, bed, t in zip(axes, scatter_colors, scatter_data, convRT_BED, title):
            # ax.axhline(bed, color='black', linestyle='--', linewidth=3)
            ax.axhline(bed_list[int(j)], color='r', linewidth=5)

            marker = ['o', '*']
            for i in range(2):
                ax.scatter(df[rsp_data[j]], df[data_pair[i]],
                           s=20, c=colors[i], alpha=0.6, label=data_pair[i], marker=marker[i])
                # Linear fit line
                # p = np.polyfit(df['conRT_final_response'], df[data_pair[i]], 1)
                # ax.plot(df['conRT_final_response'], p[0] * df['conRT_final_response'] + p[1],
                #         color=colors[i], linewidth=3)

            ax.set_xlim(min(df['conRT_final_response'] * 0.9), max(df['conRT_final_response']) * 1.1)
            ax.set_ylim(bed_list[j] * 0.9, max(df[data_pair[0]]) * 1.1)  # df[data_pair[1]] * 0.9)

            ax.fill_betweenx(y=[ax.get_ylim()[0], ax.get_ylim()[1]],
                             x1=ax.get_xlim()[0],
                             x2=self.config_info['tumor_rt_parameters']['cutoff_response'],
                             color='green', alpha=0.1)
            ax.fill_betweenx(y=[ax.get_ylim()[0], ax.get_ylim()[1]],
                             x2=ax.get_xlim()[1],
                             x1=self.config_info['tumor_rt_parameters']['cutoff_response'],
                             color='red', alpha=0.1)
            ax.set_xlabel('Relative GTV volume after conventional RT /%', fontsize=16)
            ax.set_ylabel('BED /Gy', fontsize=16)
            ax.set_title(t, fontsize=18)
            ax.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=12)

            # 去掉顶部和右侧的边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            j += 1

            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)

        plt.tight_layout()

        if img_save_path is None:
            plt.show()
        else:
            save_fig(img_save_path, pic_name)

    # def plot_dose_scheme_for_all_patients(self, df_, img_save_path=None, pic_name='dose_scheme_for_all_patients',
    #                                       env=2):
    #     cutoff_response = self.config_info['tumor_rt_parameters']['cutoff_response']
    #     if env == 2:
    #         df_long = pd.melt(df_, id_vars=['conRT_final_response', 'patient_id'], value_vars=df_.columns[27:],
    #                           var_name='Fraction', value_name='fraction_dose')
    #     else:
    #         df_long = pd.melt(df_, id_vars=['conRT_final_response', 'patient_id'], value_vars=df_.columns[35:],
    #                           var_name='Fraction', value_name='fraction_dose')
    #
    #     df_long = df_long.dropna(axis=0)
    #     df_long['survival_underConvRT'] = df_long['conRT_final_response'] <= cutoff_response
    #     df_long['survival_underConvRT'] = df_long['survival_underConvRT'].map({True: 'high', False: 'low'})
    #
    #     c_map = np.asarray([[38, 70, 83, 1],
    #                         [42, 157, 143, 1],
    #                         [233, 196, 106, 1],
    #                         [244, 162, 97, 1],
    #                         [231, 111, 81, 1]]) / 255.0
    #
    #     df_workday = df_long[~((df_long['Fraction'] % 7 == 6) | (df_long['Fraction'] % 7 == 0))]
    #
    #     plot_df = [df_workday[df_workday['survival_underConvRT'] == 'high'],
    #                df_workday[df_workday['survival_underConvRT'] == 'low']]
    #     subtitle = ['High Survival Rate Group Under ConvRT', 'Low Survival Rate Group Under ConvRT']
    #     c_map_id = [list(range(len(c_map))), [1, -1]]
    #
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #     for ax, df, st, c in zip(axes, plot_df, subtitle, c_map_id):
    #         g = sns.histplot(df, x='Fraction', discrete=True, multiple='stack',
    #                          hue='fraction_dose', ax=ax)  # palette=c_map,
    #         ax.set_title(st, fontsize=16)
    #         ax.set_xlim(df['Fraction'].min() - 1, df['Fraction'].max() + 1)
    #         ax.set_xticks(ticks=list(range(df['Fraction'].min(), df['Fraction'].max() + 1, 2)),
    #                       labels=list(range(df['Fraction'].min(), df['Fraction'].max() + 1, 2)),
    #                       fontsize=12, rotation=45)
    #         # ax.set_yticks(font_size=12)
    #         g.set_xlabel('Fraction', fontsize=14), g.set_ylabel('Count', fontsize=14)
    #     plt.tight_layout()
    #     sns.despine()
    #
    #     if img_save_path is None:
    #         plt.show()
    #     else:
    #         save_fig(img_save_path, pic_name)

    def plot(self, img_save_path):
        self.model_initialization()

        # model para analysis.
        # self.plot_different_PSI_for_response(img_save_path=img_save_path)
        # self.plot_different_alpha_lmd_for_response(img_save_path=img_save_path)

        # patient specific response and dose scheme analysis.
        # for i in np.arange(0, 1000, 50):
        #     self.plot_personal_response(i, img_save_path)
        #     self.plot_personal_fraction_scheme(i, img_save_path)

        # population analysis
        # self.plot_final_response_comparison_for_all_patients(self.df.copy(deep=True), img_save_path)
        self.plot_cutoff_BED_for_all_patients(self.df.copy(deep=True), img_save_path)
        # self.plot_dose_scheme_for_all_patients(self.df.copy(deep=True), img_save_path)

    def different_rsp_ratio_comparison(self, img_save_path, suffix):
        # Compare good / bad response rate for Lung / NH patients.
        final_rsp_HN = np.asarray(
            self.cumpute_conRT_rsp(self.path_dict['HN_model_para'], self.config_info_dict['HN'],
                                   is_final_rsp=True))

        final_rsp_Lung = np.asarray(
            self.cumpute_conRT_rsp(self.path_dict['Lung_model_para'], self.config_info_dict['Lung'],
                                   is_final_rsp=True))

        cutoff_rsp = self.config_info_dict['HN']['tumor_rt_parameters']['cutoff_response']
        df = pd.DataFrame({'final_rsp': final_rsp_HN, 'type_of_cancer': 'HN', 'prognosis': final_rsp_HN <= cutoff_rsp})

        cutoff_rsp = self.config_info_dict['Lung']['tumor_rt_parameters']['cutoff_response']
        df2 = pd.DataFrame({'final_rsp': final_rsp_Lung, 'type_of_cancer': 'Lung',
                            'prognosis': final_rsp_Lung <= cutoff_rsp})
        df = pd.concat([df, df2], ignore_index=True)

        df['prognosis'] = df['prognosis'].map({True: 'good', False: 'bad'})

        # 设置 seaborn 的绘图风格
        with sns.axes_style("white"):
            ax = sns.histplot(data=df, x="prognosis", hue="type_of_cancer", common_norm=False,
                              multiple='dodge', shrink=0.8, binwidth=15, stat='percent', legend=True)

            for p in ax.patches:
                ax.text(x=p.get_x() + p.get_width() / 2,
                        y=p.get_height() + 0.5,
                        s=f'{p.get_height() / 100:.2%}',
                        ha='center')

                ax.set_ylabel('Percentage of Patients (%)')
                ax.set_xlabel('Prognosis')

        if img_save_path is not None:
            save_fig(img_save_path, suffix)
        else:
            plt.show()

    def final_BED_comparison(self, df_):
        cutoff_response = self.config_info['tumor_rt_parameters']['cutoff_response']

        df_long = pd.melt(df_, id_vars=['adapRT_ptv_BED'],
                          value_vars=['adaRT_final_response'],
                          var_name='plan_type', value_name='final_response')
        df_long['survival_rate'] = df_long['final_response'] <= cutoff_response
        df_long['survival_rate'] = df_long['survival_rate'].map({True: 'high', False: 'low'})

        df_long = pd.melt(df_long, id_vars=['plan_type', 'final_response', 'survival_rate'],
                          value_vars=['adapRT_ptv_BED'],
                          var_name='Plan Type', value_name='PTV BED')

        g = sns.stripplot(data=df_long, y="PTV BED", x="survival_rate", hue="survival_rate",
                          dodge=False, size=3, alpha=0.8, jitter=0.1)
        g.axhline(df_['convRT_ptv_BED'][0], color='r', linewidth=1, linestyle='--')
        plt.show()

        df_long = pd.melt(df_, id_vars=['adapRT_oar_BED'],
                          value_vars=['adaRT_final_response'],
                          var_name='plan_type', value_name='final_response')
        df_long['survival_rate'] = df_long['final_response'] <= cutoff_response
        df_long['survival_rate'] = df_long['survival_rate'].map({True: 'high', False: 'low'})

        df_long = pd.melt(df_long, id_vars=['plan_type', 'final_response', 'survival_rate'],
                          value_vars=['adapRT_oar_BED'],
                          var_name='Plan Type', value_name='OAR BED')

        g = sns.stripplot(data=df_long, y="OAR BED", x="survival_rate", hue="survival_rate",
                          dodge=False, size=3, alpha=0.8, jitter=0.1)
        g.axhline(df_['convRT_oar_BED'][0], color='r', linewidth=1, linestyle='--')

        plt.show()

    def stem_plot_final_response(self, frac_path, config_info, mode=None, img_save_path=None, suffix=None):
        df, rsp = self.model_initialization_env2(frac_path, config_info)

        plot_idx = list(range(0, len(df.index), 20))

        patient_idx = np.array(list(df.index))
        conRT_final_rsp = df['conRT_final_response']
        adaRT_final_rsp = df['adaRT_final_response']
        cutoff_rsp = config_info['tumor_rt_parameters']['cutoff_response']

        fig, ax1 = plt.subplots()

        if mode is None:
            ax1.stem(patient_idx[plot_idx], conRT_final_rsp.iloc[plot_idx],
                     linefmt='#7494BC', markerfmt='o',
                     basefmt='k-', label='Conventional RT')
            ax1.set_xlabel('Patient index')  # 设置x轴标签
            ax1.set_ylabel('Relative tumor volume after RT')  # 设置y轴标签

            ax1.stem(patient_idx[plot_idx], -1 * adaRT_final_rsp[plot_idx],
                     linefmt='#DE9F83', markerfmt='*',
                     basefmt='k-', label='Adaptive RT')

            plt.legend(['Conventional RT', 'Adaptive RT'], loc='upper right', bbox_to_anchor=(1, 1.1))

            self.print_data_description(adaRT_final_rsp, thereshold=cutoff_rsp,
                                        info='ART final response info')

            max_rsp = conRT_final_rsp.to_numpy().max() * 1.1
            # y_ticks_label_1 = [str(i) for i in np.arange(0, max_rsp+1, 20)]
            # y_ticks_label_1.reverse()
            # y_ticks_label_2 = [str(i) for i in np.arange(0, max_rsp+1, 20)]

            ax1.set_xlim([-100, len(df.index) + 100])
            ax1.set_ylim([-max_rsp, max_rsp + 1])
            # ax1.set_yticks(np.arange(-max_rsp, max_rsp+1, 20), y_ticks_label_1[:-1] + y_ticks_label_2)
            ax1.axhline(y=cutoff_rsp, color='black', linestyle='--', linewidth=1)
            ax1.axhline(y=-1 * cutoff_rsp, color='black', linestyle='--', linewidth=1)

            # 去除边框
            ax1.spines['top'].set_visible(False), ax1.spines['right'].set_visible(False)

        else:
            diff = conRT_final_rsp - adaRT_final_rsp
            ax1.stem(patient_idx[plot_idx], diff.iloc[plot_idx],
                     linefmt='#7494BC', markerfmt='o',
                     basefmt='k-')  # , label='Conventional RT'
            ax1.set_xlabel('Patient index')
            ax1.set_ylabel(r'Relative tumor volume Difference after RT:($V_{conRT}$-$V_{adaRT}$)/V0')

        if img_save_path is not None:
            save_fig(img_save_path, suffix)
        else:
            plt.show()

    def hist_plot_final_rsp_ratio(self, frac_path, config_info, img_save_path=None, suffix=None):
        df, rsp = self.model_initialization_env2(frac_path, config_info)

        conRT_final_rsp = df['conRT_final_response']
        adaRT_final_rsp = df['adaRT_final_response']
        cutoff_rsp = config_info['tumor_rt_parameters']['cutoff_response']

        df_plot = pd.DataFrame({'final_rsp': conRT_final_rsp, 'PlanType': 'conventionalRT',
                                'prognosis': conRT_final_rsp <= cutoff_rsp})

        df_plot_2 = pd.DataFrame({'final_rsp': adaRT_final_rsp, 'PlanType': 'adaptiveRT',
                                  'prognosis': adaRT_final_rsp <= cutoff_rsp})

        df_plot = pd.concat([df_plot, df_plot_2], ignore_index=True)

        df_plot['prognosis'] = df_plot['prognosis'].map({True: 'good', False: 'bad'})

        with sns.axes_style("white"):
            ax = sns.histplot(data=df_plot, x="prognosis", hue="PlanType", common_norm=False,
                              multiple='dodge', shrink=0.8, binwidth=15, stat='percent', legend=True)

            for p in ax.patches:
                ax.text(x=p.get_x() + p.get_width() / 2,
                        y=p.get_height() + 0.5,
                        s=f'{p.get_height() / 100:.2%}',
                        ha='center')

                ax.set_ylabel('Percentage of Patients (%)')
                ax.set_xlabel('Prognosis')

        if img_save_path is None:
            plt.show()
        else:
            save_fig(img_save_path, suffix)

    def stem_plot_final_BED(self, frac_path, config_info, img_save_path=None, suffix='3d_scatter'):
        df, rsp = self.model_initialization_env2(frac_path, config_info)

        plot_idx = list(range(0, len(df.index), 20))

        patient_idx = np.array(list(df.index))
        conRT_ptv_BED, conRT_oar_BED = df['convRT_ptv_BED'], df['convRT_oar_BED']
        adaRT_ptv_BED, adaRT_oar_BED = df['adapRT_ptv_BED'], df['adapRT_oar_BED']

        print("conRT_oar_BED:", conRT_oar_BED.describe())
        print("adaRT_oar_BED:", adaRT_oar_BED.describe())

        # 对df.loc[:, range(1, 41)]求行和
        df['total_dose'] = df[range(1, 41)].apply(lambda x: x.sum(), axis=1)
        print("total dose:", df['total_dose'].describe())

        # fig, axes = plt.subplots(2, 1)
        #
        # # diff = conRT_ptv_BED.iloc[plot_idx] - adaRT_ptv_BED.iloc[plot_idx]
        # axes[0].stem(patient_idx[plot_idx], df['total_dose'][plot_idx], linefmt='#4E6691', markerfmt='o',
        #              basefmt='-', label='PTV')
        # axes[0].set_xlabel('Patient index')
        # axes[0].set_ylabel(r'Total Dose /Gy')
        # axes[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.25))
        #
        # diff = conRT_oar_BED.iloc[plot_idx] - adaRT_oar_BED.iloc[plot_idx]
        # axes[1].stem(patient_idx[plot_idx], diff, linefmt='#B8474D', markerfmt='*',
        #              basefmt='-', label='OAR')
        # axes[1].set_xlabel('Patient index')
        # axes[1].set_ylabel(r'BED$_{conRT}$ - BED$_{perRT}$ /Gy')
        # axes[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.25))

        self.print_data_description(conRT_ptv_BED - adaRT_ptv_BED, thereshold=0,
                                    info='Delta PTV BED info')
        self.print_data_description(conRT_oar_BED - adaRT_oar_BED, thereshold=0,
                                    info='Delta OAR BED info')

        # for ax in axes:
        #     ax.set_xlim([-100, len(df.index) + 100])
        #     ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        #
        # fig.align_ylabels(axes)
        # fig.tight_layout()
        # if img_save_path is not None:
        #     save_fig(img_save_path, suffix)
        # else:
        #     plt.show()

    def scatter_3d_patient_distribution(self, frac_path, config_info, img_save_path=None, suffix=None):
        df, rsp = self.model_initialization_env2(frac_path, config_info)

        bed_df = pd.DataFrame({'oar_bed': df['adapRT_oar_BED'],
                               'alpha': df['alpha'],
                               'lambda': df['lambda'],
                               "PSI": df['PSI']})

        # Getting descriptive statistics of 'oar_bed'
        oar_bed_stats = bed_df['oar_bed'].describe()

        # Categorizing 'oar_bed' into 'High', 'Medium', and 'Low' based on quartiles
        conditions = [
            (bed_df['oar_bed'] <= oar_bed_stats['25%']).to_numpy(),  # High-benefit
            (bed_df['oar_bed'] > oar_bed_stats['75%']).to_numpy()]  # Low-benefit

        # Define choices corresponding to conditions
        choices = ['High', 'Low']

        # Use np.select to apply conditions and choices to the new 'Type' column
        bed_df['Type'] = np.select(conditions, choices)

        color_map = {'High': '#f16c23', 'Low': '#1b7c3d'}

        # Create the first figure for the 3D scatter plot
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        for category, color in color_map.items():
            idx = bed_df['Type'] == category
            ax_3d.scatter(bed_df.loc[idx, 'alpha'], bed_df.loc[idx, 'lambda'], bed_df.loc[idx, 'PSI'],
                          c=color, alpha=0.3, label=category + '-benefit group')
        ax_3d.set_title("The Distribution of Patients with Different Dose Benefits \n in the Parameter Space",
                        fontsize=16, fontweight='medium')

        ax_3d.set_xlabel(r'$\alpha   /Gy^{-1}$', fontsize=14, fontweight='medium')
        ax_3d.set_ylabel(r'$\lambda   /day^{-1}$', fontsize=14, fontweight='medium')
        ax_3d.set_zlabel('PSI', fontsize=14, fontweight='medium')
        ax_3d.view_init(30, 30)
        ax_3d.legend(fontsize=14)

        plt.tight_layout()

        # Save or show the 3D figure
        if img_save_path is not None:
            save_fig(img_save_path,  suffix)
        else:
            plt.show()

    def SVM_results_5fold_val(self, svm_cls, X_scaled, y):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        for train_index, test_index in kf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            svm_cls.fit(X_train, y_train)

            y_pred = svm_cls.predict(X_test)
            trasfer_dict = {"High": 1, "Low": 0}
            y_pred_t = [trasfer_dict[i] for i in y_pred]
            y_true = [trasfer_dict[i] for i in y_test]

            accuracy_list.append(accuracy_score(y_true, y_pred_t))
            precision_list.append(precision_score(y_true, y_pred_t))
            recall_list.append(recall_score(y_true, y_pred_t))
            f1_list.append(f1_score(y_true, y_pred_t))

        print("Average Accuracy: {:.2f} ± {:.2f}".format(np.mean(accuracy_list), np.std(accuracy_list)))
        print("Average Precision: {:.2f} ± {:.2f}".format(np.mean(precision_list), np.std(precision_list)))
        print("Average Recall: {:.2f} ± {:.2f}".format(np.mean(recall_list), np.std(recall_list)))
        print("Average F1-Score: {:.2f} ± {:.2f}".format(np.mean(f1_list), np.std(f1_list)))

    def SVM_2D(self, frac_path, config_info, fixed_feature_name, fixed_feature_value_list,
               other_feature_names, c_list, c_sample_weight_list, atol, x_label, y_label, kernel='linear',
               img_save_path=None, suffix=None):
        df, rsp = self.model_initialization_env2(frac_path, config_info)

        bed_df = pd.DataFrame({'oar_bed': df['adapRT_oar_BED'],
                               'alpha': df['alpha'],
                               'lambda': df['lambda'],
                               "PSI": df['PSI']})

        # Getting descriptive statistics of 'oar_bed'
        oar_bed_stats = bed_df['oar_bed'].describe()

        # Categorizing 'oar_bed' into 'High', 'Medium', and 'Low' based on quartiles
        conditions = [
            (bed_df['oar_bed'] <= oar_bed_stats['25%']).to_numpy(),  # High-benefit
            (bed_df['oar_bed'] > oar_bed_stats['75%']).to_numpy()]  # Low-benefit

        # Define choices corresponding to conditions
        choices = ['High', 'Low']

        # Use np.select to apply conditions and choices to the new 'Type' column
        bed_df['Type'] = np.select(conditions, choices)

        bed_df = bed_df[bed_df['Type'] != '0']

        color_map = {'High': '#f16c23', 'Low': '#1b7c3d'}

        # Create a figure for the cross-section plots
        fig_cs, axs_cs = plt.subplots(1, 3, figsize=(18, 6))

        # Values for cross-sections
        # psi_values = [0.65, 0.75, 0.85]
        # c = [50, 10, 5]

        # Train and plot SVM for each cross-section
        for i, fix_feature_value in enumerate(fixed_feature_value_list):
            ax = axs_cs[i]
            subset = bed_df[np.isclose(bed_df[fixed_feature_name], fix_feature_value, atol=atol)]
            features = subset[other_feature_names].values
            scaler = StandardScaler()
            feature_scaled = scaler.fit_transform(features)
            labels = subset['Type']

            # Fit the SVM model
            clf = svm.SVC(C=c_list[i], class_weight='balanced', kernel=kernel, degree=3, probability=True)  # LinearSVC
            #  scaler = StandardScaler()
            # features_scaled = scaler.fit_transform(features)
            sample_weight = np.ones_like(labels)
            sample_weight[labels == 'Low'] = c_sample_weight_list[i]

            clf.fit(feature_scaled, labels, sample_weight=sample_weight)

            # 5-Fold cross validation
            print(f"{fixed_feature_name} {fix_feature_value} 5-fold cross validation: ...")
            self.SVM_results_5fold_val(clf, feature_scaled, labels)

            # Plot
            x_min, x_max = subset[other_feature_names[0]].min(), subset[other_feature_names[0]].max()
            y_min, y_max = subset[other_feature_names[1]].min(), subset[other_feature_names[1]].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

            new_x = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.predict(scaler.transform(new_x)).reshape(xx.shape)
            Z = np.where(Z == 'High', 1, 0)

            ax.contourf(xx, yy, Z, alpha=0.1, cmap=plt.cm.coolwarm)

            # Plot the scatter points
            for category, color in color_map.items():
                idx = subset['Type'] == category
                ax.scatter(subset.loc[idx, other_feature_names[0]], subset.loc[idx, other_feature_names[1]],
                           c=color, alpha=0.5,
                           label=f'{category}-benefit')

                # Use DecisionBoundaryDisplay to plot the decision boundaries
                # DecisionBoundaryDisplay.from_estimator(
                #     clf,
                #     features,
                #     plot_method="contour",
                #     response_method="predict",
                #     alpha=0.5,
                #     eps=0.05,
                #     ax=ax,
                #     levels=[-1, 1],
                #     grid_resolution=500,
                #     colors="k",
                #     linestyles=["-"],
                #     xlabel=x_label,
                #     ylabel=y_label,
                # )
                # 获取决策边界的系数和截距
                w = clf.coef_[0]  # 线性SVM的系数
                b = clf.intercept_[0]  # 截距

                print(f"The decision boundary equation is: {w[0]:.3f} * x1 + {w[1]:.3f} * x2 + {b:.3f} = 0")

                ax.set_title(f'{fixed_feature_name}={fix_feature_value}', fontsize=16, fontweight='medium')
                ax.set_xlabel(x_label, fontsize=14, fontweight='medium') # x_label r'$\alpha  /Gy^{-1}$'
                ax.set_ylabel(y_label, fontsize=14, fontweight='medium')  # '$\lambda  /day^{-1}$

                ax.tick_params(axis='both', which='major', labelsize=14)
                if i == 0:  # Only add legend to the first subplot for clarity
                    ax.legend(fontsize=14, loc='lower right')

                ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save or show the cross-section figure
        if img_save_path is not None:
            save_fig(img_save_path, suffix)
        else:
            plt.show()

    def relation_plot_BED_33(self, frac_path, config_info, img_save_path=None):
        # Values for cross-sections
        psi_values = [0.65, 0.75, 0.85]
        alpha_values = [0.06, 0.09, 0.12]
        lambda_values = [0.05, 0.07, 0.09]

        print("PSI................")
        self.SVM_2D(frac_path, config_info, 'PSI', [0.65, 0.75, 0.85],
                    ['alpha', 'lambda'], [1, 1, 1], [1, 1, 1],
                    0.03,
                    r'$\alpha /Gy^{-1}$',
                    r'$\lambda /day^{-1}$',
                    img_save_path=img_save_path, suffix='SVM_PSI')

        print("ALPHA................")
        # fix alpha
        self.SVM_2D(frac_path, config_info, 'alpha', [0.06, 0.09, 0.12],
                    ['lambda', 'PSI'], [1, 1, 1], [1, 1, 1],
                    0.01,
                    r'$\lambda /day^{-1}$',
                    'PSI',
                    kernel='linear',
                    img_save_path=img_save_path, suffix="SVM_alpha")

        print("lambda................")
        self.SVM_2D(frac_path, config_info, 'lambda', [0.05, 0.07, 0.09],
                    ['alpha', 'PSI'], [1, 1, 1], [1, 1, 1],
                    0.01,
                    r'$\alpha /Gy^{-1}$',
                    'PSI',
                    kernel='linear',
                    img_save_path=img_save_path, suffix="SVM_lambda")

    def line_plot_response(self, frac_path, config_info, time_span=41, img_save_path=None, suffix=None):
        # sns.set_style("whitegrid")
        with sns.axes_style("whitegrid"):
            # conRT response
            conRT_final_rsp_list = self.cumpute_conRT_rsp(frac_path, config_info, is_final_rsp=False)
            conRT_final_rsp_list = np.asarray(conRT_final_rsp_list)

            adaRT_final_rsp_list = self.compute_adapRT_rsp(frac_path, config_info, time_span=time_span,
                                                           is_final_rsp=False)
            adaRT_final_rsp_list = np.asarray(adaRT_final_rsp_list)

            df = pd.DataFrame(adaRT_final_rsp_list, columns=list(range(adaRT_final_rsp_list.shape[-1])))
            df['Type'] = 'Personalized de-escalation RT'

            df_2 = pd.DataFrame(conRT_final_rsp_list, columns=list(range(conRT_final_rsp_list.shape[-1])))
            df_2['Type'] = 'Conventional RT'
            df = pd.concat([df, df_2], axis=0)

            df = df.melt(id_vars='Type', value_vars=list(range(conRT_final_rsp_list.shape[-1])),
                         var_name='Time', value_name='Response')

            fig, ax = plt.subplots(figsize=(8, 6))

            ax = sns.lineplot(data=df, x="Time", y="Response", hue="Type", style="Type",
                              errorbar='sd', markers=True, palette=['#4E6691', '#B8474D'],
                              legend='auto', ax=ax)
            ax.legend(loc='lower left', fontsize=14)

            plt.axhline(y=config_info['tumor_rt_parameters']['cutoff_response'],
                        color='black', alpha=0.6, linestyle='--')

            ax.set_xlabel('Time /day', fontsize=14, fontweight='medium')  # 设置x轴标签
            ax.set_ylabel(r'Relative Tumor Volume /%', fontsize=14, fontweight='medium')

            ax.set_xlim([0, conRT_final_rsp_list.shape[-1] - 1])
            ax.set_ylim([0, 100])

            if img_save_path is not None:
                save_fig(img_save_path, suffix)
            else:
                plt.show()


    def hist_plot_dose_scheme(self, frac_path, config_info, time_span=41, img_save_path=None, suffix=None):
        df, rsp = self.model_initialization_env2(frac_path, config_info)

        dose_list = []
        for i in list(df.index):
            dose_list.append(df.loc[i, list(range(1, time_span, 1))].dropna().to_numpy())

        dose_list = np.array(dose_list)
        df = pd.DataFrame(dose_list, columns=list(range(1, time_span, 1)))

        df = df.melt(value_vars=list(range(1, time_span, 1)),
                     var_name='Time', value_name='Fraction Dose')

        df_workday = df[~((df['Time'] % 7 == 6) | (df['Time'] % 7 == 0))]
        fig, ax = plt.subplots(figsize=(8, 6))

        with sns.axes_style("ticks"):
            ax = sns.histplot(df_workday, x='Time', discrete=True, multiple='stack',
                              hue='Fraction Dose',
                              palette=['#F5CF36', '#B8474D',  '#8679be', '#8386A8', '#1c3e71',  '#78B9D2', '#8FB943', ], #
                              ax=ax)

            # ax.set_title(st, fontsize=16)
            ax.set_xlim(0, time_span)
            # ax.set_xticks(ticks=list(range(df['Fraction'].min(), df['Fraction'].max() + 1, 2)),
            #               labels=list(range(df['Fraction'].min(), df['Fraction'].max() + 1, 2)),
            #               fontsize=12, rotation=45)
            # # ax.set_yticks(font_size=12)
            ax.set_xlabel('Time /day', fontsize=14, fontweight='medium')
            ax.set_ylabel('Number of Patients', fontsize=14, fontweight='medium')

        sns.move_legend(ax, "lower left", ncol=2)

        plt.tight_layout()

        sns.despine()

        if img_save_path is None:
            plt.show()
        else:
            save_fig(img_save_path, suffix)

    def line_plot_personal_response_and_scheme(self, frac_path, config_info, patient_id_list,
                                               time_span=41, img_save_path=None, suffix=None):
        df, rsp = self.model_initialization_env2(frac_path, config_info)

        conRT_final_rsp_list = self.cumpute_conRT_rsp(frac_path, config_info, patient_id_list=patient_id_list,
                                                      is_final_rsp=False)
        conRT_final_rsp_list = np.asarray(conRT_final_rsp_list)

        adaRT_final_rsp_list = self.compute_adapRT_rsp(frac_path, config_info, patient_id_list=patient_id_list,
                                                       is_final_rsp=False)
        adaRT_final_rsp_list = np.asarray(adaRT_final_rsp_list)

        for i, p_idx in enumerate(patient_id_list):
            fig, ax = plt.subplots(figsize=(8, 6))

            ax.plot(list(range(conRT_final_rsp_list[i].size)), conRT_final_rsp_list[i],
                    c='#4292C6', linestyle='-', marker='x', label='Conventional RT', zorder=2)
            ax.plot(list(range(adaRT_final_rsp_list[i].size)), adaRT_final_rsp_list[i],
                    c='#B95A58', linestyle='--', marker='o', label='Adaptive RT', zorder=2)

            ax.set_xlabel(r'Time /day', fontsize=14, fontweight='medium')  # 设置x轴标签
            ax.set_ylabel(r'Relative Tumor Volume /%', fontsize=14, fontweight='medium')

            ax.set_xlim([0, conRT_final_rsp_list.shape[-1]])
            ax.set_ylim([0, 100])
            ax.legend()
            ax.axhline(config_info['tumor_rt_parameters']['cutoff_response'],
                       color='black', alpha=0.6, linestyle='--')
            # 建立双轴
            ax2 = ax.twinx()

            dose_list = df.loc[p_idx, list(range(0, time_span, 1))].dropna().to_numpy()

            # 柱状图
            ax2.bar(list(range(conRT_final_rsp_list[i].size)), dose_list,
                    color='#4a6d42', alpha=0.5, zorder=1)

            ax2.set_ylim([0, np.amax(dose_list) + 0.3])
            # 设置y轴tick label
            ax2.set_yticks(list(range(0, np.amax(dose_list).astype(int) + 1, 1)))
            ax2.set_ylabel(r'Fraction Dose /Gy', fontsize=14, fontweight='medium')

            if img_save_path is not None:
                save_fig(img_save_path, suffix=suffix, patient_idx=p_idx,
                         subfolder='personal_rsp_and_scheme')
            else:
                plt.show()

    def noise_plot(self, config_info, noise_path, save_path):
        # read csv data and write excel.
        data_list = glob.glob(noise_path + '/*.xlsx')
        oar_BED_dict, adaRT_final_response_dict, conRT_final_response_dict, frac_scheme_dict = \
            {}, {}, {}, {}
        df_dict = {}
        for c in data_list:
            df = read_fraction_scheme_and_write(Path(c), save_path=noise_path)
            oar_BED_dict[c[-7:-5]] = df['adapRT_oar_BED']
            adaRT_final_response_dict[c[-7:-5]] = df['adaRT_final_response']
            conRT_final_response_dict[c[-7:-5]] = df['conRT_final_response']
            frac_scheme_dict[c[-7:-5]] = df[range(15, 41, 1)]
            df_dict[c[-7:-5]] = df

        # noise_plot.kde_plot_noise_BED_diff(oar_BED_dict, img_save_path=None, suffix='Revised_noise_BED_KDE')
        noise_plot.hist_plot_noise_prognosis(config_info, df_dict, threshold=76.0,
                                             img_save_path=None,
                                             suffix='Revised_noise_hist_prognosis')

    def plot_HN(self, path_dict, config_info_dict, time_span=41):
        print("Plot for HN patients...")
        # model.stem_plot_final_response(path_dict['frac_path_HN'], config_info_dict['HN'],
        #                                img_save_path=None, suffix='HN_stem_plot_final_response')
        # stat_final_rsp(path_dict['frac_path_HN'], config_info_dict['HN'])

        # model.stem_plot_final_BED(path_dict['frac_path_HN'], config_info_dict['HN'],
        #                           img_save_path=None, suffix='HN_stem_plot_final_BED')
        #
        # model.line_plot_response(path_dict['frac_path_HN'], config_info_dict['HN'],
        #                          img_save_path=img_save_path, suffix='ab4_vol_line_plot')
        #
        # model.hist_plot_dose_scheme(path_dict['frac_path_HN'], config_info_dict['HN'], time_span=time_span,
        #                             img_save_path=img_save_path, suffix='ab4_hist_plot_dose_scheme')

        # model.line_plot_personal_response_and_scheme(path_dict['frac_path_HN'], config_info_dict['HN'],
        #                                          list(range(0, 5000, 50)), time_span=time_span,
        #                                          img_save_path=img_save_path, suffix='HN')

        # model.relation_plot_BED(path_dict['frac_path_HN'], config_info_dict['HN'],
        #                         img_save_path=img_save_path, suffix='relation_13_')
        #
        model.relation_plot_BED_33(path_dict['frac_path_HN'], config_info_dict['HN'],
                                img_save_path=None)

    def reverse_ART_dose_scheme(self, data_path, save_path, config_info):
        df, rsp = self.model_initialization_env2(data_path, config_info)
        df[list(range(15, 41, 1))] = df[list(range(15, 41, 1))].apply(lambda x: x[::-1], axis=1)
        final_rsp_list = compute_adapRT_rsp_(df, rsp, time_span=41, patient_id_list=None, is_final_rsp=True)
        df['adaRT_final_response'] = final_rsp_list

        df.to_excel(save_path,  index=False)



if __name__ == "__main__":
    config_path_HN = Path(r"../Utils/config_env_HN.yml")  # config_env2_HN

    config_info_HN = get_config_info(config_path_HN)

    model_para_path_HN = Path(r"../Data/HN_test_patients.xlsx")

    frac_path_HN = Path(r"../Data/frac_log/ppo_env2/v2_train_ppo_18o_D6a_HN_LRC_60Gy_de-escalation.csv")
    # ../Data/frac_log/ppo_env2/v2_train_ppo_18o_D6a_HN_LRC_60Gy_de-escalation.csv

    img_save_path = Path(r"../Result/env2/test_for_revision")

    noise_path_HN = r"../Data/frac_log/ppo_env2/new_noise"

    if not img_save_path.exists():
        img_save_path.mkdir(parents=True)

    path_dict = {'HN_model_para': model_para_path_HN,
                 'frac_path_HN': frac_path_HN,
                 'save_path': img_save_path,
                 'noise_path_HN': noise_path_HN}

    config_info_dict = {'HN': config_info_HN, 'Lung': None}

    model = ARTResponsePlotEnv2(path_dict, config_info_dict)

    # model.different_rsp_ratio_comparison(path_dict['save_path'], 'different_rsp_ratio_comparison_v70')
    # model.plot_HN(path_dict, config_info_dict)

    model.noise_plot(config_info_dict['HN'], path_dict['noise_path_HN'], path_dict['save_path'])


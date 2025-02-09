import numpy as np
import matplotlib.pyplot as plt
import math
import pathlib
from pathlib import Path
from Utils.readAndWrite.read_config_file import get_config_info
import pandas as pd


def exp_model(v0, tau, t):
    v_new = v0 * math.exp(math.log(2) / tau * t)
    return v_new


def gomp_model(v0, b, v_max, t):
    v_new = math.pow(v0, math.exp(-1 * b * t)) * math.pow(v_max, 1 - math.exp(-1 * b * t))
    return v_new


def log_gomp_vol_withRT_(length, v0, b, v_max, alpha, beta, d_list, log_vol_list):
    len_ = length - 1

    if len_ == 0:
        log_v_new = math.exp(-1 * b) * math.log(v0) + (1 - math.exp(-1 * b)) * math.log(v_max) - (
                alpha * d_list[len_] + beta * d_list[len_] ** 2)
        log_vol_list.append(log_v_new)
        return log_v_new
    else:
        log_v_old = log_gomp_vol_withRT_(len_, v0, b, v_max, alpha, beta, d_list, log_vol_list)
        log_v_new = math.exp(-1 * b) * log_v_old + (1 - math.exp(-1 * b)) * math.log(v_max) - (
                alpha * d_list[len_] + beta * d_list[len_] ** 2)
        log_vol_list.append(log_v_new)
        return log_v_new


def plot_tumor_growth(v0, tau, b, v_max, t_max):
    exp_vol_list = [exp_model(v0, tau, t) for t in range(t_max + 1)]
    gomp_vol_list = [gomp_model(v0, b, v_max, t) for t in range(t_max + 1)]

    fig, ax = plt.subplots()
    x = np.arange(0, t_max + 1, 1)
    ax.plot(x, exp_vol_list, label='exponential_model')
    ax.plot(x, gomp_vol_list, label='gompertzian_model')
    ax.set_xlabel('time (day)')
    ax.set_ylabel('Volume (cm3)')
    ax.set_title("Tumor growth in different model (Slow)")
    ax.legend()

    ax.text(0, 410, 'tau=50, b=exp(-6), v_max=5*v0')

    plt.show()


class ImgPloter:
    def __init__(self, config_path):
        self.config_info = get_config_info(config_path)
        self.alpha_tumor = None
        self.ab_tumor = None
        self.ab_oar = None
        self.tau = None
        self.b = None
        self.max_vol_ratio = None
        self.total_dose = None
        self.crt_fraction_dose = None
        self.growth_mode = None

    def init_para_with_config_info(self):
        self.alpha_tumor = self.config_info['tumor_rt_parameters']['alpha_tumor']
        self.ab_tumor = self.config_info['tumor_rt_parameters']['ab_tumor']
        self.ab_oar = self.config_info['tumor_rt_parameters']['ab_oar']
        self.tau = self.config_info['tumor_rt_parameters']['tau']
        self.b = self.config_info['tumor_rt_parameters']['b']
        self.max_vol_ratio = self.config_info['tumor_rt_parameters']['max_vol_ratio']
        self.total_dose = self.config_info['tumor_rt_parameters']['total_dose']
        self.crt_fraction_dose = self.config_info['tumor_rt_parameters']['crt_fraction_dose']
        self.growth_mode = self.config_info['tumor_rt_parameters']['growth_mode']

    def plot_tumor_growth(self, v0=None, tau=None, b=None, v_max=None, t_max=None):
        if v0 is None:
            v0 = 300
            tau = 50
            t_max = 30
            b = math.exp(-6)
            v_max = 5 * v0

        exp_vol_list = [exp_model(v0, tau, t) for t in range(t_max + 1)]
        gomp_vol_list = [gomp_model(v0, b, v_max, t) for t in range(t_max + 1)]

        fig, ax = plt.subplots()
        x = np.arange(0, t_max + 1, 1)
        ax.plot(x, exp_vol_list, label='exponential_model')
        ax.plot(x, gomp_vol_list, label='gompertzian_model')
        ax.set_xlabel('time (day)')
        ax.set_ylabel('Volume (cm3)')
        ax.set_title("Tumor growth in different model (Slow)")
        ax.legend()

        ax.text(0, 410, 'tau=50, b=exp(-6), v_max=5*v0')

        plt.show()

    def daily_dose_plot(self, df_frac, title, show=False):
        # plot fraction scheme
        plt.style.use('seaborn-v0_8-deep')

        df = df_frac.iloc[0, :]

        y = df.to_numpy()
        x = list(range(y.size))

        # plot
        fig, ax = plt.subplots(figsize=(14, 4.8))  # 6.4, 4.8
        ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7, align='center')

        ax.set_xlim([0, len(x) + 1])
        ax.set_ylim([0, 4.5])

        ax.set_xticks(np.arange(0, len(x), 1))
        ax.set_yticks(np.arange(0, 4, 1))

        ax.set_xlabel('time/day')
        ax.set_ylabel('fraction dose/Gy')
        ax.set_title(title)

        if show:
            plt.show()

        return ax

    def growth_rate_plot(self, ax, df_frac, df_patient, patient_idx, mode='gom'):
        ax2 = ax.twinx()
        # growth_rate_plot
        patient_data = df_patient.iloc[patient_idx, :]
        tumor_V0 = patient_data['tumor_V0']
        frac_data = df_frac.iloc[0, :]
        frac_data = frac_data.to_numpy()

        if mode == 'exp':
            growth_rate = math.log(2) / self.tau
            gr_list = np.ones(frac_data.size) * growth_rate
        elif mode == 'gom':
            log_vol_list = [math.log(tumor_V0), ]
            v_max = self.max_vol_ratio * tumor_V0
            beta = self.alpha_tumor / self.ab_tumor
            log_gomp_vol_withRT_(len(frac_data)-1, tumor_V0, self.b, v_max, self.alpha_tumor, beta,
                                 frac_data[1:], log_vol_list)
            gr_list = [self.b * (math.log(v_max) - log_vol_list[i]) for i in range(len(log_vol_list))]
        else:
            raise "Undefined growth type!"

        color = 'tab:red'
        ax2.set_ylabel('growth_rate/ $days^{-1}$', color=color)  # we already handled the x-label with ax1
        ax2.plot(list(range(frac_data.size)), gr_list, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.show()
        return ax2

    def BED_ploter(self, df_frac, df_patient, patient_idx):

        frac_data = df_frac.iloc[0, :]
        frac_data = frac_data.to_numpy()
        patient_data = df_patient.iloc[patient_idx, :]
        OAR_d_ratio = patient_data['OAR_d_ratio']

        bed_ptv_list = [d * (1 + d / self.ab_tumor) for d in frac_data]
        bed_ptv_list = np.cumsum(bed_ptv_list)

        bed_oar_list = [OAR_d_ratio * d * (1 + OAR_d_ratio * d / self.ab_tumor) for d in frac_data]
        bed_oar_list = np.cumsum(bed_oar_list)

        conv_bed_ptv_list = [2 * (1 + 2 / self.ab_tumor) for _ in range(30)]
        conv_bed_ptv_list = np.cumsum(conv_bed_ptv_list)

        conv_bed_oar_list = [OAR_d_ratio * 2 * (1 + OAR_d_ratio * 2 / self.ab_tumor) for _ in range(30)]
        conv_bed_oar_list = np.cumsum(conv_bed_oar_list)

        fig, ax = plt.subplots(figsize=(14, 5))  # 6.4, 4.8
        color = 'tab:red'
        ax.plot(list(range(len(frac_data))), bed_ptv_list, color=color, marker='+', linestyle='dashed',
                label='Optimal scheme: PTV')
        ax.plot(list(range(30)), conv_bed_ptv_list, color=color, marker='*', linestyle='dashed',
                label='Conventional scheme: PTV')

        text = 'total_dose: ' + str(np.sum(frac_data)) + '.0Gy\ntotal_BED: ' + str(bed_ptv_list[-1]) + 'Gy'
        ax.annotate(text, xy=(len(bed_ptv_list), bed_ptv_list[-1]),
                    xytext=(len(bed_ptv_list)-9, bed_ptv_list[-1]-4),
                    )  # arrowprops=dict(facecolor=color, shrink=0.05)
        text = 'total_dose: ' + str(60.0) + 'Gy\ntotal_BED: ' + str(conv_bed_ptv_list[-1]) + 'Gy'
        ax.annotate(text, xy=(30, conv_bed_ptv_list[-1]),
                    xytext=(30.5, conv_bed_ptv_list[-1]-4),
                    )

        color = 'tab:blue'
        ax.plot(list(range(len(frac_data))), bed_oar_list, color=color, marker='+',  linestyle='dashed',
                label='Optimal scheme: OAR')
        ax.plot(list(range(30)), conv_bed_oar_list, color=color, marker='*',  linestyle='dashed',
                label='Conventional scheme: OAR')

        text = 'total_dose: ' + str(round(np.sum(frac_data)*OAR_d_ratio, 1)) +\
               'Gy\ntotal_BED: ' + str(round(bed_oar_list[-1], 1)) + 'Gy'
        ax.annotate(text, xy=(len(bed_oar_list), bed_oar_list[-1]),
                    xytext=(len(bed_oar_list) - 8, bed_oar_list[-1]-1))

        text = 'total_dose: ' + str(round(np.sum(frac_data)*OAR_d_ratio, 1)) +\
               'Gy\ntotal_BED: ' + str(round(conv_bed_oar_list[-1], 1)) + 'Gy'
        ax.annotate(text, xy=(30, conv_bed_oar_list[-1]),
                    xytext=(30.5, conv_bed_oar_list[-1]-2))

        ax.set_xlim([0,  max(len(frac_data), 30) + 0.5])
        # ax.set_ylim([0, 4.5])

        ax.set_xticks(np.arange(0, max(len(frac_data), 30), 1))
        # ax.set_yticks(np.arange(0, 4, 1))

        ax.set_xlabel('time/day')
        ax.set_ylabel('Cumulative BED/Gy')
        ax.legend()
        ax.set_title("Cumulative BED for different fraction scheme(ab=10)")
        plt.show()


if __name__ == '__main__':
    config_path = Path(r"../Utils/config_env1.yml")
    frac_scheme_path = r"../Data/frac_log/ppo_env1/gom_4obs_4a_10.xlsx"
    patient_data_path = r'../Data/patient_data_env1_tumor_growth.xlsx'
    img_plotter = ImgPloter(config_path)

    # img_plotter.plot_tumor_growth()

    # initialize paramters
    img_plotter.init_para_with_config_info()
    df_frac = pd.read_excel(frac_scheme_path)
    df_patient = pd.read_excel(patient_data_path)

    # plot fraction scheme
    ax = img_plotter.daily_dose_plot(df_frac, title='Conventional Fractionation Scheme')
    img_plotter.growth_rate_plot(ax, df_frac, df_patient, patient_idx=0, mode='gom')

    img_plotter.BED_ploter(df_frac, df_patient, 1)

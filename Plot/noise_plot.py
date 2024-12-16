import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from Utils.readAndWrite.read_config_file import get_config_info
import pandas as pd
from RT_env.envs.ResponseModelClass import ResponseModelBase, LogisticPSIResponseModel, LogisticPSIResponseModelEnv3
import seaborn as sns

from plot_response_base_class import save_fig
from plot_response_1stART import compute_adapRT_rsp_

pd.set_option('display.max_columns', None)
sns.set_context("paper")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def kde_plot_noise_BED_diff(data_dict, img_save_path, suffix='noise_BED_KDE'):
    # Initialize an empty DataFrame
    all_df = pd.DataFrame()

    # Calculate the differences between the noisy and original data
    noise_levels = ['05', '10', '15']
    for noise_level in noise_levels:
        all_df[f'diff_OAR_BED_{noise_level}_noise'] = data_dict[noise_level] - data_dict['00']

    # Reshape the DataFrame from wide to long format
    all_df_long = all_df.melt(value_vars=all_df.columns,
                              var_name='noise_level', value_name='diff_OAR_BED')

    # Plot the stacked KDE plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=all_df_long, x="diff_OAR_BED", hue='noise_level', multiple="stack",
                fill=True, common_norm=False, alpha=0.6, ax=ax)

    # Calculate and annotate the percentage of diff_OAR_BED within ±2.5 Gy for each noise level
    note_positions = [0.85, 0.45, 0.2]  # Pre-defined positions for annotations

    for i, noise_level in enumerate(noise_levels):
        group = all_df_long[all_df_long['noise_level'].str.contains(noise_level)]
        percentage_within_range = (group['diff_OAR_BED'].abs() < 2.5).mean() * 100
        plt.text(4., note_positions[i], f'{percentage_within_range:.2f}%',
                 horizontalalignment='center', verticalalignment='bottom', fontsize=12)

    # Add visual cues
    plt.axvline(x=-2.5, color='black', linestyle='--')
    plt.axvline(x=2.5, color='black', linestyle='--')
    plt.axvspan(-2.5, 2.5, color='grey', alpha=0.3)

    # Set plot title and labels with LaTeX formatting for mathematical notation

    plt.title("KDE Plot: The Impact of Model Uncertainty on the OAR's BED \n Under Personalized De-escalation RT",
              fontsize=16, fontweight='medium')
    plt.xlabel("$BED^{OAR}_{original \\ plan} - BED^{OAR}_{re-optimized \\ plan}$ /Gy",
               fontsize=14, fontweight='medium')
    plt.ylabel('Density',
               fontsize=14, fontweight='medium')

    plt.xticks(np.arange(-10, 11, 2.5))
    # 增大刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Update legend labels
    legend_labels = ["15%", "10%", "5%"]
    plt.legend(title='Noise Level /%', labels=legend_labels)
    plt.tight_layout()

    if img_save_path is None:
        plt.show()
    else:
        save_fig(img_save_path, suffix=suffix)


def hist_plot_noise_prognosis(config_info, df_dict, threshold, img_save_path=None,
                              suffix='noise_hist_prognosis'):
    time_span = 41
    df_wo_noise = df_dict['00']
    plan_wo_noise = df_wo_noise.iloc[:, list(range(1, time_span, 1))]

    oar_bed_stats = df_wo_noise['adapRT_oar_BED'].describe()

    # Categorizing 'oar_bed' into 'High', 'Medium', and 'Low' based on quartiles
    conditions = [
        (df_wo_noise['adapRT_oar_BED'] <= oar_bed_stats['25%']).to_numpy(),  # High-benefit
        (df_wo_noise['adapRT_oar_BED'] > oar_bed_stats['75%']).to_numpy()]  # Low-benefit

    # Define choices corresponding to conditions
    choices = ['High', 'Low']

    # Use np.select to apply conditions and choices to the new 'Type' column
    df_wo_noise['Type'] = np.select(conditions, choices)

    noise_levels = ['05', '10', '15']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # 为整个大图加标题
    fig.suptitle("The Influence of Noise on the Prognosis of the Personalized De-escalation RT Plans",
                 fontsize=20, fontweight='medium')

    for idx, c in enumerate(choices):
        plot_df, pplot_df = pd.DataFrame(), []
        for noise_level in noise_levels:
            df_noise = df_dict[noise_level]
            df_noise_old_plan = df_noise.copy()
            df_noise_old_plan.iloc[:, list(range(1, time_span, 1))] = plan_wo_noise

            rsp_model = LogisticPSIResponseModel(df_noise_old_plan, config_info)
            final_rsp_list = compute_adapRT_rsp_(df_noise_old_plan, rsp_model, time_span, patient_id_list=None,
                                                 is_final_rsp=True)
            df_noise_old_plan['adaRT_final_response'] = final_rsp_list

            plot_df[f'art_rsp_{noise_level}_noise'] = df_noise['adaRT_final_response'] <= threshold
            plot_df[f'art_rsp_{noise_level}_noise_old_plan'] = df_noise_old_plan['adaRT_final_response'] <= threshold
            plot_df[f'con_rsp_{noise_level}_noise'] = df_noise['conRT_final_response'] <= threshold

            # 筛选出df_wo_noise['Type' == ‘High’]的索引
            LRC_index = df_wo_noise[df_wo_noise['Type'] == c].index
            art_noise = plot_df[f'art_rsp_{noise_level}_noise'][LRC_index]
            art_noise_old_plan = plot_df[f'art_rsp_{noise_level}_noise_old_plan'][LRC_index]
            crt_noise = plot_df[f'con_rsp_{noise_level}_noise'][LRC_index]

            true_index = crt_noise.index[crt_noise]

            art_noise = art_noise.loc[true_index]
            art_noise_old_plan = art_noise_old_plan.loc[true_index]

            number_noise = art_noise.value_counts().get(False, 0)
            number_noise_old_plan = art_noise_old_plan.value_counts().get(False, 0)

            pplot_df.append({
                'Noise Level': f'{noise_level}%',
                'Type': 'Re-optimized Plan',
                'Rate': number_noise / len(true_index) * 100
            })

            pplot_df.append({
                'Noise Level': f'{noise_level}%',
                'Type': 'Original Plan',
                'Rate': number_noise_old_plan / len(true_index) * 100
            })

        pplot_df = pd.DataFrame(pplot_df)
        ax = [ax1, ax2][idx]
        led = ['auto', False][idx]
        g = sns.barplot(data=pplot_df, x="Noise Level", y='Rate', hue='Type', ax=ax, legend=led)
        g.tick_params(axis='y', labelsize=14)

        for p in ax.patches:
            # 获取条形的位置信息
            x = p.get_x() + p.get_width() / 2  # x位置
            y = p.get_height()  # 高度
            if x < 0.01 and y < 0.01:
                continue
            # 在条形上方添加文本
            ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=12)

        # 获取当前x轴的刻度
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(["5%", "10%", "15%"])
        # 设定y轴的范围
        ax.set_ylim(0, 22.5)

        ax.set_ylabel('The Rate of Personalized RT Inferior to \n Conventional RT /%',
                      fontsize=14, fontweight='medium')
        ax.set_xlabel('Noise Level', fontsize=14, fontweight='medium')
        # 增大每个子图的坐标刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=12)

        # 为每个子图加标题
        ax.set_title(f'Group of {c}-benefit in BED', fontsize=16, fontweight='medium',
                     fontstyle='italic')
        sns.despine(ax=ax)

        # 增加两个子图的间隔
        plt.tight_layout(w_pad=6.)

    if img_save_path is None:
        plt.show()
    else:
        save_fig(img_save_path, suffix=suffix)















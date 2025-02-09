import pandas as pd
import numpy as np
from Utils.readAndWrite.read_config_file import get_config_info
import pathlib
from pathlib import Path


def generate_dataset(config_info, data_type, opt_mode):
    def get_range(key, point_num):
        mean_val = config_info['tumor_rt_parameters'][f'{key}_mean']
        std_val = config_info['tumor_rt_parameters'][f'{key}_std']
        min_val = config_info['tumor_rt_parameters'][f'{key}_min']
        max_val = config_info['tumor_rt_parameters'][f'{key}_max']

        val_list = np.random.normal(loc=mean_val, scale=std_val, size=point_num)
        val_list = val_list.clip(min=min_val, max=max_val)

        np.random.shuffle(val_list)

        return val_list

    min_vol = config_info['tumor_rt_parameters']['tumor_V0_min']
    max_vol = config_info['tumor_rt_parameters']['tumor_V0_max']

    if data_type == 'Lung':
        if opt_mode == 'train':
            random_numbers = np.round(np.random.uniform(min_vol, max_vol, 5000))
        else:
            random_numbers = np.round(np.random.uniform(min_vol, max_vol, 5000))
        np.random.shuffle(random_numbers)
    elif data_type == 'HN':
        if opt_mode == 'train':
            random_numbers = np.round(np.random.uniform(min_vol, max_vol, 5000))
        else:
            random_numbers = np.round(np.random.uniform(min_vol, max_vol, 5000))
        np.random.shuffle(random_numbers)
    else:
        raise 'wrong data type, only support Lung or HN!'

    df = pd.DataFrame({'patient_id': list(range(random_numbers.size)), 'tumor_V0': random_numbers})

    for k in ['alpha', 'lambda', 'PSI']:
        df[k] = get_range(k, random_numbers.size)

    df.to_excel(data_type + '_' + opt_mode + '_patients.xlsx', index=False)

    print("Finished.")


def add_random_noise_2model_para(df, noise_level):
    # noise_level: 0-1
    for k in ['alpha', 'lambda', 'PSI']:
        noise = np.clip(np.random.normal(0, noise_level, df.shape[0]), -1 * noise_level,
                        noise_level)
        df[k] += noise * df[k]

    df_w = df[['patient_id', 'tumor_V0', 'alpha', 'lambda', 'PSI']]
    df_w.to_excel(f'n_v2_train_ppo_18o_D6a_HN_LRC_60Gy_de-escalation_noise_{noise_level}.xlsx', index=False)
    # return df


if __name__ == '__main__':
    #  set random seed.
    np.random.seed(1)

    # HN_config_path = Path(r"../Utils/config_env2_HN.yml")
    # config_info = get_config_info(HN_config_path)
    # generate_dataset(config_info, 'HN', 'train')
    # generate_dataset(config_info, 'HN', 'test')
    #
    # Lung_config_path = Path(r"../Utils/config_env2_Lung.yml")
    # config_info = get_config_info(Lung_config_path)
    # generate_dataset(config_info, 'Lung', 'train')
    # generate_dataset(config_info, 'Lung', 'test')

    test_data_path = Path(r"../Result/v2_train_ppo_18o_D6a_HN_LRC_60Gy_de-escalation.xlsx")
    test_data = pd.read_excel(test_data_path)
    # add_random_noise_2model_para(test_data.copy(), 0.05)
    add_random_noise_2model_para(test_data.copy(), 0.1)
    add_random_noise_2model_para(test_data.copy(), 0.15)
    # add_random_noise_2model_para(test_data.copy(), 0.2)
    # add_random_noise_2model_para(test_data.copy(), 0.25)





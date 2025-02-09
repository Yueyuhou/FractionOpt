import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pandas as pd
from math import exp, log, pow
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from RT_env.envs.ResponseModelClass import ResponseModelBase, LogisticPSIResponseModel


# Env V2: Begin to ART after 2 weeks' convRT
class RTEnvV1(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config_info, df_, record_flag=False):

        self.render_mode = self.metadata['render_modes'][0]

        self.patient_idx = None
        self.fraction_idx = 0
        self.fraction_dose = 0
        self.dose_history_list = []
        self.alpha_dict = None
        self.ab_tumor = None
        self.ab_oar = None
        self.OAR_d_ratio = None
        self.lambda_dict = None
        self.PSI_dict = None

        self.config_info = config_info

        self.df = df_

        self.observation_space = spaces.Box(-10, 10, shape=(18,))

        self.action_space = spaces.Discrete(6)

        # self._action_to_dose = {
        #     0: 0.,
        #     1: 0.5,
        #     2: 1.,
        #     3: 2.,
        #     4: 3.,
        #     5: 4.,
        # }  # Unit: Gy

        self._action_to_dose = {
            0: 0.5,
            1: 1.,
            2: 1.5,
            3: 2.0,
            4: 2.5,
            5: 3.,
        }  # Unit: Gy

        self.total_dose = self.config_info['tumor_rt_parameters']['total_dose']
        self.crt_fraction_dose = self.config_info['tumor_rt_parameters']['crt_fraction_dose']

        self.tumor_model = LogisticPSIResponseModel(self.df, self.config_info)
        self.w_vol_rew = self.config_info['tumor_rt_parameters']['w_vol_rew']
        self.w_dose_rew = self.config_info['tumor_rt_parameters']['w_dose_rew']
        self.w_reduction_rew = self.config_info['tumor_rt_parameters']['w_reduction_rew']
        self.terminate_rew = self.config_info['tumor_rt_parameters']['terminate_rew']

        self.record_flag = record_flag
        self._seed = 0
        self.max_BED_flag = False

        self._OAR_d_ratio = 0.3

        self.fraction_BED = np.array(
            [self.tumor_model.BED(d * self._OAR_d_ratio) for d in self._action_to_dose.values()])

        self.action_mask_flag = self.config_info['training_config']['action_mask']

        self.therapy_time_span = \
            int(self.config_info['tumor_rt_parameters']['total_dose'] / self.config_info['tumor_rt_parameters'][
                'crt_fraction_dose'] / 5 * 7 - 1)

        self.old_obs = []

    def initialization(self, para_dict=None):
        self.fraction_idx = 14
        self.fraction_dose = 0
        # deliver conventional RT for 2 weeks at first.
        self.dose_history_list = [0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0]

        if para_dict is None:
            _, _, _, _ = self.tumor_model.initialization(self.patient_idx, self._seed)
        else:
            _, _, _, _ = self.tumor_model.initialization(self.patient_idx, self._seed,
                                                         alpha_=para_dict['alpha'],
                                                         lmd_=para_dict['lambda'],
                                                         PSI_=para_dict['PSI'],
                                                         tumor_V0=para_dict['tumor_V0'])

        self.alpha_dict = self.tumor_model.alpha_dict
        self.lambda_dict = self.tumor_model.lambda_dict
        self.PSI_dict = self.tumor_model.PSI_dict

        self.ab_tumor = self.tumor_model.ab_tumor
        self.ab_oar = self.tumor_model.ab_oar
        self.OAR_d_ratio = self.tumor_model.OAR_d_ratio
        self.max_BED_flag = False

    def _get_obs(self, is_reset=False):
        # obs list : patient specific parameters(6), vol info(9), BED info(3))
        _ = self.tumor_model.apply_ART_dose_scheme(self.fraction_idx, self.fraction_dose)

        SF = exp(-1 * self.alpha_dict['alpha'] * self.fraction_dose - 0.1 * self.alpha_dict[
            'alpha'] * self.fraction_dose ** 2)

        obs_list = [self.fraction_idx, self.fraction_dose, self.alpha_dict['alpha'],
                    self.lambda_dict['lambda_'], self.PSI_dict['PSI'], 1 - SF]
        vol_response = self.tumor_model.get_vol_response_state(fraction_idx=self.fraction_idx)

        dose_response = self.tumor_model.get_dose_response_state(fraction_idx=self.fraction_idx,
                                                                 dose=self.fraction_dose)

        obs_list.extend(vol_response)
        obs_list.extend(dose_response)

        return np.asarray(obs_list, dtype=np.float32).flatten()

    def _get_info(self):
        return {}

    def get_mask(self, obs, cut_off_mask=False):
        mask_ = np.ones(self.action_space.n).flatten()
        remaining_BED = obs[-1]

        if self.action_mask_flag:
            # max_OAR_BED = self.tumor_model.conRT_max_oar_BED
            # remaining_BED = max_OAR_BED - cumulative_ART_OAR_BED
            mask_[1:] = np.asarray(self.fraction_BED[1:] <= remaining_BED)
        return mask_

    def reset(self, seed=None, options=None):
        # get i-th patient data.
        if options is None or options['patient_idx'] is None:
            self.patient_idx = self.np_random.integers(0, self.df.shape[0], size=1, dtype=int)[0]
            self.initialization()
        else:
            self.patient_idx = options['patient_idx']
            patient_data = self.df.iloc[self.patient_idx, :]
            para_dict = {'alpha': patient_data['alpha'],
                         'lambda': patient_data['lambda'],
                         'PSI': patient_data['PSI'],
                         'tumor_V0': patient_data['tumor_V0'],
                         }
            self.initialization(para_dict=para_dict)

        obs = self._get_obs(is_reset=True)
        self.fraction_idx += 1

        info = {'mask': self.get_mask(obs, cut_off_mask=False)}

        return obs, info

    @staticmethod
    def weekend_judge(fraction_idx, dose):
        # There is No therapy in the weekend.
        if (fraction_idx % 7 == 6) or (fraction_idx % 7 == 0):
            return 0.
        else:
            return dose

    def max_BED_judge(self, action, dose):
        delivered_OAR_bed, oar_bed_max = self.get_cumulative_BED()
        next_delivered_OAR_bed = delivered_OAR_bed + self.BED(dose*self.OAR_d_ratio)

        if next_delivered_OAR_bed > oar_bed_max:
            self.max_BED_flag = True
            while True:
                action = int(action - 1)
                if action < 0:
                    dose = 0
                    break
                alternative_dose = self._action_to_dose[action]
                if delivered_OAR_bed + self.BED(alternative_dose*self.OAR_d_ratio) <= oar_bed_max:
                    return alternative_dose

        return dose

    def get_cumulative_BED(self):
        delivered_OAR_bed = [self.BED(d * self.OAR_d_ratio) for d in self.dose_history_list]
        delivered_OAR_bed = np.sum(delivered_OAR_bed)

        oar_bed_max = self.BED(self.crt_fraction_dose * self.OAR_d_ratio) * self.total_dose / self.crt_fraction_dose

        return delivered_OAR_bed, oar_bed_max

    def dose_transform(self, action, dose):
        dose = self.weekend_judge(self.fraction_idx, dose)
        if dose > 0:
            dose = self.max_BED_judge(action, dose)

        return dose

    def step(self, action):
        action = int(action)
        fraction_dose = self._action_to_dose[action]

        # no therapy in the weekend.
        self.fraction_dose = self.dose_transform(action, fraction_dose)

        self.dose_history_list.append(self.fraction_dose)
        obs = self._get_obs()

        rew = self.get_reward(obs)
        terminated_flag = self.is_terminated()

        # if terminated_flag and self.tumor_model.is_adaRtResponse_cutoff():
        #     rew = rew + self.terminate_rew

        # is_BED_over_maximum = obs[-2] <= 0
        # if terminated_flag and is_BED_over_maximum:
        #     rew = rew - self.terminate_rew

        self.fraction_idx = self.fraction_idx + 1

        info = {'mask': self.get_mask(obs, cut_off_mask=False)}

        return obs, rew, terminated_flag, False, info

    def get_reward(self, obs):
        is_cutoff_vol_reached = float(obs[-4] <= 0)  # 1 reached
        is_remaining_BED = float(obs[-2] >= -0.1)  # 1 remaining BED

        rew_vol = is_cutoff_vol_reached * self.w_vol_rew
        rew_BED = is_cutoff_vol_reached * is_remaining_BED * obs[-2] * self.w_dose_rew
        rew_reduction = (1 - is_cutoff_vol_reached) * np.clip(obs[-6]/5, -1, 1) * self.w_reduction_rew
        # rew_reduction = np.clip(obs[-6] / 5, -1, 1) * self.w_reduction_rew

        rew = rew_vol + rew_BED + rew_reduction
        # print("rew: ", rew, "rew_vol: ", rew_vol, "rew_BED: ", rew_BED, "obs[-6]: ", rew-rew_vol-rew_BED)
        return rew

    def is_truncated(self):
        return False

    def is_terminated(self):
        flag = False
        if self.fraction_idx >= self.therapy_time_span-1:
            flag = True

        if flag and self.record_flag:
            self.record()

        return flag

    def seed(self, sd):
        self.np_random, self._seed = seeding.np_random(sd)
        return self.np_random, self._seed

    def BED(self, d):
        return d * (1 + d / self.ab_tumor)

    def record(self):
        log_path = Path(self.config_info['training_config']['logdir'])
        l_name = self.config_info["training_config"]["log_name"]
        mode = self.config_info['training_config']['training_mode']

        save_path = log_path.joinpath(self.config_info['training_config']['task'][-2:] +
                                      l_name + mode + '.csv')

        data = np.asarray(self.dose_history_list).reshape(1, -1)

        total_BED = np.sum([self.BED(d) for d in self.dose_history_list])
        total_oar_BED = np.sum([self.BED(d * self.OAR_d_ratio) for d in self.dose_history_list])

        conv_BED = self.BED(self.crt_fraction_dose) * self.total_dose / self.crt_fraction_dose
        conv_oar_BED = self.BED(self.crt_fraction_dose * self.OAR_d_ratio) * self.total_dose / self.crt_fraction_dose
        conRT_cutoff_OAR_BED = conv_oar_BED

        alpha = self.alpha_dict['alpha']
        lmd = self.lambda_dict['lambda_']
        PSI = self.PSI_dict['PSI']
        V0 = self.tumor_model.tumor_V0

        patient_idx = self.df.loc[self.patient_idx, 'patient_id']

        conRT_final_response = self.tumor_model.convRT_response[-1]
        adaRT_final_response = self.tumor_model.adapRT_response[-1]

        data = np.insert(data, 0, [patient_idx, conv_BED, total_BED, conv_oar_BED, conRT_cutoff_OAR_BED,
                                   total_oar_BED, alpha, lmd, PSI, V0,
                                   conRT_final_response, adaRT_final_response]).reshape(1, -1)

        # conRT_response_list = np.asarray(self.tumor_model.convRT_response).reshape(1, -1)
        # conRT_response_list = np.insert(conRT_response_list, 0, patient_idx).reshape(1, -1)
        #
        # adaRT_response_list = np.asarray(self.tumor_model.adapRT_response).reshape(1, -1)
        # adaRT_response_list = np.insert(adaRT_response_list, 0, patient_idx).reshape(1, -1)

        df_act = pd.DataFrame(data)
        # df_conRT_response = pd.DataFrame(conRT_response_list)
        # df_adaRT_response = pd.DataFrame(adaRT_response_list)

        if not save_path.exists():
            df_act.to_csv(save_path, index=False, header=False)
            # df_conRT_response.to_csv(save_path.parent.joinpath(self.config_info['training_config']['task'][-2:] +
            #                                                    self.config_info['training_config'][
            #                                                        'log_name'] + '_convRT_response.csv'),
            #                          index=False, header=False)
            # df_adaRT_response.to_csv(save_path.parent.joinpath(self.config_info['training_config']['task'][-2:] +
            #                                                    self.config_info['training_config'][
            #                                                        'log_name'] + '_adaRT_response.csv'),
            #                          index=False, header=False)
        else:
            df_act.to_csv(save_path, index=False, mode='a', header=False)
            # df_conRT_response.to_csv(save_path.parent.joinpath(self.config_info['training_config']['task'][-2:] +
            #                                                    self.config_info['training_config'][
            #                                                        'log_name'] + '_convRT_response.csv'),
            #                          index=False, mode='a', header=False)
            # df_adaRT_response.to_csv(save_path.parent.joinpath(self.config_info['training_config']['task'][-2:] +
            #                                                    self.config_info['training_config'][
            #                                                        'log_name'] + '_adaRT_response.csv'),
            #                          index=False, mode='a', header=False)

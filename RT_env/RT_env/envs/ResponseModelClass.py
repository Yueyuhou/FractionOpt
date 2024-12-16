import numpy as np
from math import exp, pow, log
import pandas as pd
from pathlib import Path
from gymnasium.utils import seeding
import warnings


class ResponseModelBase(object):
    def __init__(self, df, config_info):
        self.config_info = config_info
        self.patient_idx = None
        self.patient_data = None
        self.df = df
        self.tumor_V0 = None
        self.alpha = None
        self._ab_tumor = self.config_info['tumor_rt_parameters']['ab_tumor']
        self._ab_oar = self.config_info['tumor_rt_parameters']['ab_oar']
        self._OAR_d_ratio = None
        # self.current_vol = None
        # self.current_vol_history = []
        self.seed = None
        self.rng = None
        self.therapy_time_span = \
            int(self.config_info['tumor_rt_parameters']['total_dose'] / self.config_info['tumor_rt_parameters'][
                'crt_fraction_dose'] / 5 * 7 - 1)

    def initialization(self, *args, **kwargs):
        pass

    def BED(self, d):
        return d * (1 + d / self._ab_tumor)

    def cumulative_BED(self, dose_list):
        BED_list = [self.BED(d) for d in dose_list]
        return np.sum(BED_list)

    @property
    def ab_tumor(self):
        return self._ab_tumor

    @property
    def ab_oar(self):
        return self._ab_oar

    @property
    def OAR_d_ratio(self):
        return self._OAR_d_ratio


class LogisticPSIResponseModel(ResponseModelBase):
    def __init__(self, df, config_info):
        super().__init__(df, config_info)

        self.OARs_list = ['OAR', ]

        self._alpha_dict = {"mean": self.config_info['tumor_rt_parameters']['alpha_mean'],
                            "std": self.config_info['tumor_rt_parameters']['alpha_std'],
                            "min": self.config_info['tumor_rt_parameters']['alpha_min'],
                            "max": self.config_info['tumor_rt_parameters']['alpha_max']}

        self._lambda_dict = {"mean": self.config_info['tumor_rt_parameters']['lambda_mean'],
                             "std": self.config_info['tumor_rt_parameters']['lambda_std'],
                             "min": self.config_info['tumor_rt_parameters']['lambda_min'],
                             "max": self.config_info['tumor_rt_parameters']['lambda_max']}

        self._PSI_dict = {"mean": self.config_info['tumor_rt_parameters']['PSI_mean'],
                          "std": self.config_info['tumor_rt_parameters']['PSI_std'],
                          "min": self.config_info['tumor_rt_parameters']['PSI_min'],
                          "max": self.config_info['tumor_rt_parameters']['PSI_max']}

        self.K = None  # tumor carrying capacity (K) as the maximum tumor volume  supported by a given environment.

        self._OAR_d_ratio = 0.3

        self.conventional_dose_list = 2 * np.ones(self.therapy_time_span - 1)
        weekend_idx = np.logical_or(np.arange(self.therapy_time_span - 1) % 7 == 5,
                                    np.arange(self.therapy_time_span - 1) % 7 == 6)
        self.conventional_dose_list[weekend_idx] = 0
        self.conventional_dose_list = np.insert(self.conventional_dose_list, 0, 0)
        self.adaRT_dose_list = []

        self.convRT_response = []
        self.adapRT_response = []
        self.cutoff_response = float(config_info['tumor_rt_parameters']['cutoff_response'])

        self.conRT_max_oar_BED = 0.0
        self.last_final_ART_rsp = 0.0

        # training_mode: de-escalation for sensitive tumor / escalation for insensitive tumor
        # The training_mode will determine how to sample parameters of the tumor model.
        training_mode_dict = {"de-escalation": -1, "escalation": 1}
        self.training_mode = training_mode_dict[self.config_info['training_config']['training_mode']]

    def set_model_parameters(self, alpha_=None, lmd_=None, PSI_=None):
        # if values are assigned
        alpha, lambda_, PSI = alpha_, lmd_, PSI_

        # else: sample values from a truncated Gaussian distribution
        if alpha is None:
            alpha = float(
                np.clip(self.rng.normal(loc=self._alpha_dict['mean'],
                                        scale=self._alpha_dict['std'], size=1)[0],
                        a_min=self._alpha_dict['min'],
                        a_max=self._alpha_dict['max']))

        if lambda_ is None:
            lambda_ = float(np.clip(self.rng.normal(loc=self._lambda_dict['mean'],
                                                    scale=self._lambda_dict['std'], size=1)[0],
                                    a_min=self._lambda_dict['min'],
                                    a_max=self._lambda_dict['max']))

        if PSI is None:
            PSI = float(np.clip(self.rng.normal(loc=self._PSI_dict['mean'],
                                                scale=self._PSI_dict['std'], size=1)[0],
                                a_min=self._PSI_dict['min'],
                                a_max=self._PSI_dict['max']))

        self._alpha_dict['alpha'] = alpha
        self._lambda_dict["lambda_"] = lambda_
        self._PSI_dict['PSI'] = PSI
        self.K = self.tumor_V0 / self._PSI_dict['PSI']

        return alpha, lambda_, PSI

    def _cal_response(self, init_vol, dose_list):
        alpha = self._alpha_dict['alpha']
        lmd = self._lambda_dict['lambda_']

        pre_v = init_vol
        response = []

        if isinstance(dose_list, int) or isinstance(dose_list, float):
            dose_list = [dose_list]
        i = 0
        for d in dose_list:
            pre_v = self._cal_RT_vol(pre_v, d, alpha, lmd, self.K)
            response.append(pre_v)
            i += 1

        # convert absolute volume changes to relative changes.
        response = np.array(response) / self.tumor_V0 * 100

        return response

    def cal_RT_response(self, init_vol=None, dose_list=None):
        response = []
        if init_vol is None:
            init_vol = float(self.tumor_V0)
            response = [100.0]

        if dose_list is None:
            dose_list = self.conventional_dose_list[1:]
            response.extend(self._cal_response(init_vol, dose_list))
        else:
            response.extend(self._cal_response(init_vol, dose_list))

        return response

    def cal_RT_response_single_fraction(self, init_vol, dose_list):
        rsp = self._cal_response(init_vol, dose_list)
        return rsp[0]
        # self.current_vol = self.adapRT_response[-1] / 100 * self.tumor_V0

    def set_model_parameters_based_on_training_mode(self):
        while True:
            alpha, lambda_, PSI = self.set_model_parameters()

            conRT_60_dose_list = np.array([2, 2, 2, 2, 2, 0, 0,
                                           2, 2, 2, 2, 2, 0, 0,
                                           2, 2, 2, 2, 2, 0, 0,
                                           2, 2, 2, 2, 2, 0, 0,
                                           2, 2, 2, 2, 2, 0, 0,
                                           2, 2, 2, 2, 2])

            convRT_response = self.cal_RT_response(dose_list=conRT_60_dose_list)

            # {"de-escalation": -1, "escalation": 1}
            if self.training_mode > 0 and convRT_response[-1] > self.cutoff_response:
                break
            elif self.training_mode < 0 and convRT_response[-1] <= self.cutoff_response:
                break
            else:
                pass

        return alpha, lambda_, PSI

    def initialization(self, patient_idx, seed, alpha_=None, lmd_=None, PSI_=None, tumor_V0=None):
        if self.seed is None:
            self.rng, self.seed = seeding.np_random(seed)

        self.patient_idx = patient_idx
        self.patient_data = self.df.iloc[self.patient_idx, :]
        # Response = volume / tumor_V0 * 100
        if tumor_V0 is None:
            self.tumor_V0 = self.patient_data['tumor_V0']
        else:
            self.tumor_V0 = tumor_V0

        if alpha_ is not None and lmd_ is not None and PSI_ is not None:
            alpha, lambda_, PSI = self.set_model_parameters(alpha_, lmd_, PSI_)
        else:
            alpha, lambda_, PSI = self.set_model_parameters_based_on_training_mode()

        self.convRT_response = self.cal_RT_response()

        self.adapRT_response = self.convRT_response.copy()

        self.adaRT_dose_list = self.conventional_dose_list.copy()

        oar_dose_list = self.conventional_dose_list * self.OAR_d_ratio
        self.conRT_max_oar_BED = self.cumulative_BED(oar_dose_list)
        self.last_final_ART_rsp = self.adapRT_response[-1]

        return alpha, lambda_, PSI, self.tumor_V0

    def set_adaRT_dose_scheme(self, fraction_idx, fraction_dose):
        self.adaRT_dose_list[fraction_idx] = fraction_dose
        cumulative_OAR_BED_adaRT = \
            self.cumulative_BED(np.asarray(self.adaRT_dose_list[:fraction_idx + 1]) * self._OAR_d_ratio)
        remaining_OAR_BED = max(self.conRT_max_oar_BED - cumulative_OAR_BED_adaRT, 0.0)
        remaining_conRT_num = np.floor(remaining_OAR_BED / self.BED(2.0 * self._OAR_d_ratio))

        adaRT_dose_list_copy = self.adaRT_dose_list[:fraction_idx + 1].copy()
        adaRT_dose_list_copy = np.concatenate((adaRT_dose_list_copy, np.zeros(200)))

        #  After a new ART fraction is determined, the remaining fractions are set according to
        #  conRT mode with remaining BED.
        i, count = 1, 0
        while True:
            if count >= remaining_conRT_num:
                break

            idx = fraction_idx + i
            if (idx % 7 == 6) or (idx % 7 == 0):
                adaRT_dose_list_copy[idx] = 0.0
            else:
                adaRT_dose_list_copy[idx] = 2.0
                count += 1
            i += 1

        #  After the remaining conRT fractions are set, the last fraction is determined by the remaining BED.
        diff_BED = max(remaining_OAR_BED - remaining_conRT_num * self.BED(2.0 * self._OAR_d_ratio), 0.0)
        fraction_dose = [0, 0.5, 1, 1.5]
        fraction_BED_list = np.asarray([self.BED(d * self._OAR_d_ratio) for d in fraction_dose])
        last_dose = fraction_dose[fraction_BED_list[diff_BED >= fraction_BED_list].argmax()]

        while True:
            idx = int(fraction_idx + i)
            if (idx % 7 == 6) or (idx % 7 == 0):
                adaRT_dose_list_copy[idx] = 0.0
            else:
                adaRT_dose_list_copy[idx] = last_dose
                break
            i += 1

        self.adaRT_dose_list = adaRT_dose_list_copy[:int(self.therapy_time_span)]
        # print("adaRT_dose_list:")
        # print(self.adaRT_dose_list[1:8])
        # print(self.adaRT_dose_list[8:15])
        # print(self.adaRT_dose_list[15:22])
        # print(self.adaRT_dose_list[22:29])
        # print(self.adaRT_dose_list[29:36])
        # print(self.adaRT_dose_list[36:42])
        return adaRT_dose_list_copy[:int(self.therapy_time_span)]

    def apply_ART_dose_scheme(self, fraction_idx, fraction_dose):
        adaRT_dose_list = self.set_adaRT_dose_scheme(fraction_idx, fraction_dose)
        self.adapRT_response = self.cal_RT_response(dose_list=adaRT_dose_list[1:])

        return self.adapRT_response

    def get_current_ART_vol(self, fraction_idx):
        return self.adapRT_response[fraction_idx] / 100 * self.tumor_V0

    def get_current_conRT_vol(self, fraction_idx):
        return self.convRT_response[fraction_idx] / 100 * self.tumor_V0

    def get_ART_plan(self):
        return self.adaRT_dose_list.copy()

    @staticmethod
    def make_tumor_resistant(alpha, lmd, PSI, factor=0.1):
        new_alpha = np.random.uniform(low=(1 - factor) * alpha, high=alpha, size=1)[0]
        new_lmd = np.random.uniform(low=lmd, high=(1 + factor) * lmd, size=1)[0]
        PSI = np.random.uniform(low=PSI, high=(1 + factor) * PSI, size=1)[0]
        return new_alpha, new_lmd, PSI

    def _cal_RT_vol(self, pre_v, dose, alpha, lambda_, K):
        # First, the pre-irradiated volume get killed and thus shrink. -> Post-irradiated vol
        # Then post-irradiated volume grow according to the pattern of logistic model. t = 1 day

        t = 1  # day
        SF = exp(-1 * alpha * dose * (1 + 1 / self.ab_tumor * dose))
        post_v = pre_v - (1 - SF) * pre_v * (1 - pre_v / K)

        new_v = K / (1 + (K / post_v - 1) * exp(-1 * lambda_ * t))
        return new_v

    def get_vol_response_state(self, fraction_idx):
        # i-th Vconrt: conRT_response
        # i-th Vadart: adaptive_response
        # the last Vconrt: last_conRT_response
        # cutoff response: self.cutoff_response

        # vol obs: delta_V: 100-conRT, delta_V: 100-adaRT,
        # delta_V:conRT  delta_V:adaRT  delta_V:conRT-adaRT  delta_V:conRT-cutoff,  delta_V:adaRT-cutoff
        PSI_fraction = self.get_current_ART_vol(fraction_idx) / self.K

        final_conRT_response = self.convRT_response[-1]
        final_adaRT_response = self.adapRT_response[-1]

        diff_final_ART_rsp = self.last_final_ART_rsp - final_adaRT_response
        self.last_final_ART_rsp = final_adaRT_response

        cutoff_rsp = self.cutoff_response - 1

        return [PSI_fraction,
                diff_final_ART_rsp,
                1 / max(diff_final_ART_rsp, 0.1),
                1 / max(final_conRT_response - cutoff_rsp, 0.1),
                1 / max(final_adaRT_response - cutoff_rsp, 0.1),
                1 / max(final_adaRT_response - final_conRT_response, 0.1),
                final_conRT_response - final_adaRT_response,
                final_conRT_response - cutoff_rsp,
                final_adaRT_response - cutoff_rsp]

    def get_dose_response_state(self, fraction_idx, dose):
        # dose obs:
        # delta_BED: BED_conRT_i-BED_adaRT_i, Sum(BED_conRT), Sum(BED_adaRT), delta_sum

        BED_diff = self.BED(self.conventional_dose_list[fraction_idx] * self._OAR_d_ratio) - self.BED(
            dose * self._OAR_d_ratio)
        cumulative_BED_adaRT = self.cumulative_BED(np.asarray(self.adaRT_dose_list) * self._OAR_d_ratio)

        remain_OAR_BED = self.conRT_max_oar_BED - cumulative_BED_adaRT

        return [BED_diff, remain_OAR_BED, 5 / max(self.conRT_max_oar_BED - cumulative_BED_adaRT, 0.5)]

    def is_adaRtResponse_cutoff(self):
        return self.adapRT_response[-1] <= self.cutoff_response

    # Property Function
    @property
    def alpha_dict(self):
        return self._alpha_dict

    @property
    def lambda_dict(self):
        return self._lambda_dict

    @property
    def PSI_dict(self):
        return self._PSI_dict


class LogisticPSIResponseModelEnv3(LogisticPSIResponseModel):
    def __init__(self, df, config_info):
        super().__init__(df, config_info)

    def initialization_for_env3(self, patient_idx, seed, is_resistant=True, resist_paras_dict=None):
        # Initialize the random number generator.
        if seed is None:
            self.rng = np.random.default_rng()
            warnings.warn("default_rng is used for Response Model!")
        else:
            self.rng, self.seed = seeding.np_random(seed)

        self.patient_idx = patient_idx
        self.patient_data = self.df.iloc[patient_idx, :]

        # Response = volume / tumor_V0 * 100
        self.tumor_V0 = self.patient_data['tumor_V0']

        # get the original model parameters from the Dataframe.
        # This is how the model reacts to RT in the first three weeks.
        self._alpha_dict['alpha'] = self.patient_data['alpha'].tolist()
        self._lambda_dict["lambda_"] = self.patient_data['lmd'].tolist()
        self._PSI_dict['PSI'] = self.patient_data['PSI'].tolist()
        self.K = self.tumor_V0 / self._PSI_dict['PSI']

        self.convRT_response = self.cal_RT_response()

        self.adaRT_dose_list = self.patient_data[list(range(0, 41))].dropna().values
        self.adapRT_response = self.apply_ART_dose_scheme(fraction_idx=14, fraction_dose=0)

        # NOTE: The resistance of the tumor may change after the first three weeks.
        #  So we should adapt our model to the changes.
        # Make the tumor more resistant to RT.
        if is_resistant:
            if resist_paras_dict is None:
                self._alpha_dict['alpha'], self._lambda_dict["lambda_"], self._PSI_dict['PSI'] = \
                    self.make_tumor_resistant(alpha=self._alpha_dict['alpha'],
                                              lmd=self._lambda_dict["lambda_"],
                                              PSI=self._PSI_dict['PSI'],
                                              factor=0.2)
            else:
                self._alpha_dict['alpha'], self._lambda_dict["lambda_"], self._PSI_dict['PSI'] = \
                    resist_paras_dict['alpha'], resist_paras_dict['lmd'], resist_paras_dict['PSI']

            # Update tumor response under convRT after tumor become more resistant.
            latter_response = self.cal_RT_response(init_vol=self.get_current_conRT_vol(21),
                                                   dose_list=self.conventional_dose_list[22:])
            self.convRT_response[22:] = latter_response[:]

            # Update tumor response under ART after tumor become more resistant.
            latter_response = self.cal_RT_response(init_vol=self.get_current_ART_vol(21),
                                                   dose_list=self.adaRT_dose_list[22:])
            self.adapRT_response[22:] = latter_response[:]

        oar_dose_list = self.conventional_dose_list * self.OAR_d_ratio
        self.conRT_max_oar_BED = self.cumulative_BED(oar_dose_list)
        self.last_final_ART_rsp = self.adapRT_response[-1]

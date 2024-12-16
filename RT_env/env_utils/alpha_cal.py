import numpy as np
import pandas as pd
import yaml
from enum import Enum
import math


class HypoxicMode(Enum):
    oxic = 1
    hypoxic = 2
    hypoxic_with_vol_effect = 3
    hp_with_vol_and_shrinkage = 4


class AlphaFunctionType(Enum):
    const = 1
    linear = 2
    quadratic = 3


class Alpha_calculator:
    def __init__(self, config_info):
        self.config_info = config_info
        data_path = config_info['data_config']['data_path']
        self.df = pd.read_excel(data_path)

        self.hypoxic_mode = HypoxicMode(config_info['tumor_rt_parameters']['hypoxic_mode'])
        self.alpha_function_type = AlphaFunctionType(config_info['tumor_rt_parameters']['alpha_function_type'])

        self.alpha_oxic = config_info['tumor_rt_parameters']['alpha_oxic']
        self.alpha_hypo = config_info['tumor_rt_parameters']['alpha_hypo']

    def get_alpha(self):
        R = self.df.loc['R']
        r0 = self.df.loc['r0']
        V = self.df.loc['tumor_V0']

        if self.hypoxic_mode == HypoxicMode.oxic:
            return self.config_info['tumor_rt_parameters']['alpha_tumor']
        elif self.hypoxic_mode == HypoxicMode.hypoxic and self.alpha_function_type == AlphaFunctionType.const:
            return self.hypoxic_const(R, r0, V)
        elif self.hypoxic_mode == HypoxicMode.hypoxic_with_vol_effect:
            if self.alpha_function_type == AlphaFunctionType.linear:
                return self.hypoxic_vol_linear(R, r0, V)
        else:
            raise "Wrong HypoxicMode or AlphaFunctionType!"

    def hypoxic_const(self, R, r0, V):
        return (self.alpha_hypo * np.power((R - r0), 3) +
                self.alpha_oxic * (np.power(R, 3) - np.power(R - r0, 3))) / V

    def hypoxic_vol_linear(self, R, r0, V):
        ratio = np.asarray(r0 / R)
        return 0.75 * self.alpha_hypo * np.power(1 - ratio, 4) + self.alpha_oxic * (1 - np.power((1 - ratio), 3))

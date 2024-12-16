from gymnasium.envs.registration import register

register(
     # tumor model + with weekend + discrete action
     id="RT_env/RTEnv-v1",
     entry_point="RT_env.envs:RTEnvV1"
)


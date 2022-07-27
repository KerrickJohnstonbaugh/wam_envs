from gym.envs.registration import registry, register, make, spec

from wam_envs.envs.WAMWipe import WAMWipeEnv

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='WAMWipe{}-v1'.format(suffix),
        entry_point='wam_envs.envs.WAMWipe:WAMWipeEnv',
        kwargs=kwargs,
        max_episode_steps=200,
    )
from odqm import metrics
from odqm.data import dummy_buffer
import argparse

import yaml

KEY_TO_METRIC = {
    'bwd': metrics.BellmanWassersteinDistance,   
}



def run(cfg):
    timesteps = cfg['train_timesteps']
    eval_freq = cfg['eval_freq']

    batch_size = cfg['batch_size']
    device = cfg['device']
    if cfg['data_name'] == 'dummy':
        state_dim = 10
        action_dim = 4
        method = KEY_TO_METRIC[cfg['method']['name']](state_dim=state_dim, action_dim=action_dim, device=device, **cfg['method'])
        buffer = dummy_buffer(state_dim=state_dim, action_dim=action_dim, max_size=10000, device=device)
    
    for i in range(timesteps):
        method.train(buffer.sample(batch_size=batch_size))

        if (i + 1) % eval_freq == 0:
            print(f'Timestep: {i + 1}')
            print(method.estimate(buffer.sample(batch_size=batch_size)))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='ODQM',
                    description='ODQM provides methods for estimating data quality for offline reinforcement learning',
                    epilog='Text at the bottom of help')

    parser.add_argument('config_path')
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    run(config)
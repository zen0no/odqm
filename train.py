from odqm import metrics
from odqm.data import make_buffer
import argparse

import yaml

KEY_TO_METRIC = {
    'bwd': metrics.BellmanWassersteinDistance,
}


def run(cfg):
    buffer = make_buffer(cfg['data'])

    method_cfg = cfg['method']
    method_cfg['state_dim'] = buffer.state_dim
    method_cfg['action_dim'] = buffer.action_dim
    method = KEY_TO_METRIC[method_cfg['name']](**method_cfg)

    method.estimate(buffer)
    

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
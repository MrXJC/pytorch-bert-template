from sklearn.model_selection import ParameterGrid, ParameterSampler
import argparse
from parse_config import ConfigParser, MockConfigParser
from train import train
from collections import OrderedDict
import hashlib
import pandas as pd
# /opt/xujc/Projects/PycharmProjects/DeepLearning/NLP/Bert/BertESIM


def record(df, parameter_dict, results):
    for key, value in parameter_dict.items():
        if key[-1] not in df:
            df[key[-1]] = [value]
            continue
        df[key[-1]].append(value)
    for key, value in results.items():
        if key not in df:
            df[key] = [value]
            continue
        df[key].append(value)
    return df


def get_md5_from_param(param, column):
    src = ''
    for k in column:
        src += str(k)
        src += str(param[k])
    m = hashlib.md5()
    print(src)
    m.update(src.encode('utf-8'))
    return m.hexdigest()


def _search(config, parameter_dict, parameters):
    column = [k for k, _ in parameter_dict.items()]

    filepath = config.base_save_dir / 'models' / config.exper_name / 'SearchResult' / 'parameter-results.csv'

    if filepath.exists():
        df = pd.read_csv(str(filepath)).to_dict(orient='list')
    else:
        df = OrderedDict()
        df['md5'] = []

    columns = []
    for idx, parameter in enumerate(parameters):
        m = get_md5_from_param(parameter, column)
        if m in df['md5']:
            continue
        df['md5'].append(m)

        config.update_config(parameter)
        result = train(config)
        record(df, parameter, result)
        if len(columns) == 0:
            columns = ['md5']
            columns.extend([k[-1] for k, _ in parameter_dict.items()])
            for k, _ in result.items():
                columns.append(k)
        df2 = pd.DataFrame(df, columns=columns)
        df2.to_csv(str(filepath), index=None)


def grid_search(config, parameter_dict):
    parameters = list(ParameterGrid(parameter_dict))
    _search(config, parameter_dict, parameters)
    return


def random_search(config, parameter_dict, n_iter=10):
    parameters = list(ParameterSampler(parameter_dict, n_iter))
    _search(config, parameter_dict, parameters)
    return


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="ATEC_BERT/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-debug', '--debug', default=False, type=bool,
                      help='debug')

    args.add_argument('-sm', '--searchMode', default='random', type=str,
                      help='random or grid(default:random)')

    config = ConfigParser(args)
    # config = MockConfigParser()
    parameter_dict = OrderedDict()
    parameter_dict[('optimizer', 'args', 'lr')] = [2e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    parameter_dict[('data_loader', 'args', 'batch_size')] = [16, 32, 48, 64, 96]

    # random_search(config, parameter_dict, 2)
    # grid_search(config, parameter_dict)

    args = args.parse_args()
    if args.searchMode == 'random':
        random_search(config, parameter_dict, 10)
    elif args.searchMode == 'grid':
        grid_search(config, parameter_dict)
    else:
        print(f"{args.searchMode} is the unknown mode of ParameterSearch ")

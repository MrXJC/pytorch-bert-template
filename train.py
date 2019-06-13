import argparse
import collections
import data_loader.data_loaders as module_data
import data_loader.processor as module_processor
from parse_config import ConfigParser
import model.model as module_arch
from pytorch_pretrained_bert.modeling import BertConfig
from agent import Agent


# /opt/xujc/Projects/PycharmProjects/DeepLearning/NLP/Bert/BertESIM

def train(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    processor = config.initialize('processor', module_processor, logger, config)

    data_loader = config.initialize('data_loader', module_data, processor.data_dir, mode = "train", debug= config.debug_mode)
    test_data_loader = config.initialize('data_loader', module_data, processor.data_dir, mode = "test", debug= config.debug_mode)

    if config.all:
        valid_data_loader = test_data_loader
    else:
        valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    if config.bert_config_path:
        bert_config = BertConfig(config.bert_config_path)
        model = config.initialize('arch', module_arch, config=bert_config, num_labels = processor.nums_label())
    else:
        model = config.initialize_bert_model('arch', module_arch, num_labels = processor.nums_label())

    logger.info(model)
    agent = Agent(model,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader)

    agent.train()
    return agent.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="ATEC_BERT/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-a', '--all', default=False, type=bool,
                      help='all for training, test as validation')
    args.add_argument('-debug', '--debug', default=False, type=bool,
                      help='debug')
    args.add_argument('-reset', '--reset', default=False, type=bool,
                      help='debug')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--ep', '--epoch'], type=int, target=('trainer', 'epochs'))
    ]

    config = ConfigParser(args, options)
    # config = MockConfigParser()
    train(config)
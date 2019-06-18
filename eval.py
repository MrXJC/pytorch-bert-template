import argparse
import data_loader.data_loaders as module_data
import data_loader.processor as module_processor
from parse_config import ConfigParser
import model.model as module_arch
from agent import Agent
from pytorch_pretrained_bert.modeling import BertConfig
import pandas as pd
import numpy as np

def eval(config, filename):
    logger = config.get_logger('test')
    # setup data_loader instances
    processor = config.initialize('processor', module_processor, logger, config)
    processor.get_eval(filename)
    test_data_loader = config.initialize('data_loader', module_data, processor.data_dir, mode="eval",
                                         debug=config.debug_mode)

    # build model architecture, then print to console
    if config.bert_config_path:
        bert_config = BertConfig(config.bert_config_path)
        model = config.initialize('arch', module_arch, config=bert_config, num_labels=processor.nums_label())
    else:
        model = config.initialize_bert_model('arch', module_arch, num_labels=processor.nums_label())
    logger.info(model)
    agent = Agent(model, config=config, test_data_loader=test_data_loader)
    return agent.test(detail=True), processor


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-e', '--eval', default="eval.csv", type=str,
                      help='the path of eval file (default: eval.csv)')
    args.add_argument('-reset', '--reset', default=False, type=bool,
                      help='debug')

    config = ConfigParser(args)
    args = args.parse_args()
    (qts, labels, outputs, _), processor = eval(config, args.eval)
    result_path = processor.data_dir / "RAW/result.csv"
    df = {"qt": [], 'label': [], "output": []}
    for index in range(len(qts)):
        if labels[index] != int(np.argmax(outputs[index].cpu().numpy())):
            df["qt"].append(processor.tokenizer.convert_ids_to_tokens(qts[index].cpu().numpy()))
            df["label"].append(labels[index].cpu().numpy())
            df["output"].append(outputs[index].cpu().numpy())
    pd.DataFrame(df, columns=["qt", "label", "output"]).to_csv(result_path)

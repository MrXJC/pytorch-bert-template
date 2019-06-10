import argparse
import data_loader.data_loaders as module_data
import data_loader.processor as module_processor
from parse_config import ConfigParser
import model.model as module_arch
from agent import Agent
from pytorch_pretrained_bert.modeling import BertConfig

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
    return agent.test(detail=True), preprocess


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    filename = "data/poc_test_balance.csv"
    config = ConfigParser(args)
    (qs, ts, labels, outputs), preprocess = eval(config, filename)

    result = "data/result.csv"
    df = {"qs": [], "ts": [], 'label': [], "output": []}
    for index in range(len(qs)):
        if labels[index] != int(outputs[index] > 0.5):
            df["qs"].append(
                "".join(map(lambda x: preprocess.idx2word[x] if x != 0 else '', qs[index])))
            df["ts"].append(
                "".join(map(lambda x: preprocess.idx2word[x] if x != 0 else '', ts[index])))
            df["label"].append(labels[index].cpu().numpy())
            df["output"].append(outputs[index].cpu().numpy())
    pd.DataFrame(df).to_csv("data/result.csv")

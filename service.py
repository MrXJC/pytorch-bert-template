from flask import Flask
from flask import request
import argparse
from flask_json import FlaskJSON, JsonError, as_json
import data_loader.processor as module_processor
import model.model as module_arch
from parse_config import ConfigParser
from agent import Agent
from pytorch_pretrained_bert.modeling import BertConfig

app = Flask(__name__)
json = FlaskJSON(app)


def predict(q, t):
    batch = processor.handle_bert_on_batch([q], [t])
    return agent.predict(batch)


@app.route('/api/bertOrigin', methods=['POST'])
@as_json
def test():
    data = request.get_json(
        force=False,
        silent=False,
        cache=True)

    try:
        _, label = predict(
            data['query'],
            data['target'])
        response = {'label': int(label[0])}
    except (KeyError, TypeError, ValueError):
        raise JsonError(description='Invalid value.')
    return response


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    logger = config.get_logger('test')
    # setup data_loader instances
    processor = config.initialize(
        'processor', module_processor, logger, config)
    # build model architecture, then print to console
    # build model architecture, then print to console
    if config.bert_config_path:
        bert_config = BertConfig(config.bert_config_path)
        model = config.initialize('arch', module_arch, config=bert_config, num_labels=processor.nums_label())
    else:
        model = config.initialize_bert_model('arch', module_arch, num_labels=processor.nums_label())
    # logger.info(model)
    agent = Agent(model, config=config)

    app.run(host='0.0.0.0', port=5000, debug=True)
import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json, setup_seed


class ConfigParser:
    def __init__(self, args, options='', timestamp=True):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        self.bert_config_path = None

        if args.resume:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume.parent / 'config.json'
            self.bert_config_path = str(self.resume.parent / 'BertConfig.json')
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            self.resume = None
            self.cfg_fname = 'config' / Path(args.config)

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        self._config = _update_config(config, options, args)

        # set save_dir where trained model and log will be saved.
        self.base_save_dir = Path(self.config['trainer']['save_dir'])
        self.exper_name = self.config['processor']['args']['data_name'] + \
            '_' + self.config['arch']['type']

        timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''
        self._save_dir = self.base_save_dir / 'models' / self.exper_name / timestamp
        self._log_dir = self.base_save_dir / 'log' / self.exper_name / timestamp

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        setup_seed(self._config['seed'])

        self.debug_mode = args.debug if "debug" in args else False
        self.all = args.all if "all" in args else False
        self.reset = args.reset if "all" in args else False

        self.gradient_accumulation_steps = self.config['trainer']['gradient_accumulation_steps']

        if self.all:
            self.config["data_loader"]["args"]["validation_split"] = 0.0

        if self.debug_mode:
            self.config["trainer"]["epochs"] = 2

    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        if 'pretrained_model_name_or_path' in module_args:
            module_args.pop("pretrained_model_name_or_path")
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def initialize_bert_model(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name).from_pretrained(
            cache_dir="data/.cache", *args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def save(self):        # save updated config file to the checkpoint dir
        if not os.path.exists(str(self.save_dir / 'config.json')):
            self.save_dir.mkdir(parents=True, exist_ok=True)
            write_json(self.config, self.save_dir / 'config.json')
    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def update_config(self, parameter):

        for key, value in parameter.items():
            _set_by_path(self.config, key, value)


        self._save_dir = self.base_save_dir / 'models' / self.exper_name / 'SearchResult'
        self._log_dir = self.base_save_dir / 'log' / self.exper_name / 'SearchResult'

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # configure logging module
        setup_logging(self.log_dir)

        self.gradient_accumulation_steps = self.config['trainer']['gradient_accumulation_steps']

        return
# helper functions used to update config dict with custom cli options


def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


class MockConfigParser:
    def __init__(self, cfg_path='ATEC_BERT/config.json', resume=None):
        # parse default and custom cli options
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        if resume:
            self.resume = Path(resume)
            self.cfg_fname = self.resume.parent / 'config.json'
            self.bert_config_path = str(self.resume.parent / 'BertConfig.json')
        else:
            self.cfg_fname = 'config' / Path(cfg_path)

        # load config file and apply custom cli options
        self._config = read_json(self.cfg_fname)

        # set save_dir where trained model and log will be saved.
        self.base_save_dir = Path(self.config['trainer']['save_dir'])
        self.exper_name = self.config['processor']['args']['data_name'] + \
                          '_' + self.config['arch']['type']

        self._save_dir = self.base_save_dir / 'models' / self.exper_name / 'mock'
        self._log_dir  = self.base_save_dir / 'log' / self.exper_name / 'mock'

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        self.debug_mode = True
        self.all = False
        self.reset = False
        self.gradient_accumulation_steps = self.config['trainer']['gradient_accumulation_steps']

        if self.debug_mode:
            self.config["trainer"]["epochs"] = 2

    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def initialize_bert_model(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name).from_pretrained(
            cache_dir="data/.cache", *args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def save(self):        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def update_config(self, parameter):

        for key, value in parameter.items():
            _set_by_path(self.config, key, value)


        self._save_dir = self.base_save_dir / 'models' / self.exper_name / 'SearchResult'
        self._log_dir = self.base_save_dir / 'log' / self.exper_name / 'SearchResult'

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # configure logging module
        setup_logging(self.log_dir)

        self.gradient_accumulation_steps = self.config['trainer']['gradient_accumulation_steps']

        return
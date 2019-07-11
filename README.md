# pytorch-bert-template
A  flexible pytorch template for Natural Language Processing  based on Bert. 

Now, it just support the `NLC` (Natural Language Classification), `NLI` (Natural Language Inference) and other simple classification mission. It will support the `NER` and `Machine Comprehension` in the future

* [pytorch-bert-template](#pytorch-bert-template)
	* [Requirements](#requirements)
  * [How to use](#How-to-use)
    * [Start](#starting)
    * [Training](#training)
    * [Testing](#testing)
    * [Eval](#eval)
    * [Predict](#predict)
  * [ParameterSearch](#parameter-search)
  * [Folder Structure](#folder-structure)
  * [Usage](#usage)
    * [Config file format](#config-file-format)
    * [Using config files](#using-config-files)
    * [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
    * [Additional logging](#additional-logging)
    * [Validation data](#validation-data)
    * [Checkpoints](#checkpoints)
    * [TensorboardX Visualization](#tensorboardx-visualization)
  * [Customization](#customization)
  * [TODOs](#todos)
  * [Acknowledgments](#acknowledgments)
  
## Requirements

* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4
* tqdm
* tensorboard >= 1.7.0 (Optional for TensorboardX) or tensorboard >= 1.14 (Optional for pytorch.utils.tensorboard)
* tensorboardX >= 1.2 (Optional for TensorboardX), see [Tensorboard Visualization][#tensorboardx-visualization]

## How to use

### Start

1. Write your own processor like `ATECProcessor`.

```
class ATECProcessor(BaseBertProcessor):
    def __init__(self, logger, config, data_name, data_path, bert_vocab_file, max_len=50, query_max_len=20,
                 target_max_len=20, do_lower_case=True, test_split=0.0, training=True):
        self.skip_row = 0
        super().__init__(logger, config, data_name, data_path, bert_vocab_file, max_len, query_max_len,
                 target_max_len, do_lower_case, test_split, training)

    def get_labels(self):
        """See base class."""
        return [u'0', u'1']

    def split_line(self, line):
        line = line.strip().split('\t')
        q, t, label = line[1], line[2], line[-1]
        return q, t, label
```
* You also should realize the interface `get_labels` ,  `split_line` and  the variable `self.skip_row`.

2. Move your data into the directory `data/RAW/`.
3. Create New configuration File `config/{DataName}_{ModelName}/config.json` like `config/ATEC_BERT/config.json`.
* `{DataName}`  is representative of the name of DataSet
* `{ModelName}`  is representative of the name of Model
4. Adjust the `processor` configuration. The content of the `config.json` is as follows.
```
{
    "n_gpu": 1,
    "seed" : 28,
    "processor": {
        "type": "ATECProcessor", ## the name of Processor
        "args": {
            "data_name":  "ATEC", ## the name of DataName
            "bert_vocab_file": "bert-base-chinese",
            "data_path":  "atec_nlp_sim_train.csv", ## the name of DataSet File Name
            "test_split": 0.2,
            "max_len": 63,
            "query_max_len": 20,
            "target_max_len": 20,
            "do_lower_case" : true
        }
    },
    ....
}
```
### Training

* `{DataName}_{ModelName}` ==> `ATEC_BERT`
```
python train.py --config {DataName}_{ModelName}/config.json
```

### Testing
```
python test.py -r saved/models/{DataName}_{ModelName}/timestamp/~.pth
```

### Eval
```
python eval.py -r saved/models/{DataName}_{ModelName}/timestamp/~.pth -e $EVAL_DATA_PATH
```
  
### Predict
```
python predict.py -r saved/models/{DataName}_{ModelName}/timestamp/~.pth -s1 str1 -s2 str2
```
### Service
```
python service.py -r saved/models/{DataName}_{ModelName}/timestamp/~.pth
```

## Parameter Search

It is more complicated.

```
python ParameterSearch.py -sm random
```

## Folder Structure
 
**Todo: ......**

## Usage

The code in this repo is an ATEC example of the template.

Try `python train.py -c ATEC_BERT/config.json` to run code.

### Config file format

Config files are in `.json` format:

```javascript
{
    "n_gpu": 1,  // number of GPUs to use for training.
    "seed" : 28, // random seed
    "processor": {
        "type": "ATECProcessor",
        "args": {
            "data_name":  "ATEC",
            "bert_vocab_file": "bert-base-chinese",
            "data_path":  "atec_nlp_sim_train.csv",  // dataset path
            "test_split": 0.2, // size of test dataset.
            "max_len": 63, // the max length of bert input
            "query_max_len": 20, // not use
            "target_max_len": 20, // not use
            "do_lower_case" : true
        }
    },

    "data_loader": {
        "type": "BertDataLoader",  // selecting data loader
        "args": {
            "batch_size": 96,
            "shuffle": true,          // shuffle training data before splitting
            "validation_split": 0.1,  // size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 2  // number of cpu processes to be used for data loading
        }
    },

    "arch": {
        "type": "BertOrigin",  // name of model architecture to train
        "args": {  // args of model architecture 
            "pretrained_model_name_or_path": "bert-base-chinese" // bert pretrained_model_name_or_path
        }
    },

    "optimizer": {
        "type": "BertAdam",
        "args":{
                "lr"  : 1e-4,
            "warmup"  : 0.1 ,
            "schedule": "warmup_linear"
        }
    },
    "loss": {
        "type": "cross_entropy_loss",
        "args": {
            "weights": [0.223, 1] // loss weight
        }
    },

    "metrics": [
        "F1","acc"
    ],

    "trainer": {
        "epochs": 50, // number of training epochs

        "save_dir": "saved/", // checkpoints are saved in save_dir/models/name
        "save_period": 1, // save checkpoints every save_freq epochs
        "verbosity": 2, // 0: quiet, 1: per epoch, 2: full
        
        "monitor": "max val_F1", // mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 5, // number of epochs to wait before early stop. set 0 to disable.
        "gradient_accumulation_steps": 1, 
        "tensorboardX": true  // enable tensorboardX visualization
    }
}
```
### Using config files

Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config ATEC_BERT/config.json
  ```
  
### Resuming from checkpoints

You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```
  
### Using Multiple GPU

You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c ATEC_BERT/config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c ATEC_BERT/config.py
  ```

### [Additional logging](https://github.com/victoresque/pytorch-template/blob/master/README.md#additional-logging)

### [Validation data](https://github.com/victoresque/pytorch-template/blob/master/README.md#validation-data)

### [Checkpoints](https://github.com/victoresque/pytorch-template/blob/master/README.md#checkpoints)

### [TensorboardX Visualization](https://github.com/victoresque/pytorch-template/blob/master/README.md#tensorboard-visualization)

## Customization 

**Todo: ......**

## TODOs

- [X] Rename trainer to agent
- [ ] Finish the document
- [X] Finish the ParameterSearch Function.
- [ ] Monitor and control the agent by Wechat
- [ ] Realize the NER
- [ ] Realize the Machine Comprehension

## Acknowledgments
This project is inspired by the project [pytorch-template](https://github.com/victoresque/pytorch-template) by [victoresque](https://github.com/victoresque)

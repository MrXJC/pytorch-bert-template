import pytorch_pretrained_bert
import math

def bert_optimizer(model, config, data_loader):
    # # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    num_train_optimization_steps = int(math.ceil(data_loader.n_samples / data_loader.batch_size + 0.5) / config.gradient_accumulation_steps)\
                                   *config['trainer']['epochs']

    print(data_loader.n_samples, data_loader.batch_size, config.gradient_accumulation_steps,
          config['trainer']['epochs'], num_train_optimization_steps)
    # start = False
    # for name, param in model.named_parameters():
    #     if "11" in name:
    #         start = True
    #     if start:
    #         print(name, param.requires_grad)
    #         param.requires_grad = True

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = config.initialize('optimizer', pytorch_pretrained_bert.optimization,
                                  params=optimizer_grouped_parameters, t_total=num_train_optimization_steps)
    return optimizer

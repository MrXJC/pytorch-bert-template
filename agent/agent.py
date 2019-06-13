import numpy as np
import time
import torch
from base import BaseAgent
import agent.loss as module_loss
import agent.metric as module_metric
from agent.optimizer import bert_optimizer


class Agent(BaseAgent):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, config=None,
                 data_loader =None, valid_data_loader=None, test_data_loader=None, lr_scheduler=None):
        super().__init__(model, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader  = test_data_loader
        self.do_train = self.data_loader is not None
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # # get function handles of loss and metric

        self.loss = self.config.initialize('loss', module_loss, device=self.device)
        self.metrics = [getattr(module_metric, met) for met in config['metrics']]

        if self.do_train:
            self.log_step = int(np.sqrt(data_loader.batch_size))
            self.optimizer = bert_optimizer(model, config, data_loader)

        if config.resume is not None:
            self.best_path = config.resume
            self._resume_checkpoint(config.resume)

    def _eval_metrics(self, outputs, labels):
        outputs = outputs.detach().cpu().numpy()
        labels  = labels.detach().cpu().numpy()
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(outputs, labels)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, batch in enumerate(self.data_loader):

            output, label_ids = self._predict(batch)
            loss = self.loss(output, label_ids)

            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            loss.backward()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            now_metrics = self._eval_metrics(output, label_ids)
            total_metrics += now_metrics

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if batch_idx % self.log_step == 0:
                metric_str =''
                for i, mtr in enumerate(self.metrics):
                    metric_str += " {}: {:.6f}".format(mtr.__name__, now_metrics[i])

                self.logger.debug('[{}] Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}{}{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(), metric_str, ' lr: {}'.format(self.optimizer.get_lr()[0])))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                output, label_ids = self._predict(batch)
                loss = self.loss(output, label_ids)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, label_ids)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
             if "embedding" not in name:
                 self.writer.add_histogram(name, p, bins='auto')
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def test(self, detail = False):
        """
        Test after training finishing

        :return: A log that contains information about test

        Note:
            The test metrics in log must have the key 'val_metrics'.
        """
        if not self.do_test:
            return

        if self.do_train:
            self._resume_checkpoint(self.best_path, "test")

        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            # if detail:
            #     qs, ts, labels, outputs = [], [], [], []
            for batch_idx, batch in enumerate(self.test_data_loader):

                output, label_ids = self._predict(batch)

                loss = self.loss(output, label_ids)
                total_test_loss += loss.item()
                total_test_metrics += self._eval_metrics(output, label_ids)
                # if detail:
                #     qs.extend(q), ts.extend(t), labels.extend(label), outputs.extend(output)

        for key, value in {
            'test_loss': total_test_loss / len(self.test_data_loader),
            'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
        }.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))


        value = (total_test_metrics / len(self.test_data_loader)).tolist()
        # if detail:
        #     return qs, ts, labels, outputs, {'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)}
        return {'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)}

    def _predict(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        output = self.model(input_ids, input_mask, segment_ids)
        label_ids = label_ids.view(-1)
        return output, label_ids

    def predict(self, batchs):
        batch = tuple(torch.LongTensor(t).to(self.device) for t in batchs)
        input_ids, input_mask, segment_ids, _ = batch
        outputs = self.model(input_ids, input_mask, segment_ids)
        return outputs, np.argmax(outputs.detach().cpu().numpy(), axis=1)



def load_data(data_dir, tokenizer, processor, max_length, batch_size, data_type):
    """ 导入数据， 并返回对应的迭代器
    Args:
        data_dir: 原始数据目录
        tokenizer： 分词器
        processor: 定义的 processor
        max_length: 句子最大长度
        batch_size: batch 大小
        data_type: "train" or "dev", "test" ， 表示加载哪个数据

    Returns:
        dataloader: 数据迭代器
        examples_len: 样本大小
    """

    label_list = processor.get_labels()

    if data_type == "train":
        examples = processor.get_train_examples(data_dir)
    elif data_type == "dev":
        examples = processor.get_dev_examples(data_dir)
    elif data_type == "test":
        examples = processor.get_test_examples(data_dir)
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_examples_to_features(
        examples, label_list, max_length, tokenizer)

    dataloader = convert_features_to_tensors(features, batch_size)

    examples_len = len(examples)

    return dataloader, examples_len

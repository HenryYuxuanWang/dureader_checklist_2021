import os

if __name__ == '__main__':
    from model_pytorch import Model
    from training_args import training_args

    train_data_dir = "./dataset"
    dev_data_dir = "./dataset"
    model = Model()

    train_examples = model.processor.get_train_examples(train_data_dir)
    dev_examples = model.processor.get_dev_examples(dev_data_dir)

    train_features = model.prepare_training_features(train_examples)
    dev_features = model.prepare_validation_features(dev_examples)

    dev_data = {"examples": dev_examples, "features": dev_features}

    model.train(train_features, dev_data, args=training_args)

    # dev_data_dir = "./test1"
    # model = Model(path='./outputs')
    # dev_examples = model.processor.get_dev_examples(dev_data_dir)
    # dev_features = model.prepare_validation_features(dev_examples)
    # dev_data = {"examples": dev_examples, "features": dev_features}
    # model.evaluate(dev_data, args=training_args, prefix='test1')

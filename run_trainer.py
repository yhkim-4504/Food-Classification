import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help='config yaml path', default='config.yaml')
parser.add_argument("-m", "--model", help='model name', default='SwinV2_Base_256')
parser.add_argument("-o", "--optimizer", help='optimizer name', default='AdamW')
parser.add_argument("-s", "--scheduler", help='scheduler_name', default='CustomCosineAnnealingWarmupRestarts')
args = parser.parse_args()

CONFIG_SELECT_DICT = {
    'model': args.model,
    'optimizer': args.optimizer,
    'scheduler': args.scheduler
}

if __name__ == '__main__':
    # Load Modules & config
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config['trainer']['cuda_device']
    import mlflow
    from utils import load_config_to_mlflow
    from trainer import Trainer

    # Set mlflow experiment & run_name
    experiment_name = config['trainer']['experiment_name']
    run_name = config['trainer']['run_name']

    # Start Training
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        load_config_to_mlflow(config, **CONFIG_SELECT_DICT)
        trainer = Trainer(config, CONFIG_SELECT_DICT, test_mode=False)
        trainer.run_train()
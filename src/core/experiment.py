# experiment.py: Top-level assembler — wires dataset, model, and execution objects from config.

from src.config import get_default_config
from src.task import get_task_spec
from src.data.mnist import MnistDataset
from src.data.dataloader import DataLoader
from src.models.mlp import MLP
from src.core.optimizers import SGD
from src.core.trainer import Trainer
from src.core.evaluator import Evaluator
from src.core.predictor import Predictor


class Experiment:
    def __init__(self, config=None):
        if config is None:
            config = get_default_config()
        self.config = config

        task = config["task"]
        task_spec = get_task_spec(task)
        dataset_dir = config.get("dataset_dir")
        batch_size = config["batch_size"]

        train_dataset = MnistDataset("train", task, dataset_dir=dataset_dir)
        test_dataset = MnistDataset("test", task, dataset_dir=dataset_dir)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model = MLP(task=task, seed=config["seed"])
        optimizer = SGD(self.model, lr=config.get("lr", 0.01))
        self.trainer = Trainer(self.model, optimizer, task_spec)
        self.evaluator = Evaluator(self.model, task_spec)
        self.predictor = Predictor(self.model, task_spec)

    def run(self):
        """Train for num_epochs. Returns list of per-epoch log dicts."""
        logs = []
        for epoch in range(1, self.config["num_epochs"] + 1):
            train_log = self.trainer.fit(self.train_loader)
            test_log = self.evaluator.evaluate(self.test_loader)
            logs.append({"epoch": epoch, "train": train_log, "test": test_log})
        return logs

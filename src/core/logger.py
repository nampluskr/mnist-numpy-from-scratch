# logger.py: Epoch-level loss/metric logger with CSV export.

import os


class Logger:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.metrics = []

    def append(self, epoch, loss, metric):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.metrics.append(metric)

    def load(self, logs):
        for log in logs:
            self.append(log["epoch"], log["loss"], log["metric"])

    def to_dict(self):
        return {
            "epochs": self.epochs,
            "losses": self.losses,
            "metrics": self.metrics,
        }

    def to_csv(self, save_path):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("epoch,loss,metric\n")
            for epoch, loss, metric in zip(self.epochs, self.losses, self.metrics):
                f.write(f"{epoch},{loss:.6f},{metric:.6f}\n")

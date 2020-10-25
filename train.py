from arch_model.arch_model import ArchModel
from dataset.yalefaces import Yalefaces
from dataset.thermo import Thermo
from trainers.batch_trainer import BatchTrainer
import os


if __name__ == "__main__":
    split = 0.8

    trainer = BatchTrainer()
    Thermo().raw_load(split=split)
    # Return the dataset in a dict
    (dataset_train, dataset_eval) = Thermo().get(split=split)
    # Get the already compiled model from the
    # respective architecture
    # Build a trainer with our model
    trainer.set_model(ArchModel().get_arch('siam'))
    # Train the model
    trainer.set_path_to_save(os.path.join(Thermo().dataset_name, str(split)))
    trainer.loop_train(dataset_train, loops=5000, checkpoint=100)
    exit()

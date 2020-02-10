from arch_model.arch_model import ArchModel
from dataset.yalefaces import Yalefaces
from trainers.batch_trainer import BatchTrainer

if __name__ == "__main__":
    database = Yalefaces()
    arch = ArchModel()

    # Load database in database
    database.load()
    # Return the database in a dict
    dataset_dict = database.get()   
    # Get the already compiled model from the
    # respective architecure
    model = arch.get_arch("siam")
    # Build a trainer with our model
    trainer = BatchTrainer(model)
    # print(trainer.model.summary())
    # Train the model
    trainer.enable_training()
    trainer.loop_train(dataset_dict)

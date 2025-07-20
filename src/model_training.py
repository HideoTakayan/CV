from preprocessing import DataLoader
import config
from model import create_model, train_and_save_model

if __name__ == "__main__":
    print("[1] Loading data...")
    loader = DataLoader()
    loader.load_data()
    class_names = loader.get_classes()

    print(f"[2] Number of classes: {len(class_names)}")

    print("[3] Creating model...")
    model = create_model(num_classes=len(class_names))
    model.summary()

    print("[4] Training model...")
    train_and_save_model(model, loader.train_data, loader.test_data, class_names)

import os
import json
import matplotlib.pyplot as plt
import config

def load_history():
    """
    Load lại lịch sử huấn luyện từ file JSON.
    """
    if not os.path.exists(config.MODEL_HISTORY_PATH):
        raise FileNotFoundError(f"Không tìm thấy file lịch sử huấn luyện tại {config.MODEL_HISTORY_PATH}")

    with open(config.MODEL_HISTORY_PATH, 'r') as f:
        history = json.load(f)
    return history

def plot_accuracy_and_loss(history):
    """
    Vẽ biểu đồ accuracy và loss.
    """
    epochs = range(1, len(history['accuracy']) + 1)

    # Accuracy plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'bo-', label='Training acc')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/history_plot.png")
    plt.show()
    print("Đã lưu biểu đồ huấn luyện tại: results/plots/history_plot.png")

if __name__ == "__main__":
    history = load_history()
    plot_accuracy_and_loss(history)

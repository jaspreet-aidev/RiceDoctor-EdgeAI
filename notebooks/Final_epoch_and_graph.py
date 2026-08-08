import matplotlib.pyplot as plt

# 1. Extract the final epoch metrics from the history dictionary
final_acc = history.history['accuracy'][-1]
final_loss = history.history['loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]

# 2. Print the formatted telemetry for your Master Logbook
print("="*50)
print("FINAL TELEMETRY DATA FOR LOGBOOK")
print("="*50)
print(f"Training Accuracy   : {final_acc:.4f} ({(final_acc*100):.2f}%)")
print(f"Training Loss       : {final_loss:.4f}")
print(f"Validation Accuracy : {final_val_acc:.4f} ({(final_val_acc*100):.2f}%)")
print(f"Validation Loss     : {final_val_loss:.4f}")
print("="*50)

# 3. Generate the Structural Diagnostic Graphs
plt.figure(figsize=(12, 5))

# Graph Alpha: Accuracy Trajectory
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
plt.title('System Accuracy Trajectory')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# Graph Beta: Loss / Memorization Trajectory
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.title('System Loss Trajectory')
plt.ylabel('Loss (Categorical Crossentropy)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

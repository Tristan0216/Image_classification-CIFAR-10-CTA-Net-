import numpy as np
import matplotlib.pyplot as plt

with open('log.txt', 'r') as f:
    content = f.read()
loss_values = list(map(float, content.split(',')))

plt.figure(figsize=(10, 6))
plt.plot(range(len(loss_values)), loss_values, label='Average Loss')
plt.title('Training Loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss Value')
plt.grid(True)

plt.yticks(np.arange(0.003, 1.8, 0.05))  
plt.savefig('model_training_loss.png')
plt.close()

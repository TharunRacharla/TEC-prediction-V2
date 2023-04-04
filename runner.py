import pickle
import numpy as np

# Opening saved model
with open("TEC_model.pkl", "rb") as file:
    current_model = pickle.load(file)
inputs = np.array([[2030, 188, 12, 200, 300, 5, 10.5]])
inputs.reshape(1, -1)
prediction = current_model.predict(inputs)
output = prediction
print(output)

import tensorflow as tf
import numpy as np
import os

# 1. Initialize the TFLite Interpreter Engine
tflite_model_path = '/content/drive/MyDrive/model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 2. Extract System Hardware Input/Output Tensor Details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("--- Engine Telemetry Checked ---")
print(f"Expected Input Shape: {input_details['shape']}")
print(f"Expected Data Type:  {input_details['dtype']}")
print("--------------------------------\n")

# 3. Create a Synthetic Sample (or point to a real image path)
# MobileNetV2 uses an input size of (1, 224, 224, 3)
# Generating a sample normalized between 0.0 and 1.0
test_image_float = np.random.rand(1, 224, 224, 3).astype(np.float32)

# 4. Mandatory INT8 Quantization Scaling Check
# Because input_type is int8, we must translate floating values using the model's quantization parameters
if input_details['dtype'] == np.int8:
    input_scale, input_zero_point = input_details['quantization']
    # Formula: quantized_value = (float_value / scale) + zero_point
    test_image_quantized = (test_image_float / input_scale) + input_zero_point
    test_image_final = test_image_quantized.astype(np.int8)
    print("Execution Log: Input mapped smoothly to INT8 range.")
else:
    test_image_final = test_image_float
    print("Execution Log: Running native Float32 array layout.")

# 5. Feed the Intelligence Pipeline
interpreter.set_tensor(input_details['index'], test_image_final)

# 6. Fire the Inference Loop
print("Computing neural matrix paths...")
interpreter.invoke()

# 7. Collect the Output Probabilities
raw_predictions = interpreter.get_tensor(output_details['index'])[0]

# Decode Output Scale if Output is Quantized INT8
if output_details['dtype'] == np.int8:
    output_scale, output_zero_point = output_details['quantization']
    predictions = (raw_predictions.astype(np.float32) - output_zero_point) * output_scale
else:
    predictions = raw_predictions

# 8. Render Metrics Report
predicted_class_index = np.argmax(predictions)
confidence_score = predictions[predicted_class_index]

print("\n=== INFERENCE TRIAL COMPLETED SUCCESSFULLY ===")
print(f"Predicted Class Index:  {predicted_class_index}")
print(f"Model Confidence Array: {predictions}")
print(f"Highest Confidence:     {confidence_score:.4f}")
print("==============================================")

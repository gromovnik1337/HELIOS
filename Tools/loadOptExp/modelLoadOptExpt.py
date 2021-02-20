import os
import tensorflow as tf 
from tensorflow import keras

print(tf.version.VERSION)

# Change into current directiory 
os.chdir(os.path.dirname(os.path.realpath(__file__)))

converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
#                                       tf.lite.OpsSet.TFLITE_BUILTINS]
#converter.target_spec.supported_types = [tf.float16]

tflite_model_quant = converter.convert()

with open('detect.tflite', 'wb') as f:
  f.write(tflite_model_quant)
  
"""
import pathlib
# Creae the export folder
tflite_models_dir = pathlib.Path("/media/luxc/Seagate Expansion Drive/Vice/02. Projects/THEIA/05. Models/customObjectDetection_210213/HELIOS_1/SavedModel_format/saved_model/quantized_model")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"detect.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
"""

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
checkpoint_path = '/datasets/BigLearning/haowen/autoLoss/saved_models/autoLosstoy/ffn-toy-0'
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)

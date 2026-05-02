# Install required Python libraries

# Code Execution Order

# Data Preprocessing and Saving  
<!-- For all_point in -all.txt, dental arch points need to be preprocessed and uniformly sampled to 512 points. Example:
ctr_point = get_line_data("1000813648_20180116-ctr.txt"):
all_point = sample_ctr_to_densePoint(ctr_point, sample_point = 512)
save all_point as "1000813648_20180116-all.txt" -->

python 1_data_preprocess.py

# Train, Validate, and Test the Model
python 2_train_TransArchNet.py

# Inference on External Data
# Data Path Parameters
# Test data
<!-- mesh_folder = r'CBCT_Mesh_data\test_data' -->
# # Manually specify the best model weight path
# model_path = TransFArchNet_checkpoints/checkpoints_best.pth

python 3_predict_TransArchNet.py
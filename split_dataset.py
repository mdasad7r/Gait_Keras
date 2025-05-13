import os
import shutil

# Define paths
DATASET_PATH = "/content/casia-b/output"
TRAIN_PATH = "/content/casia-b/train"
TEST_PATH = "/content/casia-b/test"

# Walking conditions
TRAIN_CONDITIONS = {"nm": ["nm-01", "nm-02", "nm-03", "nm-04"], "bg": ["bg-01"], "cl": ["cl-01"]}
TEST_CONDITIONS = {"nm": ["nm-05", "nm-06"], "bg": ["bg-02"], "cl": ["cl-02"]}

def move_condition_folders(subject_id, conditions, src_root, dst_root):
    src_subject_path = os.path.join(src_root, subject_id)
    dst_subject_path = os.path.join(dst_root, subject_id)
    os.makedirs(dst_subject_path, exist_ok=True)

    for _, condition_list in conditions.items():
        for condition in condition_list:
            src_cond_path = os.path.join(src_subject_path, condition)
            dst_cond_path = os.path.join(dst_subject_path, condition)

            if os.path.isdir(src_cond_path):
                os.makedirs(os.path.dirname(dst_cond_path), exist_ok=True)
                shutil.move(src_cond_path, dst_cond_path)

# Perform split
subject_ids = sorted(os.listdir(DATASET_PATH))

for subject_id in subject_ids:
    move_condition_folders(subject_id, TRAIN_CONDITIONS, DATASET_PATH, TRAIN_PATH)
    move_condition_folders(subject_id, TEST_CONDITIONS, DATASET_PATH, TEST_PATH)

print("✅ Dataset split complete — original video frame structure preserved.")

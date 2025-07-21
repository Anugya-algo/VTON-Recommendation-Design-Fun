import os
import shutil
from viton_backend import run_viton_hd, VitonHDOptions

# --- CONFIGURE THESE ---
# You can use a custom path for the person image (e.g., an uploaded or external image)
person_image_path = None  # Set to a path (e.g., '/path/to/custom/person.jpg') or None to use dataset
person = "14684_00.jpg"  # Base name for pose, parse, etc.
cloth = "14683_00.jpg"    # Cloth image filename (from dataset)
src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../clothes_tryon_dataset/train'))
dest_root = os.path.join(os.path.dirname(__file__), 'VITON-HD/datasets/single_test')

# --- PREPARE SINGLE TEST PAIR ---
def prepare_single_viton_dataset(person, cloth, src_root, dest_root, person_image_path=None):
    test_dir = os.path.join(dest_root, 'test')
    os.makedirs(test_dir, exist_ok=True)
    subdirs = ['image', 'cloth', 'cloth-mask', 'openpose-json', 'openpose-img', 'image-parse']
    for sub in subdirs:
        os.makedirs(os.path.join(test_dir, sub), exist_ok=True)

    # Copy person image (from custom path or dataset)
    if person_image_path is not None:
        shutil.copy(person_image_path, os.path.join(test_dir, 'image', person))
    else:
        shutil.copy(os.path.join(src_root, 'image', person), os.path.join(test_dir, 'image', person))
    # Copy cloth image
    shutil.copy(os.path.join(src_root, 'cloth', cloth), os.path.join(test_dir, 'cloth', cloth))
    # Copy cloth mask
    shutil.copy(os.path.join(src_root, 'cloth-mask', cloth), os.path.join(test_dir, 'cloth-mask', cloth))
    # Copy openpose keypoints (json)
    pose_json = person.replace('.jpg', '_keypoints.json')
    shutil.copy(os.path.join(src_root, 'openpose_json', pose_json), os.path.join(test_dir, 'openpose-json', pose_json))
    # Copy openpose rendered image (png)
    pose_img = person.replace('.jpg', '_rendered.png')
    shutil.copy(os.path.join(src_root, 'openpose_img', pose_img), os.path.join(test_dir, 'openpose-img', pose_img))
    # Copy parsing image (png)
    parse_img = person.replace('.jpg', '.png')
    shutil.copy(os.path.join(src_root, 'image-parse-v3', parse_img), os.path.join(test_dir, 'image-parse', parse_img))
    # Write test_pairs.txt in dest_root (not in test/)
    with open(os.path.join(dest_root, 'test_pairs.txt'), 'w') as f:
        f.write(f"{person} {cloth}\n")

if __name__ == "__main__":
    # Example: set person_image_path to a custom image if desired
    # person_image_path = '/path/to/custom/person.jpg'
    prepare_single_viton_dataset(person, cloth, src_root, dest_root, person_image_path=person_image_path)
    opt = VitonHDOptions(
        name="single_test",
        dataset_dir=dest_root + '/',
        checkpoint_dir=os.path.join(os.path.dirname(__file__), 'VITON-HD/checkpoints/'),
        save_dir=os.path.join(os.path.dirname(__file__), 'VITON-HD/results/')
    )
    result_files = run_viton_hd(opt)
    print("Result file(s):", result_files) 
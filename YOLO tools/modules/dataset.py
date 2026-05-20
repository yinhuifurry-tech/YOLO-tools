import os
import random
import shutil
import glob
from .base import BaseModule
from framework.events import Events


class DatasetManager(BaseModule):
    def __init__(self):
        super().__init__()
        self.dataset_dir = None

    def _find_images_labels(self):
        """Search for images/ and labels/ in cwd and subdirectories."""
        search_dirs = ['.', 'dataset', 'data', 'datasets', 'train_data']
        for base in search_dirs:
            imgs = os.path.join(base, 'images')
            lbls = os.path.join(base, 'labels')
            if os.path.isdir(imgs) and os.path.isdir(lbls):
                return imgs, lbls
        return None, None

    def _has_yolo_structure(self, base_dir):
        return (os.path.isdir(os.path.join(base_dir, 'train', 'images')) and
                os.path.isdir(os.path.join(base_dir, 'train', 'labels')) and
                os.path.isdir(os.path.join(base_dir, 'val', 'images')) and
                os.path.isdir(os.path.join(base_dir, 'val', 'labels')))

    def organize(self, images_dir='images', labels_dir='labels', dataset_dir='dataset', split_ratio=0.8):
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            imgs, lbls = self._find_images_labels()
            if imgs and lbls:
                images_dir, labels_dir = imgs, lbls
            else:
                expected = [
                    "  dataset/\n"
                    "    ├── train/\n"
                    "    │   ├── images/   ← .jpg/.png files\n"
                    "    │   └── labels/   ← .txt files\n"
                    "    └── val/\n"
                    "        ├── images/\n"
                    "        └── labels/\n"
                    "\nOr: images/ + labels/ at root, we auto-organize for you."
                ]
                return False, "Missing images/ or labels/ directories.\nExpected structure:\n" + "\n".join(expected)

        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif'):
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))

        image_names = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        label_names = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}
        matched = [(image_names[n], label_names[n]) for n in image_names if n in label_names]

        if not matched:
            return False, f"No matching image-label pairs found ({len(image_files)} images, {len(label_files)} labels)"

        train_images = os.path.join(dataset_dir, 'train', 'images')
        train_labels = os.path.join(dataset_dir, 'train', 'labels')
        val_images = os.path.join(dataset_dir, 'val', 'images')
        val_labels = os.path.join(dataset_dir, 'val', 'labels')
        for d in [train_images, train_labels, val_images, val_labels]:
            os.makedirs(d, exist_ok=True)

        random.shuffle(matched)
        split_idx = int(len(matched) * split_ratio)
        train_files = matched[:split_idx]
        val_files = matched[split_idx:]

        for img_path, lbl_path in train_files:
            shutil.copy2(img_path, os.path.join(train_images, os.path.basename(img_path)))
            shutil.copy2(lbl_path, os.path.join(train_labels, os.path.basename(lbl_path)))
        for img_path, lbl_path in val_files:
            shutil.copy2(img_path, os.path.join(val_images, os.path.basename(img_path)))
            shutil.copy2(lbl_path, os.path.join(val_labels, os.path.basename(lbl_path)))

        self.dataset_dir = dataset_dir
        return True, f"Organized {len(matched)} pairs: {len(train_files)} train, {len(val_files)} val"

    def upload_archive(self, archive_path, dest='dataset'):
        import zipfile
        import tarfile

        if not os.path.exists(archive_path):
            return False, f"Archive not found: {archive_path}"

        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.makedirs(dest, exist_ok=True)

        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(dest)
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tf:
                    tf.extractall(dest)
            elif archive_path.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:gz') as tf:
                    tf.extractall(dest)
            else:
                return False, f"Unsupported format: {archive_path}"
        except Exception as e:
            return False, f"Extraction failed: {e}"

        self.dataset_dir = dest

        for split in ['train', 'val']:
            sp = os.path.join(dest, split)
            if not os.path.isdir(sp):
                continue
            img_dir = os.path.join(sp, 'images')
            lbl_dir = os.path.join(sp, 'labels')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir, exist_ok=True)
                imgs = [f for f in os.listdir(sp) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                for f in imgs:
                    shutil.move(os.path.join(sp, f), os.path.join(img_dir, f))
            if not os.path.exists(lbl_dir):
                os.makedirs(lbl_dir, exist_ok=True)

        if self._has_yolo_structure(dest):
            return True, f"Dataset ready (YOLO structure): {dest}"

        imgs, lbls = self._find_images_labels()
        if imgs and lbls:
            self.organize(imgs, lbls, dest)
            if self._has_yolo_structure(dest):
                return True, f"Dataset auto-organized: {dest}"
            return True, f"Dataset extracted to {dest} (manual organize may be needed)"

        return True, f"Dataset extracted to {dest}. Use 'Organize' to split into train/val."

    def create_yaml(self, dataset_path):
        try:
            import yaml
        except ImportError:
            self._install('PyYAML', 'yaml')
            import yaml

        if not os.path.exists(dataset_path):
            return False, f"Dataset path not found: {dataset_path}"

        if not self._has_yolo_structure(dataset_path):
            imgs, lbls = self._find_images_labels()
            if imgs and lbls and not self._has_yolo_structure(dataset_path):
                ok, msg = self.organize(imgs, lbls, dataset_path)
                if not ok:
                    return False, f"Dataset structure incomplete and auto-organize failed: {msg}"

        train_path = os.path.join(dataset_path, 'train', 'images')
        val_path = os.path.join(dataset_path, 'val', 'images')
        test_path = os.path.join(dataset_path, 'test', 'images')

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            return False, (
                "Dataset structure incomplete. Expected:\n"
                f"  {dataset_path}/\n"
                "    ├── train/\n"
                "    │   ├── images/   ← .jpg/.png\n"
                "    │   └── labels/   ← .txt\n"
                "    └── val/\n"
                "        ├── images/\n"
                "        └── labels/\n"
                "\nUpload a dataset archive, or place images/ and labels/ in the project root and click 'Organize'."
            )

        labels_path = os.path.join(dataset_path, 'train', 'labels')
        class_ids = set()
        if os.path.exists(labels_path):
            for lf in glob.glob(os.path.join(labels_path, "*.txt")):
                try:
                    with open(lf, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except Exception:
                    continue
        nc = len(class_ids) if class_ids else 1
        names = [f"class_{i}" for i in sorted(list(class_ids))] if class_ids else ['object']

        data_config = {
            'path': os.path.abspath(dataset_path).replace('\\', '/'),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images' if os.path.exists(test_path) else '',
            'nc': nc,
            'names': names,
        }

        yaml_path = 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        return True, yaml_path

    def _install(self, package, import_name):
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

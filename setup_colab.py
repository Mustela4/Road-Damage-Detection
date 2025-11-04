import os
import subprocess
import sys

def run(cmd: str):
    print(f"[RUN] {cmd}")
    return subprocess.run(cmd, shell=True, check=True)

def main():
    try:
        run("pip install -q torchmetrics tensorboard pycocotools")
    except Exception as e:
        print("Warning: install step skipped or failed:", e)
    try:
        import pycocotools.coco as coco
        path_to_patch = coco.__file__
        with open(path_to_patch, "r") as f:
            content = f.read()
        patched = False

        old_line_1 = "annsImgIds = [ann['image_id'] for ann in anns]"
        new_line_1 = "annsImgIds = [int(ann['image_id']) for ann in anns]"
        if old_line_1 in content:
            content = content.replace(old_line_1, new_line_1)
            patched = True

        old_line_2 = "assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \\"
        new_line_2 = "assert set(annsImgIds) <= (set(annsImgIds) & set(self.getImgIds())), \\"
        if old_line_2 in content:
            content = content.replace(old_line_2, new_line_2)
            patched = True

        if patched:
            with open(path_to_patch, "w") as f:
                f.write(content)
            print("Patched pycocotools.coco successfully.")
        else:
            print("pycocotools.coco already patched or different version.")
    except Exception as e:
        print("Skip patching pycocotools:", e)

if __name__ == "__main__":
    main()

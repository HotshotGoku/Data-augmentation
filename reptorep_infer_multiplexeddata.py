import os, glob, cv2
import sys

# Import local pipeline BEFORE shared_resources_config to avoid path conflicts
import pipeline

# Now import shared_resources_config (adds original project to path)
import Data_augmentation.utils.shared_resources_config as shared_resources_config
from cldm.preprocess import preprocess_experimental_backgroundblack
import argparse
from Data_augmentation.utils.local_config import OUTPUT_DIR_REPTOREP, TEST_FOLDER_MULTIPLEXED_SENSING

p = argparse.ArgumentParser()
p.add_argument('--specific_folder', type=str, default='reptorep')
args = p.parse_args()


# ------------------------------
# Set up output folder using current date/time
# ------------------------------

OUTPUT_DIR = OUTPUT_DIR_REPTOREP
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Parameters and File Paths 
# ------------------------------


INPUT_DIR      = [TEST_FOLDER_MULTIPLEXED_SENSING]  # test images folder



os.makedirs(OUTPUT_DIR, exist_ok=True)

# fixed hyper-params:
ARGS = {
  "prompt":   "",
  "a_prompt": "",
  "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
  "num_samples":   2,
  "image_resolution":256,
  "ddim_steps":     50,
  "guess_mode":     False,
  "strength":       1.0,
  "scale":           9.0,  #15.1,
  "seed":           729397049,
  "eta":            0.0
}

# ------------------------------
# Pick one file per numeric prefix
# ------------------------------
prefix_map = {}

for INPUT_DIR in INPUT_DIR:
    for fp in sorted(glob.glob(os.path.join(INPUT_DIR, "*.TIF"))):
        # files are of type Repx_Rowy_Columnz.TIF, so for prefix we need to combine all three parts
        prefix = "_".join(os.path.basename(fp).split(".")[0].split("_")[:3])
        if prefix not in prefix_map:
            prefix_map[prefix] = fp


# ------------------------------
# Inference over each prefix
# ------------------------------
for prefix, fp in prefix_map.items():
    # load & convert to H×W×C RGB
    # img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)

    # process images with the simulation preprocessing step
    # img= preprocess_experimental_backgroundblack(fp)  
    img= cv2.imread(fp, cv2.IMREAD_COLOR)
    # source = cv2.imread(source_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    # run the shared pipeline
    outs = pipeline.process(img, **ARGS)

    # save as prefix_1.png … prefix_5.png
    for i, out in enumerate(outs, start=1):
        fn  = f"{prefix}_{i}.png"
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, fn), out_bgr)

    print(f"Done: {prefix}")
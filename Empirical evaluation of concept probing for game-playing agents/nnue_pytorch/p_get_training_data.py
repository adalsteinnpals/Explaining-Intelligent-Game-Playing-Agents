import os
import sys
import subprocess
import gdown

training_data_url = "https://drive.google.com/file/d/14wLaIw5vx38AgnQt0fs7HlIc2j0kLbB7/view?usp=sharing"

if not os.path.exists("training_data.binpack"):
    gdown.download(training_data_url, quiet=False,fuzzy=True)
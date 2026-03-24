# pipeline.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from stardist.models import StarDist2D
from umap import UMAP

# 1. Load a sample H&E image
img = imread("images/sample_image.tif")

# 2. Segment nuclei using StarDist
model = StarDist2D.from_pretrained('2D_versatile_fluo')
labels, _ = model.predict_instances(img)

# 3. Save masks
# You can save labels for later use
np.save("masks/sample_mask.npy", labels)

# 4. Load FDIM CSV features (already exported)
features = pd.read_csv("features/sample_features.csv")

# 5. t-SNE / UMAP embedding
embedding = UMAP(n_components=2).fit_transform(features.drop(['nucleus_id','label'], axis=1))

# 6. Plot
plt.scatter(embedding[:,0], embedding[:,1], c=features['label'].map({'intact':0,'BRCA':1}))
plt.show()
def segment_image(img_path):
    img = imread(img_path)
    labels, _ = model.predict_instances(img)
    return labels
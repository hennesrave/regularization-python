# Description
Python implementation of our GPU-based algorithm for scatterplot regularization.

# Installation (Ubuntu 22.04 LTS)
1. Simply run `setup.sh` or install the Vulkan SDK 1.3, Python 3.10 and the required Python packages (see below)
2. Execute the script `python3 regularization.py`

### Vulkan SDK
```sh
# Install Vulkan SDK
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
sudo apt update
sudo apt install -y vulkan-sdk
```

### Python
```sh
sudo apt-get install -y python3-pip
python3 -m pip install matplotlib numpy vulkan
```

# Installation (Windows)
1. Install the Vulkan SDK 1.3 from https://vulkan.lunarg.com/sdk/home#windows
2. Install Python 3.10 https://www.python.org/downloads/
3. Install the required python packages: `python3 -m pip install matplotlib numpy vulkan`
4. Execute the script: `python3 regularization.py`

# Citation
**Paper:** Hennes Rave, Vladimir Molchanov, Lars Linsen. "De-cluttering Scatterplots with Integral Images." IEEE Transactions on Visualization and Computer Graphics (2024): 1-13, DOI: https://www.doi.org/10.1109/tvcg.2024.3381453

**Abstract:** Scatterplots provide a visual representation of bivariate data (or 2D embeddings of multivariate data) that allows for effective analyses of data dependencies, clusters, trends, and outliers. Unfortunately, classical scatterplots suffer from scalability issues, since growing data sizes eventually lead to overplotting and visual clutter on a screen with a fixed resolution, which hinders the data analysis process. We propose an algorithm that compensates for irregular sample distributions by a smooth transformation of the scatterplot's visual domain. Our algorithm evaluates the scatterplot's density distribution to compute a regularization mapping based on integral images of the rasterized density function. The mapping preserves the samples' neighborhood relations. Few regularization iterations suffice to achieve a nearly uniform sample distribution that efficiently uses the available screen space. We further propose approaches to visually convey the transformation that was applied to the scatterplot and compare them in a user study. We present a novel parallel algorithm for fast GPU-based integral-image computation, which allows for integrating our de-cluttering approach into interactive visual data analysis systems.
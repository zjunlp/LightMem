# You may need to install git-lfs to download data properly

### Training data
# 1. Change to desired directory for storing data
DATA_BASEDIR="~/data/vlm2vec_train"
mkdir -p $DATA_BASEDIR
cd $DATA_BASEDIR

# 2.1 Download image data and unzip
git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-train
cd MMEB-train
python unzip_file.py

# 2.2 Download LLaVA-Hound video data
cd $DATA_BASEDIR
mkdir video
sudo apt-get install git-lfs
git clone https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/
# unzip frames in train_300k/ and train_600k/

# 3. Adjust the data path in the data config yaml, refer to `experiments/release_public/train/train_alltasks.yaml`


### Eval data
DATA_BASEDIR="~/data/vlm2vec_eval"
mkdir -p $DATA_BASEDIR
cd $DATA_BASEDIR

git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-V2
cd MMEB-V2/image-tasks
tar -xzvf mmeb_v1.tar.gz

cd ../video-tasks/frames
# extract frame files accordingly
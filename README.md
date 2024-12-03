
Ader环境：
pip3 install timm==0.8.15dev0 mmselfsup pandas transformers openpyxl imgaug numba numpy tensorboard fvcore accimage Ninja
pip3 install mmdet==2.25.3
pip3 install --upgrade protobuf==3.20.1 scikit-image faiss-gpu
pip3 install adeval
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install fastprogress geomloss FrEIA mamba_ssm adeval fvcore==0.1.5.post20221221
(or) conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia


配置realiad的数据路径：
在configs/mambaad/mambaad_realiad.py中设置self.data.root

训练：
参数设置好了，直接运行
python run.py

mambaad的训练日志好像是100轮，我多设置了50轮

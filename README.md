## Install
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install typing-extensions --upgrade # with ubuntu22.04 system pip

git clone https://github.com/chadwick-yao/Grounded-SAM-2.git

cd Grounded-SAM-2

pip install -e grounding_dino

pip install -e .

pip install git+https://github.com/huggingface/transformers

cd ../semantic_mapping/byte_track
pip install .

cd ../cython_bbox
pip install -e .

pip install supervision

pip install open3d

pip install spacy
python3 -m spacy download en_core_web_sm
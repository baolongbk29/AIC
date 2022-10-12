# INSTALL REQUIREMENTS

## Create Virtual Environment
```shell
python -m venv venv

source venv/Scripts/activate
```
```shell
# Intall a newer version of plotly
pip install plotly==4.14.3

# Install CLIP from the GitHub repo
pip install git+https://github.com/openai/CLIP.git

# Install torch 1.7.1 with GPU support
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install tensorflow
pip install tensorflow==2.1 

```
## INSTALL TransNetV2 
```shell
# Install transnet-V2 for Search with Image
git clone https://github.com/soCzech/TransNetV2.git
cd TransNetV2
pip install .
```

## INSTALL LAVIS
```shell
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install .
```
## INSTALL WITH REQUIREMENTS.TXT
```shell
pip install -r requirements.txt
```
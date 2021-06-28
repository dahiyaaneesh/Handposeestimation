# Instructions for installing pytorch and pytorch lightening.

There are few issue with newer version of pytorch and pytorch lightening because of missing modules and API changes. 
For reproducing and running the models in repo use following versions of these libraries. 
# with CUDA
```pytorch-lightning==1.0.8
pytorch-lightning-bolts==0.2.2
torch==1.7.0+cu110
torchaudio==0.7.0
torchvision==0.8.1+cu110
```
If you don't have GPU and want to proceed with cpu version than use the cpu counterpart. Here is the pip command.
Find more info on officail [pytorch page](https://pytorch.org/get-started/previous-versions/)
```
pip install torch==1.7.0+cpu \
                    torchvision==0.8.1+cpu  \
                    torchaudio==0.7.0 \
                    -f https://download.pytorch.org/whl/torch_stable.html```
                    

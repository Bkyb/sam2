# SAM2

*Annotation tool by using SAM2*

Forked from https://github.com/facebookresearch/sam2

### installation
```
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```

### Download checkpoint
*IF necessary*
```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### Usage
*Recommend to use Virtual Environment*
```
pip install streamlit>=1.40.0

# SAM2 app
cd ~/sam2
source venv/bin/activate
streamlit run app.py
```

```
# Kepoint labeling
# Different OpenCV vesrion (I'm not sure)

streamlit run streamlit_app.py
```




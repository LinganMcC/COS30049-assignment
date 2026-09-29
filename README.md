# COS30049-assignment

Install Python package
pip install pandas numpy spacy

Download additional dataset 
python download_hc3.py

Raw document proprocessing
python preprocess.py --drcat data/raw/train_v2_drcat_02.csv --hc3 data/raw/hc3_all.jsonl


Split processed documents into sentence
python sentence_splitter.py --processed-dir data/processed --out-dir data/sentences
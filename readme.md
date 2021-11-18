# 10kGNAD_GERMAN_NLP


 ## Repo Content
* 10kGNAD_classification.ipynb noteboook contains EDA + Modeling : 
   1. EDA
   2. Preprocessing
   3. Baseline Model: TF-IDF + Different models
   4. Transformer models: Bert-German finetuned
   
 * `Streamlit` APP with `FastAPI`
 
 ## How to RUN `Streamlit` APP

Clone this repo and run the below docker command:

`To Start Application:`
```docker
docker-compose up -d --build
```
and navigate to http://localhost:8501/

`To Stop Application:`
```docker
docker-compose down
```

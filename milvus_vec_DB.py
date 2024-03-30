from fastapi import FastAPI
from loguru import logger
import uvicorn
import pandas as pd
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import re
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility

from base import doc_ret

app = FastAPI()

@app.get('/')
def index(name:str):
    return f'hello {name}, Welcome to document Retrieval Web APP' 


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)

#-------------------------------------------Cleaning-----------------------------------
import pandas as pd
df = pd.read_csv('train.csv')
df = df.iloc[0:500,0]
df = pd.DataFrame(df)


import re
# Lowercasing the text
df['cleaned']=df['Story'].apply(lambda x:x.lower())

# Dictionary of english Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not","can't": "can not","can't've": "cannot have",
"'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
"didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
"hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
"he'll've": "he will have","how'd": "how did","how'd'y": "how do you","how'll": "how will","i'd": "i would",
"i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have",
"isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have",
"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
"mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
"needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
"oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
"shan't've": "shall not have","shed": "she would","she'd've": "she would have","she'll": "she will",
"she'll've": "she will have","should've": "should have","shouldn't": "should not",
"shouldn't've": "should not have","so've": "so have","that'd": "that would","that'd've": "that would have",
"there'd": "there would","there'd've": "there would have",
"they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
"they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",
"we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
"weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
"what've": "what have","when've": "when have","where'd": "where did",
"where've": "where have","who'll": "who will","who'll've": "who' will have","who've": "who have",
"why've": "why have","will've": "will have","won't": "will not","won't've": "will not have",
"would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you would","you'd've": "you would have","you'll": "you will","you'll've": "you will have",
"you're": "you are","you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

# Expanding Contractions
df['cleaned']=df['cleaned'].apply(lambda x:expand_contractions(x))


def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\S+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text
 
# Cleaning corpus using RegEx
df['cleaned']=df['cleaned'].apply(lambda x: clean_text(x))

df.drop("Story",axis = 1, inplace = True)

#-------------------------------------------Cleaning-----------------------------------


# Function to generate embeddings
def embeddings(text):
    return np.array(model(text)).flatten().tolist()


msgs = df.iloc[:, 0].tolist()
embdngs = [embeddings([x]) for x in msgs]
indx = list(range(1, len(msgs) + 1))
data_to_insert = [indx, msgs, embdngs]


connections.connect(
  alias="default",
  host='localhost',
  port='19530'
)

# Field Schema
id = FieldSchema(
  name="id",
  dtype=DataType.INT64,
  is_primary=True,
)
message = FieldSchema(
  name="message",
  dtype=DataType.VARCHAR,
  max_length=6000,
)
message_vec = FieldSchema(
  name="message_embeddings",
  dtype=DataType.FLOAT_VECTOR,
  dim=512
)
# collection schema
collection_schema = CollectionSchema(
  fields=[id, message, message_vec],
  description="Similar Document Retreival"
)
# Create collection
collection = Collection(
    name="SDR",
    schema=collection_schema,
    using='default')
utility.list_collections()



data_insert = collection.insert(data_to_insert)


# Create Index
index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024},
  "index_name": "SDR_IVF_FLAT_TEST"
}

# Index on vector field
collection.create_index(
  field_name="message_embeddings",
  index_params=index_params
)


# Load the collection
collection.load(replica_number=1)

@app.post('/predict')
def doc_retrieve(test_messages: doc_ret):
    test_message = test_messages
    
    test_message_vector = embeddings(test_message.test_messages)

    search_params = {"metric_type": "L2", "params": {"nprobe": 64}}

    results = collection.search(
        data=[test_message_vector],
        anns_field="message_embeddings",
        param=search_params,
        limit=10,
        expr=None,
        output_fields=['message']
    )

    response_data = []

    for result in results[0]:
        response_data.append(result)

    return {"results": response_data}


if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)



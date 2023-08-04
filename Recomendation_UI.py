import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def get_model_info(model_ID, device):
# Save the model to device
	model = CLIPModel.from_pretrained(model_ID).to(device)
 	# Get the processor
	processor = CLIPProcessor.from_pretrained(model_ID)
# Get the tokenizer
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)
       # Return model, processor & tokenizer
	return model, processor, tokenizer
 
@st.cache_data
def get_image_data(path):
    df = pd.read_csv(path,sep = '\t')
    df['img_embeddings'] = df['img_embeddings'].apply(lambda x: 
                           np.array([np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' ')]))
    return df[['image_url', 'img_embeddings']]

def get_single_text_embedding(text): 
    inputs = tokenizer(text, return_tensors = "pt")
    text_embeddings = model.get_text_features(**inputs)
    # convert the embeddings to numpy array
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np

def get_top_N_images(query, data, top_K=1, search_criterion="text"):
    # Text to image Search
    if(search_criterion.lower() == "text"):
        query_vect = get_single_text_embedding(query)
    # Image to image Search
    else:
        query_vect = query
    # Relevant columns
    revevant_cols = ["image_url", "img_embeddings", "cos_sim"]
    # Run similarity Search
    data["cos_sim"] = data["img_embeddings"].apply(lambda x: cosine_similarity(query_vect, x))# line 17
    data["cos_sim"] = data["cos_sim"].apply(lambda x: x[0][0])

    most_similar_articles = data.sort_values(by='cos_sim',  ascending=False)[1:top_K+1] # line 24
    return most_similar_articles[revevant_cols].reset_index()


# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"
model, processor, tokenizer = get_model_info(model_ID, device)

#Load Bird Image data
image_data_df = get_image_data('CUB_200_211_CLIP_Embeddings.tsv')
reco_image_df = pd.DataFrame([], columns=['image_url', 'img_embeddings'])

st.title('Bird Recommendation Engine')

with st.sidebar:
     
    title = st.text_input('Search for a type of bird:', 'black and white bird with short tail')
    top_images = get_top_N_images(title, image_data_df)
    st.image(Image.open(top_images.iloc[0].image_url))

new_reco_df = get_top_N_images(top_images.iloc[0].img_embeddings, image_data_df, top_K=15, search_criterion="image")
reco_image_df = pd.concat([reco_image_df,new_reco_df]).sample(frac=1)

st.image(reco_image_df.apply(lambda x : Image.open(x['image_url']).resize((325, 325), Image.ANTIALIAS), axis=1).values.tolist(), width=250)
# print(reco_image_df)


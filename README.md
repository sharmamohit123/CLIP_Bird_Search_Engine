# Bird recommendation engine

1. Implemented a search and recommendation engine for bird images.
2. Used **CUB_200_2011** dataset having 11,788 Images of around 200 Bird Species.
3. Used a pretrained **CLIP** model from huggingface to generate query and image embeddings.
4. Used inbuilt vector **cosine similarity** function from sklearn to power search and recommendation.
5. Used **Streamlit** tool to power up the user interface.

# Usage

1. Git clone the code on your system.
2. Install the required packages given in `requirements.txt` using `pip install -r requirements.txt`
3. Run the main Recomendation_UI.py app using `streamlit run Recomendation_UI.py`
4. Open your localhost endpoint, the webapp will look like below:
   ![alt text](https://github.com/sharmamohit123/CLIP_Bird_Search_Engine/blob/main/UI_SS.png?raw=true)
6. Play around with the app by entering different queries in the search box.


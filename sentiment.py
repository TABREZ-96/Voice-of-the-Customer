#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[33]:


df = pd.read_csv('E:\sent\Reviews.csv')
print(df.shape)
df=df.head(500)
print(df.shape)


# In[34]:


df.head()


# # QuicK EDA
# 

# In[35]:


ax = df['Score'].value_counts().sort_index()     .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10,5))
ax.set_xlabel('Review Stars')


# ## BASIC NLTK

# In[36]:


example = df['Text'][50]
print(example)


# In[45]:


tokens = nltk.word_tokenize(example)
tokens[:10]


# In[52]:


tagged= nltk.pos_tag(tokens)
tagged[:10]


# In[60]:


entities=nltk.chunk.ne_chunk(tagged)
entities.pprint()


# # VADER

# In[81]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


# In[73]:


sia.polarity_scores('I am so Happy')


# In[74]:


sia.polarity_scores('This is the worst thing ever')


# In[75]:


sia.polarity_scores(example)


# # RUn The polarity score

# In[80]:


res = {}
for i, row in tqdm(df.iterrows(),total=len(df)):
    text=row['Text']
    myid=row['Id']
    res[myid]=sia.polarity_scores(text)


# In[87]:


vaders=pd.DataFrame(res).T
vaders=vaders.reset_index().rename(columns={'index':'Id'})
vaders=vaders.merge(df,how='left')


# # sentiment score and metadat

# In[88]:


vaders.head()


# # Plot Vader Result

# In[94]:



ax=sns.barplot(data=vaders, x='Score',y='compound')
ax.set_title('Compound Scores by Amazon Star Review')
plt.show()


# In[102]:


fig,axs=plt.subplots(1,3,figsize=(12,3))
sns.barplot(data=vaders,x='Score',y='pos',ax= axs[0])
sns.barplot(data=vaders,x='Score',y='neu',ax=axs[1])
sns.barplot(data=vaders,x='Score',y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[110]:


get_ipython().system('pip install transformers')
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[ ]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[ ]:


print(example)
sia.polarity_scores(example)


# In[ ]:





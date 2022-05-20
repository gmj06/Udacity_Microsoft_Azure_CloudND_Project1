#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[1]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# ## Importing Azure Form Recognizer python modules

# In[2]:


from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient


# In[3]:


AZURE_FORM_RECOGNIZER_ENDPOINT = "https://proj1formrecognizer195668.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = "2ab67e45f11d49e49a3cfe41ea09de24"


# In[4]:


endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY


# In[5]:


form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# ### Source Document

# In[26]:


content_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/digital_id/ca-dl-gowri-chandrashekhar.png"


# In[27]:


id_content_from_url = form_recognizer_client.begin_recognize_identity_documents_from_url(content_url)


# ### Use the following if your source document is located the local disk

# In[28]:


collected_id_cards = id_content_from_url.result()


# In[29]:


collected_id_cards


# In[30]:


type(collected_id_cards[0])


# ## Processing the results

# In[31]:


def get_id_card_details(identity_card):
    first_name = identity_card.fields.get("FirstName")
    if first_name:
        print("First Name: {} has confidence: {}".format(first_name.value, first_name.confidence))
    last_name = identity_card.fields.get("LastName")
    if last_name:
        print("Last Name: {} has confidence: {}".format(last_name.value, last_name.confidence))
    document_number = identity_card.fields.get("DocumentNumber")
    if document_number:
        print("Document Number: {} has confidence: {}".format(document_number.value, document_number.confidence))
    dob = identity_card.fields.get("DateOfBirth")
    if dob:
        print("Date of Birth: {} has confidence: {}".format(dob.value, dob.confidence))
    doe = identity_card.fields.get("DateOfExpiration")
    if doe:
        print("Date of Expiration: {} has confidence: {}".format(doe.value, doe.confidence))
    sex = identity_card.fields.get("Sex")
    if sex:
        print("Sex: {} has confidence: {}".format(sex.value, sex.confidence))
    address = identity_card.fields.get("Address")
    if address:
        print("Address: {} has confidence: {}".format(address.value, address.confidence))
    country_region = identity_card.fields.get("CountryRegion")
    if country_region:
        print("Country/Region: {} has confidence: {}".format(country_region.value, country_region.confidence))
    region = identity_card.fields.get("Region")
    if region:
        print("Region: {} has confidence: {}".format(region.value, region.confidence))


# In[32]:


get_id_card_details(collected_id_cards[0])


# In[34]:


for index_id, id_card in enumerate(collected_id_cards):
    print("Displaying identity card details ....... # {}".format(index_id+1))
    get_id_card_details(id_card)
    print("---------------- EOL -------------------------")


# In[ ]:





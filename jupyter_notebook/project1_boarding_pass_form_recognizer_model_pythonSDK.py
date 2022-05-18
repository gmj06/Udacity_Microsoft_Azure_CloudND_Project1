#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[1]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# ### Importing Azure Form Recognizer Python modules

# In[2]:


import os
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.formrecognizer import FormTrainingClient
from azure.core.credentials import AzureKeyCredential


# ### TODO: Enter Form Recognizer endpoint and key and instantiate object

# In[3]:


AZURE_FORM_RECOGNIZER_ENDPOINT = "https://proj1formrecognizer195668.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = "2ab67e45f11d49e49a3cfe41ea09de24"


# In[4]:


endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY


# In[5]:


form_training_client = FormTrainingClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# In[6]:


saved_model_list = form_training_client.list_custom_models()


# ### TODO: Provide training source data URL

# In[13]:


trainingDataUrl = "https://proj1blobstorage195668.blob.core.windows.net/project1boardingpasstraining?sp=racwdli&st=2022-05-16T02:33:18Z&se=2022-05-18T10:33:18Z&spr=https&sv=2020-08-04&sr=c&sig=uN9RDMZPVivARD6SLqw66rBaofy3xQjlLYu313bbpcE%3D"


# ### TODO: Perform labeled traning
# * Use_training_labels=True
# * You need at least 5 labeled documents with the `ocr.json` and `labels.json` files; otherwise, you will run into errors.

# In[14]:


# Fill in the code to begin labeled training
training_process = form_training_client.begin_training(trainingDataUrl, use_training_labels=True)
custom_model = training_process.result()


# ### Get model info

# In[15]:


custom_model


# In[16]:


# Fill in the code to get model ID
custom_model.model_id


# In[17]:


# Fill in the code to get status
custom_model.status


# In[18]:


custom_model.training_started_on


# In[19]:


custom_model.training_completed_on


# In[20]:


# Fill in the code to get training documents
custom_model.training_documents


# In[21]:


for doc in custom_model.training_documents:
    print("Document name: {}".format(doc.name))
    print("Document status: {}".format(doc.status))
    print("Document page count: {}".format(doc.page_count))
    print("Document errors: {}".format(doc.errors))


# In[22]:


custom_model.properties


# In[23]:


custom_model.submodels


# In[24]:


for submodel in custom_model.submodels:
    print(
        "The submodel with form type '{}' has recognized the following fields: {}".format(
            submodel.form_type,
            ", ".join(
                [
                    field.label if field.label else name
                    for name, field in submodel.fields.items()
                ]
            ),
        )
    )


# In[25]:


custom_model.model_id


# In[26]:


custom_model_info = form_training_client.get_custom_model(model_id=custom_model.model_id)
print("Model ID: {}".format(custom_model_info.model_id))
print("Status: {}".format(custom_model_info.status))
print("Training started on: {}".format(custom_model_info.training_started_on))
print("Training completed on: {}".format(custom_model_info.training_completed_on))


# ### TODO: Perform prediction
# * Please download and save one of the PDFs not used in training, and then upload it to your Azure blob container.
# * After that, please create an Azure SAS URL with only READ access. Use the URL below to predict test/prediction with the model.

# In[27]:


new_test_url = "https://proj1blobstorage195668.blob.core.windows.net/project1boardingpasstraining/boarding_pass_rathna.pdf?sp=r&st=2022-05-16T03:10:25Z&se=2022-05-18T11:10:25Z&spr=https&sv=2020-08-04&sr=b&sig=IgKC%2FujNGIw6VEZkOSPmaHh9zvcOliwv1Nauz2kDAp8%3D"


# In[28]:


new_test_url


# In[29]:


form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# In[30]:


custom_model.model_id


# In[31]:


custom_model_info.model_id


# In[32]:


# Fill in code to begin prediction
custom_test_action = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=custom_model.model_id, form_url=new_test_url)


# In[33]:


custom_test_action.result


# In[34]:


custom_test_action.status()


# In[35]:


custom_test_action_result = custom_test_action.result()


# In[36]:


for recognized_content in custom_test_action_result:
    print("Form type: {}".format(recognized_content.form_type))
    for name, field in recognized_content.fields.items():
        print("Field '{}' has label '{}' with value '{}' and a confidence score of {}".format(
            name,
            field.label_data.text if field.label_data else name,
            field.value,
            field.confidence
        ))


# In[ ]:





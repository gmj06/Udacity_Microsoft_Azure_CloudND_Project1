#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[5]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# # Azure Custom Vision - Object Detection

# ## Import utility functions and Python modules 

# In[6]:


import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os, time, uuid


# In[7]:


def show_image_in_cell(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.show()


# In[8]:


from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials


# ### Resources:
# - Azure Custom Vision Endpoint
# - Training Reource ID and Key
# - Prediction Resource ID and Key

# ## Training and Prediction Endpoints, Keys and Resource IDs 

# In[9]:


TRAINING_ENDPOINT = "https://proj1customvision195668.cognitiveservices.azure.com/"
training_key = "d47f7271ec324a3fbebe2f182a7aeb78"
training_resource_id = '/subscriptions/8cf3eb54-2749-4fcf-bff9-057875575878/resourceGroups/ODL-AIND-195668/providers/Microsoft.CognitiveServices/accounts/proj1customvision195668'


# In[10]:


PREDICTION_ENDPOINT = 'https://proj1customvision195668-prediction.cognitiveservices.azure.com/'
prediction_key = "c7183a2a5e474c4c86220b6f03aa121b"
prediction_resource_id = "/subscriptions/8cf3eb54-2749-4fcf-bff9-057875575878/resourceGroups/ODL-AIND-195668/providers/Microsoft.CognitiveServices/accounts/proj1customvision195668-Prediction"


# ## Instantiate and authenticate the training client with endpoint and key 

# In[11]:


training_credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, training_credentials)


# In[12]:


trainer.api_version


# ## Instantiate and authenticate the prediction client with endpoint and key

# In[13]:


prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)


# In[14]:


predictor.api_version


# ## Creating Training Project First

# In[15]:


# Find the object detection domain
obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection" and domain.name == "General (compact) [S1]")

# Create a new project
print ("Your Object Detection Training project has been created. Please move on.")
project_name = uuid.uuid4()
project = trainer.create_project(project_name, domain_id=obj_detection_domain.id)


# ## Getting Project Details as collective information 

# In[16]:


project.as_dict()


# In[17]:


project.status


# ## Adding Tags based on training requirements
#   
# - In the demo, we used images and tags for Bird and Flower. For this exercise, you can use the "bottle" tag or any other class/type of your own images.
# - Please modify the code accordingly.

# In[18]:


lighter_tag = trainer.create_tag(project.id, "Lighter")


# ### Uploaded provided lighter images to the project on https://customvision.ai and labeled it manually

# ## Start the Object Detection Training
# - We will keep checking every 10 seconds during the training progress

# In[19]:


iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    print ("Waiting 10 seconds...")
    time.sleep(10)


# ## After training is complete, we will check model performance

# In[20]:


iteration.as_dict()


# In[21]:


iteration_list = trainer.get_iterations(project.id)
for iteration_item in iteration_list:
    print(iteration_item)


# In[22]:


model_perf = trainer.get_iteration_performance(project.id, iteration_list[0].id)


# In[23]:


model_perf.as_dict()


# ## Publishing the Model to the Project Endpoint

# In[24]:


## Setting the Iteration Name, this will be used when Model training is completed
publish_iteration_name = "project1-object-detection-lighter"


# In[25]:


# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")


# ## Performing Prediction
# - Using the predictor object 

# In[26]:


local_image_path = '/home/workspace/lighter_test_images'


# In[27]:


get_ipython().system('ls $local_image_path')


# In[28]:


def perform_prediction(image_file_name):
    with open(os.path.join (local_image_path,  image_file_name), "rb") as image_contents:
        results = predictor.detect_image(project.id, publish_iteration_name, image_contents.read())
        # Display the results.
        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                  ": {0:.2f}%".format(prediction.probability * 100))


# ### Prediction Test for lighter_test_set_1of5.jpg

# In[33]:


test_file_name1 = "lighter_test_set_1of5.jpg"


# In[34]:


perform_prediction(test_file_name1)


# In[36]:


## Checking the Image
with open(os.path.join (local_image_path, test_file_name1), 'rb') as img_code:
    img_view_ready = Image.open(img_code)
    plt.figure()
    plt.imshow(img_view_ready)


# ### Prediction Test for lighter_test_set_2of5.jpg

# In[37]:


test_file_name2 = "lighter_test_set_2of5.jpg"


# In[38]:


perform_prediction(test_file_name2)


# In[39]:


## Checking the Image
with open(os.path.join (local_image_path, test_file_name2), 'rb') as img_code:
    img_view_ready = Image.open(img_code)
    plt.figure()
    plt.imshow(img_view_ready)


# ### Prediction Test for lighter_test_set_3of5.jpg

# In[42]:


test_file_name3 = "lighter_test_set_3of5.jpg"


# In[43]:


perform_prediction(test_file_name3)


# In[44]:


## Checking the Image
with open(os.path.join (local_image_path, test_file_name3), 'rb') as img_code:
    img_view_ready = Image.open(img_code)
    plt.figure()
    plt.imshow(img_view_ready)


# ### Prediction Test for lighter_test_set_4of5.jpg

# In[45]:


test_file_name4 = "lighter_test_set_4of5.jpg"


# In[47]:


perform_prediction(test_file_name4)


# In[49]:


## Checking the Image
with open(os.path.join (local_image_path, test_file_name4), 'rb') as img_code:
    img_view_ready = Image.open(img_code)
    plt.figure()
    plt.imshow(img_view_ready)


# ### Prediction Test for lighter_test_set_5of5.jpg

# In[50]:


test_file_name5 = "lighter_test_set_5of5.jpg"


# In[51]:


perform_prediction(test_file_name5)


# In[52]:


## Checking the Image
with open(os.path.join (local_image_path, test_file_name5), 'rb') as img_code:
    img_view_ready = Image.open(img_code)
    plt.figure()
    plt.imshow(img_view_ready)


# ## Exporting Model 
# Note: This code is exact same as the Image classification model export. You just need to have the proper project ID and iteration ID specific to any custom vision solution, and the respective model from the selected iteration will be exported to the target platform and flavor. 

# In[53]:


platform = "TensorFlow"
flavor = "TensorFlowLite"


# ### Using the export_iteration method

# In[54]:


export_process = trainer.export_iteration(project.id, iteration.id, platform, flavor, raw=True)


# In[55]:


print(export_process.output)


# In[56]:


print(export_process.output.status)


# In[59]:


### Code snippet is from Azure SDK and Documentation
### https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/export-programmatically
### This step may take long time 
while (export_process.output.status == "Exporting"):
    print ("Waiting 10 seconds...")
    time.sleep(10)
    exports = trainer.get_exports(project.id, iteration.id)
    for e in exports:
        if e.platform == export_process.output.platform and e.flavor == export_process.output.flavor:
            export = e
            break
    print("Export status is: ", export_process.output.status)


# In[ ]:


print(export_process.output.status)


# In[ ]:


print(export_process.output.download_uri)


# ### You can choose any preferred name of the file download as exported model

# In[ ]:


## Downloading the model from url
if export_process.output.status == "Done":
    # Ready to Download
    model_export_file = requests.get(export_process.output.download_uri)
    with open("od_model_tensorflow.zip", "wb") as file:
        file.write(model_export_file.content)


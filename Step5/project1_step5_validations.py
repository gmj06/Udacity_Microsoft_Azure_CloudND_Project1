#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[1]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# ### Use the latest pillow package to show the rectangular bounding box around the face, so please upgrade the pillow package using the command below:

# ### Importing Useful Python Libraries or Packages

# In[2]:


import io
import datetime
import pandas as pd
from PIL import Image
import requests
import io
import glob, os, sys, time, uuid

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw

from video_indexer import VideoIndexer
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType
from msrest.authentication import CognitiveServicesCredentials
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.formrecognizer import FormTrainingClient
from azure.core.credentials import AzureKeyCredential


# ### Form Recognizer endpoint and key and instantiate object

# In[3]:


AZURE_FORM_RECOGNIZER_ENDPOINT = "https://proj1formrecognizer195668.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = "2ab67e45f11d49e49a3cfe41ea09de24"


# In[4]:


endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY


# In[5]:


form_training_client = FormTrainingClient(endpoint=endpoint, credential=AzureKeyCredential(key))
form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# ### STEP 2 

# #### Part 1 : Extracting data from ID  using pre-built model

# In[6]:


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


# In[7]:


def display_card_details(passenger_id_cards):
    for index_id, id_card in enumerate(passenger_id_cards):
        print("Displaying identity card details ....... # {}".format(index_id+1))
        get_id_card_details(id_card[0])
        print("---------------- EOL -------------------------")


# In[8]:


passenger_id_cards = []


# #### Passenger Akash Chauhan's ID

# In[9]:


avkash_id_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/digital_id/ca-dl-avkash-chauhan.png"
avkash_id_content_from_url = form_recognizer_client.begin_recognize_identity_documents_from_url(avkash_id_url)
avkash_id_card = avkash_id_content_from_url.result()
passenger_id_cards.append(avkash_id_card)
#type(avkash_id_card)


# #### Passenger James Jackson's ID

# In[10]:


james_id_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/digital_id/ca-dl-james-jackson.png"
james_id_content_from_url = form_recognizer_client.begin_recognize_identity_documents_from_url(james_id_url)
james_id_card = james_id_content_from_url.result()
passenger_id_cards.append(james_id_card)


# #### Passenger Libby Herold's ID

# In[11]:


libby_id_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/digital_id/ca-dl-libby-herold.png"
libby_id_content_from_url = form_recognizer_client.begin_recognize_identity_documents_from_url(libby_id_url)
libby_id_card = libby_id_content_from_url.result()
passenger_id_cards.append(libby_id_card)


# #### Passenger Gowri Chandrashekhar's ID

# In[12]:


gowri_id_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/digital_id/ca-dl-gowri-chandrashekhar.png"
gowri_id_content_from_url = form_recognizer_client.begin_recognize_identity_documents_from_url(gowri_id_url)
gowri_id_card = gowri_id_content_from_url.result()
passenger_id_cards.append(gowri_id_card)


# #### Passenger Radha S Kumar's ID

# In[13]:


radha_id_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/digital_id/ca-dl-radha-s-kumar.png"
radha_id_content_from_url = form_recognizer_client.begin_recognize_identity_documents_from_url(radha_id_url)
radha_id_card = radha_id_content_from_url.result()
passenger_id_cards.append(radha_id_card)


# In[14]:


display_card_details(passenger_id_cards)


# In[15]:


type(passenger_id_cards[0][0])


# #### Part 2- Data Extraction from Passenger Boarding Pass using Custom Model 
# 

# In[16]:


def get_boarding_pass_details(passenger_boarding_pass):
    for recognized_content in passenger_boarding_pass:
        print("Form type: {}".format(recognized_content.form_type))
        for name, field in recognized_content.fields.items():
            print("Field '{}' has label '{}' with value '{}' and a confidence score of {}".format(
                name,
                field.label_data.text if field.label_data else name,
                field.value,
                field.confidence
            ))


# In[17]:


def display_boarding_pass_details(passenger_boarding_passes):
    for index_id, boarding_pass in enumerate(passenger_boarding_passes):
        print("Displaying boarding pass details ....... # {}".format(index_id+1))
        get_boarding_pass_details(boarding_pass)
        print("---------------- EOL -------------------------")


# In[18]:


# Custom model generated in STEP 3 earlier
custom_model_id = '1eac902d-75c7-498f-80bb-70b2fbf70558'

custom_model_info = form_training_client.get_custom_model(model_id=custom_model_id)
print("Model ID: {}".format(custom_model_info.model_id))
print("Status: {}".format(custom_model_info.status))
print("Training started on: {}".format(custom_model_info.training_started_on))
print("Training completed on: {}".format(custom_model_info.training_completed_on))


# In[19]:


passenger_boarding_passes = []


# ### Boarding Pass data extraction for 5 passengers

# #### Passenger Avkash Chauhan's boarding pass 

# In[20]:


avkash_boarding_pass_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/boarding_pass/boarding-avkash.pdf?sp=r&st=2022-05-18T20:28:25Z&se=2022-05-21T04:28:25Z&spr=https&sv=2020-08-04&sr=b&sig=tlE2aDhVZENE6HaMnW7TKSaFKICz7yp7VWoOv9Phl3g%3D"
avkash_boarding_pass = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=custom_model_id, form_url=avkash_boarding_pass_url)
avkash_boarding_pass_data = avkash_boarding_pass.result()
passenger_boarding_passes.append(avkash_boarding_pass_data)


# #### Passenger James Jackson's boarding pass  

# In[21]:


james_boarding_pass_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/boarding_pass/boarding-james.pdf?sp=r&st=2022-05-18T20:35:44Z&se=2022-05-21T04:35:44Z&spr=https&sv=2020-08-04&sr=b&sig=zQcM03v6AsgnUDqeZb6zu5kNd%2FqcfpmrY4cX8e%2BffNA%3D"
james_boarding_pass = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=custom_model_id, form_url=james_boarding_pass_url)
james_boarding_pass_data = james_boarding_pass.result()
passenger_boarding_passes.append(james_boarding_pass_data)


# #### Passenger Libby Herold's boarding pass 

# In[22]:


libby_boarding_pass_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/boarding_pass/boarding-libby.pdf?sp=r&st=2022-05-18T20:37:53Z&se=2022-05-21T04:37:53Z&spr=https&sv=2020-08-04&sr=b&sig=wOvfx7njjuGcPBKJSxosybrFler7x5%2BSQy6DFeWipZs%3D"
libby_boarding_pass = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=custom_model_id, form_url=libby_boarding_pass_url)
libby_boarding_pass_data = libby_boarding_pass.result()
passenger_boarding_passes.append(libby_boarding_pass_data)


# #### Passenger Gowri Chandrashekhar's boarding pass 

# In[23]:


gowri_boarding_pass_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/boarding_pass/boarding_gowri.pdf?sp=r&st=2022-05-19T05:43:20Z&se=2022-05-21T13:43:20Z&spr=https&sv=2020-08-04&sr=b&sig=YqNNi3ZUoiqiWJOF3s%2B87bZDX7T%2Bkdmt%2Fg%2FignFo%2FAc%3D"
gowri_boarding_pass = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=custom_model_id, form_url=gowri_boarding_pass_url)
gowri_boarding_pass_data = gowri_boarding_pass.result()
passenger_boarding_passes.append(gowri_boarding_pass_data)


# #### Passenger Radha S Kumar's boarding pass 

# In[24]:


radha_boarding_pass_url = "https://proj1blobstorage195668.blob.core.windows.net/project1/boarding_pass/boarding-radha-s-kumar.pdf?sp=r&st=2022-05-18T20:43:36Z&se=2022-05-21T04:43:36Z&spr=https&sv=2020-08-04&sr=b&sig=hBAzhaXuw%2BcESFAq7WedyjKjEsSNOOpkdg7dzuCKGaM%3D"
radha_boarding_pass = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=custom_model_id, form_url=radha_boarding_pass_url)
radha_boarding_pass_data = radha_boarding_pass.result()
passenger_boarding_passes.append(radha_boarding_pass_data)


# In[25]:


display_boarding_pass_details(passenger_boarding_passes)


# ### Part 3 - Validations

# In[26]:


# Manifest tables from Azure blob storage
passenger_manifest_df = pd.read_csv('passenger_manifest_table.csv', index_col='FirstName', header=0)
passenger_manifest_df


# In[27]:


def validate_data(boarding_pass_data, id_card, firstName):
    boarding_pass_name = boarding_pass_data[0].fields["Passenger Name"].value
    id_firstname = id_card[0].fields['FirstName'].value
    id_lastname = id_card[0].fields['LastName'].value
    id_fullname = id_firstname + " " + id_lastname
    manifest_name = firstName + " "+ passenger_manifest_df.loc[firstName, "Last Name"]

    # Name Validation
    # For Passenger "Radha S Kumar", pre-built trained ID model for is 
    # reading the name as "Radha SKumar" as a result Name validation is failing for this passenger.
    passenger_manifest_df.loc[firstName, "NameValidation"] = [boarding_pass_name.lower() == id_fullname.lower() == manifest_name.lower()]

    #print(boarding_pass_name.lower())
    #print(id_fullname.lower())
    #print(manifest_name.lower())
    
    # DOB Validation
    passenger_manifest_df.loc[firstName, "DoBValidation"] = [pd.to_datetime(id_card[0].fields['DateOfBirth'].value) == pd.to_datetime(passenger_manifest_df.loc[firstName, "DateOfBirth"])]

    # Boarding Pass Validation
    boarding_pass_Carrier = boarding_pass_data[0].fields["Carrier"].value 
    boarding_pass_FlightNo = boarding_pass_data[0].fields["Flight Number"].value
    passenger_manifest_df.loc[firstName, "BoardingPassValidation"] = [(boarding_pass_Carrier + "-" + boarding_pass_FlightNo == passenger_manifest_df.loc[firstName, "Flight No"]  and 
    boarding_pass_data[0].fields["Seat Number"].value == passenger_manifest_df.loc[firstName, "SeatNo"] and 
    boarding_pass_data[0].fields["Class"].value == passenger_manifest_df.loc[firstName, "Class"] and 
    boarding_pass_data[0].fields["Flight Origin"].value == passenger_manifest_df.loc[firstName, "Origin"] and 
    boarding_pass_data[0].fields["Destination"].value == passenger_manifest_df.loc[firstName, "Destination"] and 
    pd.to_datetime(boarding_pass_data[0].fields["Flight Date"].value) == pd.to_datetime(passenger_manifest_df.loc[firstName, "Date"])and 
    boarding_pass_data[0].fields["Boarding Time"].value == passenger_manifest_df.loc[firstName, "Time"] )]

    # Person Validation
    # Based on Step 3: my digital_id (ca-dl-gowri-chandrashekhar.png) matches  with 
    # person model created based on thumbnail from gowri_boarding_pass_kiosk_video.mp4 
    # with 71.114%
    passenger_manifest_df.loc[firstName, "PersonValidation"] = True if (firstName == "Gowri")  else False

    # Luggage Validation
    # As we cannot validate the luggage setting it to "False" if Baggage  = 'YES' in boarding pass else "True" 
    passenger_manifest_df.loc[firstName, "LuggageValidation"] = False if (boarding_pass_data[0].fields["Baggage"].value == "YES")  else True
    
    
    return passenger_manifest_df


# In[28]:


validate_data(avkash_boarding_pass_data, avkash_id_card, "Avkash")
validate_data(james_boarding_pass_data, james_id_card, "James")
validate_data(libby_boarding_pass_data, libby_id_card, "Libby")
validate_data(gowri_boarding_pass_data, gowri_id_card, "Gowri")
validate_data(radha_boarding_pass_data, radha_id_card, "Radha")

# creating new .csv file with new updated values
passenger_manifest_df.to_csv("passenger_manifest_table_final.csv", index="FirstName", header=True)

passenger_manifest_df


# ### FINAL STATEMENT ON THE KIOSK

# In[62]:


def final_message(firstName):
    name_validation = passenger_manifest_df.loc[firstName, "NameValidation"]
    dob_validation = passenger_manifest_df.loc[firstName, "DoBValidation"]
    person_validation = passenger_manifest_df.loc[firstName, "PersonValidation"]
    boarding_pass_validation = passenger_manifest_df.loc[firstName, "BoardingPassValidation"]
    baggage_validation = passenger_manifest_df.loc[firstName, "LuggageValidation"]
    
    passenger_name = firstName + " "+ passenger_manifest_df.loc[firstName, "Last Name"]
    flight_no = passenger_manifest_df.loc[firstName, "Flight No"]   
    seat_no = passenger_manifest_df.loc[firstName, "SeatNo"]  
    origin = passenger_manifest_df.loc[firstName, "Origin"]  
    destination = passenger_manifest_df.loc[firstName, "Destination"]  
    boarding_time = passenger_manifest_df.loc[firstName, "Time"] 

    baggage_msg_success = '\nWe did not find a prohibited item (lighter) in your carry-on baggage. \nThanks for following the procedure.'
    baggage_msg_error = '\nWe have found a prohibited item in your carry-on baggage, and it is flagged for removal.'
    identity_error = '\nYour identity could not be verified. Please see a customer service representative.'
    identity_success = '\nYour identity is verified so please board the plane.'
    welcome_msg = '\nYou are welcome to flight # {0} leaving at {1} from {2} to {3}. \nYour seat number is {4}, and it is confirmed.'.format(flight_no, boarding_time, origin,  destination, seat_no)

    salutation_msg = 'Dear Mr/Mrs {0},'.format(passenger_name) 
    
    if(boarding_pass_validation == False):
        final_msg = 'Dear Sir/Madam,\nSome of the information in your boarding pass does not match the flight manifest data, so you cannot board the plane. \nPlease see a customer service representative.'
    elif(name_validation == False or dob_validation == False):
        final_msg = 'Dear Sir/Madam,\nSome of the information on your ID card does not match the flight manifest data, so you cannot board the plane. \nPlease see a customer service representative.'
    elif(baggage_validation == False):
        if(person_validation == True):
            final_msg = '{0}{1}{2}\nYour identity is verified. However, your baggage verification failed, so please see a customer service representative.'.format(salutation_msg,welcome_msg,baggage_msg_error)
        else:
            final_msg = '{0}{1}{2}{3}'.format(salutation_msg,welcome_msg,baggage_msg_error, identity_error)
    elif(person_validation == False):
        final_msg = '{0}{1}{2}{3}'.format(salutation_msg,welcome_msg, baggage_msg_success,identity_error)
    else:
        final_msg = '{0}{1}{2}{3}'.format(salutation_msg,welcome_msg, baggage_msg_success,identity_success)
        
    print(final_msg)
            


# In[63]:


final_message("Avkash")


# In[64]:


final_message("James")


# In[65]:


final_message("Libby")


# In[66]:


final_message("Gowri")


# In[67]:


final_message("Radha")


# In[ ]:





# In[ ]:





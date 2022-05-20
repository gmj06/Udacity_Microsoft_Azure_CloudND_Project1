#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[3]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# ### This demo uses the latest pillow package to show the rectangular bounding box around the face, so please upgrade the pillow package using the command below:

# In[4]:


get_ipython().system('pip install Pillow==8.4')


# ## Importing Useful Python Libraries or Packages 

# In[5]:


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


# In[8]:


CONFIG = {
    'SUBSCRIPTION_KEY': 'ebeef7d31fad40eba82c553b7f9de48b',
    'LOCATION': 'trial',
    'ACCOUNT_ID': '65c89254-4913-4947-a60d-1d886948ae01'
}

video_analysis = VideoIndexer(
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY'],
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID']
)


# ### Location Parameter
#  - For paid service, please use service region i.e. westus2, eastus, etc.
#  - For trial or free service, just use "trial" as I have used above. 

# In[9]:


video_analysis.check_access_token()


# ## Uploading .mp4 file from local path

# In[68]:


# Todo: the video id of uploaded video from local path
video_id = video_analysis.upload_to_video_indexer(
    input_filename='gowri_boarding_pass_kiosk_video.mp4',
    video_name='gowri-boarding-pass-kiosk-video',
    video_language='English'
)


# In[69]:


video_id


# In[70]:


video_analysis.get_video_info(video_id)


# In[71]:


info = video_analysis.get_video_info(video_id, video_language='English')


# ## Processing RAW Json 
# ### Getting a list of thumbnails where we find human face

# In[72]:


if len(info['videos'][0]['insights']['faces'][0]['thumbnails']):
    print("We found {} faces in this video.".format(str(len(info['videos'][0]['insights']['faces'][0]['thumbnails']))))


# In[73]:


info['videos'][0]['insights']['faces'][0]['thumbnails']


# ## Getting Thumbnail ID from the Analysis JSON 

# In[74]:


images = []
img_raw = []
img_strs = []
for each_thumb in info['videos'][0]['insights']['faces'][0]['thumbnails']:
    if 'fileName' in each_thumb and 'id' in each_thumb:
        file_name = each_thumb['fileName']
        thumb_id = each_thumb['id']
        img_code = video_analysis.get_thumbnail_from_video_indexer(video_id,  thumb_id)
        img_strs.append(img_code)
        img_stream = io.BytesIO(img_code)
        img_raw.append(img_stream)
        img = Image.open(img_stream)
        images.append(img)


# ## Now, let's view the face-specific thumbnails 

# In[75]:


for img in images:
    print(img.info)
    plt.figure()
    plt.imshow(img)


# ## Let's extract and save these face thumbnails to the local disk 
# - Download from Cloud

# In[76]:


i = 1
for img in images:
    print(type(img))
    img.save('human-face' + str(i) + '.jpg')
    i= i+ 1


# ## Verify the download process 

# In[77]:


get_ipython().system('ls human-face*.jpg')


# ## Getting thumbnail from the SDK 

# In[78]:


# Enter one of the thumbnail output you got from the previous cell, 
# under the "Getting Thumbnail ID from the Analysis JSON" section.
thumbnail_id='89b67340-68c1-4034-9d2f-cb6862b070a4'


# In[79]:


img_code = video_analysis.get_thumbnail_from_video_indexer(video_id,  thumbnail_id)
print(img_code)


# ## Converting encoded image to visible image

# In[80]:


img_code = video_analysis.get_thumbnail_from_video_indexer(video_id,  thumbnail_id)
img_stream = io.BytesIO(img_code)
img = Image.open(img_stream)
imshow(img)


# ## Getting thumbnails using Keyframes 

# In[81]:


keyframes = []
for shot in info["videos"][0]["insights"]["shots"]:
    for keyframe in shot["keyFrames"]:
        keyframes.append(keyframe["instances"][0]['thumbnailId'])


# In[82]:


for keyframe in keyframes:
    img_str = video_analysis.get_thumbnail_from_video_indexer(video_id,  keyframe)


# ## Emotion from the Video Analyzer

# In[83]:


info['summarizedInsights']['sentiments']


# In[84]:


info['summarizedInsights']['emotions']


# # Collecting Faces from  Video Analyzer

# In[85]:


GOWRI_FACE_KEY = "2aba6f0a4fec40d18ee5bf4daa46993f"
GOWRI_FACE_ENDPOINT = "https://proj1faceapi195668.cognitiveservices.azure.com/"


# In[86]:


# Create a client
face_client = FaceClient(GOWRI_FACE_ENDPOINT, CognitiveServicesCredentials(GOWRI_FACE_KEY))


# In[87]:


face_client.api_version


# # Creating Person Model Based on Faces in the Video

# In[88]:


PERSON_GROUP_ID = str(uuid.uuid4())
person_group_name = 'person-gowri'

# Note if this UUID already used earlier, you will get an error 


# In[89]:


## This code is taken from Azure Face SDK 
## ---------------------------------------
def build_person_group(client, person_group_id, pgp_name):
    print('Create and build a person group...')
    # Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
    print('Person group ID:', person_group_id)
    client.person_group.create(person_group_id = person_group_id, name=person_group_id)

    # Create a person group person.
    human_person = client.person_group_person.create(person_group_id, pgp_name)
    # Find all jpeg human images in working directory.
    human_face_images = [file for file in glob.glob('*.jpg') if file.startswith("human-face")]
    # Add images to a Person object
    for image_p in human_face_images:
        with open(image_p, 'rb') as w:
            client.person_group_person.add_face_from_stream(person_group_id, human_person.person_id, w)

    # Train the person group, after a Person object with many images were added to it.
    client.person_group.train(person_group_id)

    # Wait for training to finish.
    while (True):
        training_status = client.person_group.get_training_status(person_group_id)
        print("Training status: {}.".format(training_status.status))
        if (training_status.status is TrainingStatusType.succeeded):
            break
        elif (training_status.status is TrainingStatusType.failed):
            client.person_group.delete(person_group_id=PERSON_GROUP_ID)
            sys.exit('Training the person group has failed.')
        time.sleep(5)


# In[90]:


build_person_group(face_client, PERSON_GROUP_ID, person_group_name)


# # Making sure the Person model has faces and they all belong to the same person
# 

# In[91]:


'''
Detect all faces in query image list, then add their face IDs to a new list.
'''
def detect_faces(client, query_images_list):
    print('Detecting faces in query images list...')

    face_ids = {} # Keep track of the image ID and the related image in a dictionary
    for image_name in query_images_list:
        image = open(image_name, 'rb') # BufferedReader
        print("Opening image: ", image.name)
        time.sleep(5)

        # Detect the faces in the query images list one at a time, returns list[DetectedFace]
        faces = client.face.detect_with_stream(image)  

        # Add all detected face IDs to a list
        for face in faces:
            print('Face ID', face.face_id, 'found in image', os.path.splitext(image.name)[0]+'.jpg')
            # Add the ID to a dictionary with image name as a key.
            # This assumes there is only one face per image (since you can't have duplicate keys)
            face_ids[image.name] = face.face_id

    return face_ids


# In[92]:


test_images = [file for file in glob.glob('*.jpg') if file.startswith("human-face")]


# In[93]:


test_images


# In[95]:


ids = detect_faces(face_client, test_images)


# In[96]:


ids


# ### Verifying that 2 random images from the list belong to the same person
# - #### Note: So far we have not used the face recognition part, only face detection.

# In[97]:


# Verification example for faces of the same person.
verify_result = face_client.face.verify_face_to_face(ids['human-face1.jpg'], ids['human-face2.jpg'])


# In[98]:


if verify_result.is_identical:
    print("Faces are of the same (Positive) person, similarity confidence: {}.".format(verify_result.confidence))
else:
    print("Faces are of different (Negative) persons, similarity confidence: {}.".format(verify_result.confidence))


# ## Matching face from ID card with face from Video Analyzer 

# In[99]:


def show_image_in_cell(face_url):
    response = requests.get(face_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(10,5))
    plt.imshow(img)
    plt.show()


# In[100]:


dl_source_url = 'https://proj1blobstorage195668.blob.core.windows.net/project1/digital_id/ca-dl-gowri-chandrashekhar.png'


# In[101]:


show_image_in_cell(dl_source_url)


# In[102]:


## -------
## Reading file locally
## -------
# If I had image file locally, I would have used the following method
# dl_image = open('/Users/avkashchauhan99/work/avkash/udacity/cal-dl.png', 'rb')
# dl_faces = face_client.face.detect_with_stream(dl_image)  


# In[103]:


dl_faces = face_client.face.detect_with_url(dl_source_url) 


# ## Viewing Face ID and then saving it into the list of already saved Face IDs

# In[104]:


for face in dl_faces:
    print('Face ID', face.face_id, 'found in image', dl_source_url)
    # Add the ID to a dictionary with image name as a key.
    # This assumes there is only one face per image (since you can't have duplicate keys)
    ids['ca-dl-sample.png'] = face.face_id


# ## Now, we have ca-dl-sample.png with Face ID in our Face ID list

# In[105]:


ids


# ## Perform face verification between the Face ID from the identity card and one of the Face IDs from the video

# In[106]:


# Verification example for faces of the same person.
dl_verify_result = face_client.face.verify_face_to_face(ids['human-face1.jpg'], ids['ca-dl-sample.png'])


# In[107]:


if dl_verify_result.is_identical:
    print("Faces are of the same (Positive) person, similarity confidence: {}.".format(dl_verify_result.confidence))
else:
    print("Faces are of different (Negative) persons, similarity confidence: {}.".format(dl_verify_result.confidence))


# In[108]:


ids['ca-dl-sample.png']


# In[109]:


ids.values()


# In[110]:


dl_faces[0].face_rectangle.as_dict()


# In[111]:


# TAKEN FROM THE Azure SDK Sample
# Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    
    return ((left, top), (right, bottom))


# In[112]:


def drawFaceRectangles(source_file, detected_face_object) :
    # Download the image from the url
    response = requests.get(source_file)
    img = Image.open(BytesIO(response.content))
    # Draw a red box around every detected faces
    draw = ImageDraw.Draw(img)
    for face in detected_face_object:
        draw.rectangle(getRectangle(face), outline='red', width = 10)
    return img


# In[113]:


drawFaceRectangles(dl_source_url, dl_faces)


# ## Matching Face ID from the identity card with Video Analyzer Person Model 

# In[114]:


# A list of Face ID
ids


# ## Using the face ID from the identify card and matching the identity with the Person Group model

# In[115]:


# Enter the face ID of ca-dl-sample.png from the output of the cell above
get_the_face_id_from_the_driving_license = '4743e98b-cc35-46ee-889d-d007db62e1ac'


# In[116]:


person_gp_results = face_client.face.identify([get_the_face_id_from_the_driving_license], PERSON_GROUP_ID)


# In[117]:


for result in person_gp_results:
    for candidate in result.candidates:
        print("The Identity match confidence is {}".format(candidate.confidence))


# In[ ]:





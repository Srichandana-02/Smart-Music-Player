import streamlit as st
import cv2
import base64
#import matplotlib.pyplot as plt
#from deepface import DeepFace
import spotipy

from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import test
import random
#import vlc
from PIL import Image
#font

img=Image.open('./images/download.jpeg')
st.set_page_config(
    page_title="SMP",
    page_icon=img
)
#st.image('./images/mus.png')
# option = st.selectbox(
#     'Select any one from below',
#     ('automood','mood-selector'))
streamlit_style = """
			<style>
            @import url('https://fonts.googleapis.com/css2?family=Rancho&display=swap');
			html, body, [class*="css"]  {
            font-family: 'Rancho', cursive;
			font-size:1.5rem;
            }
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./images/smp.png')    
    

option1=0


client_id = '9a29ac56fc614fa5b98d8d106d22b790'
client_secret = 'd0c4cdd4489f4b0fa5dea7e13b5888bc'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager =client_credentials_manager)

header  = st.container()
inp = st.container()
pred = st.container()
st.write("Select any one from the following")
b=st.checkbox('automood-selector')
a=st.checkbox('mood-selector')
f=0
c=0
if a:
    #st.write('You selected:', )
    
    option1 = st.selectbox(
    'How are you feeling?',
    ('select','Angry', 'Happy', 'Sad','Neutral','Surprised'))
    st.write('You selected:',option1)
    f=1
    if option1=='select':
        c=1

if b:
    with header:
       
        st.title('Emotion Detection and Song Recommendation')
        st.markdown('**Aim : To detect the emotion of the person and predict a song**')

    with inp:
        
        st.title("Image Capture")
        st.markdown("**Capturing an image of your face**")
        
     
    with pred:
        count=1
        st.title("Let's see what songs you should listen to !!")
        dominant_emotion=test.take_input(count)
        option1=dominant_emotion
        print(option1)
        f=1

if c!=1:
    if f==1:
        if(option1 == 'Happy'):
            print("Happy emotion detected.")
            add_bg_from_local('./images/hap.jpg')    
            st.markdown("**You are happy!!**")
            playlist_id = '4jW37umAGFKr2oQRAk5pAe'
        elif(option1== 'Sad') :
            print("Sad emotion detected.")
            add_bg_from_local('./images/pq.png')    
            st.markdown("**You don't look too cheerful....Here are some songs to lift your mood up!!**")
            playlist_id = '72w1d4pSY1KYud4brnbh0F'
        elif(option1== 'Angry') :
            print("Angry emotion detected.")
            add_bg_from_local('./images/ang.jpeg')    
            st.markdown("**Angry!!**")
            playlist_id = '0ffnLxCftwLzmXDO7DJEXc'
        elif(option1== 'Surprised') :
            print("Surprise emotion detected.")
            add_bg_from_local('./images/sr.png')    
            st.markdown("**surprise!!**")
            playlist_id = '3xOiXGpE08faE5nAOcxZh5' 
        elif(option1=='Neutral'):
            st.markdown("neutral emotion detected!!")                        
            add_bg_from_local('./images/sri.avif')    
            playlist_id = '7a5IVfR0StcDsRRHjcpPwD'
        else:
            st.markdown("neutral emotion detected!!")                        
            add_bg_from_local('./images/neutral.png')    
            playlist_id = '7a5IVfR0StcDsRRHjcpPwD'






        def get_track_ids(playlist_id):
            music_id_list = []
            playlist = sp.playlist(playlist_id)

            for item in playlist['tracks']['items']:
                music_track = item['track']
                music_id_list.append(music_track['id'])
            return music_id_list 
        track_ids = get_track_ids(playlist_id)


        for i in range(5):

            random.shuffle(track_ids)

            my_html = '<iframe src="https://open.spotify.com/embed/track/{}" width="300" height="100" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'.format(track_ids[0])

            st.markdown(my_html, unsafe_allow_html=True)

else:     
    st.write("select emotion")
            

        

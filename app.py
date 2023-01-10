import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.keras as keras
import cv2

model = keras.models.load_model('model_1_ver3.h5')
true_dict = np.load('true_dict.npy', allow_pickle=True).item()
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

st.title("犬種予測")
st.text("""
        １２０種の犬種を学習済みモデルにより予測します。
        以下が予測可能犬種です。
        'Afghan_hound', 'African_hunting_dog', 'Airedale', 
        'American_Staffordshire_terrier', 'Appenzeller', 'Australian_terrier', 
        'Bedlington_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel', 
        'Border_collie', 'Border_terrier', 'Boston_bull', 'Bouvier_des_Flandres', 
        'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever', 
        'Chihuahua', 'Dandie_Dinmont', 'Doberman', 'English_foxhound', 'English_setter', 
        'English_springer', 'EntleBucher', 'Eskimo_dog', 'French_bulldog', 'German_shepherd', 
        'German_short', 'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 
        'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 
        'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 
        'Kerry_blue_terrier', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 
        'Maltese_dog', 'Mexican_hairless', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 
        'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian', 
        'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed', 'Scotch_terrier', 
        'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog', 'Shih', 'Siberian_husky', 
        'Staffordshire_bullterrier', 'Sussex_spaniel', 'Tibetan_mastiff', 'Tibetan_terrier', 
        'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier', 
        'Yorkshire_terrier', 'affenpinscher', 'basenji', 'basset', 'beagle', 'black', 'bloodhound', 
        'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn', 'chow', 'clumber', 
        'cocker_spaniel', 'collie', 'curly', 'dhole', 'dingo', 'flat', 'giant_schnauzer', 
        'golden_retriever', 'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute',
        'malinois', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'otterhound', 
        'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier', 'soft', 'standard_poodle', 
        'standard_schnauzer', 'toy_poodle', 'toy_terrier', 'vizsla', 'whippet', 'wire'
       
       JPG、PNGファイルの画像を入力すると、予測結果が出力されます。
       予測精度は約80%となっています。
       
       使用データセット：Stanford Dogs Dataset（http://vision.stanford.edu/aditya86/ImageNetDogs/main.html）
        """)
uploaded_file = st.file_uploader("画像アップロード", type=['jpg', 'png'])
if uploaded_file:
    image=Image.open(uploaded_file)
    st.image(image,caption = 'アップロード画像',use_column_width = True)
    image = np.array(image)
    height, width, color = image.shape

    if height > width:
        diffsize = height - width
        padding_half = int(diffsize / 2)
        padding_img = cv2.copyMakeBorder(image, 0, 0, padding_half, padding_half, cv2.BORDER_CONSTANT, (0, 0, 0))

    elif width > height:
        diffsize = width - height
        padding_half = int(diffsize / 2)
        padding_img = cv2.copyMakeBorder(image, padding_half, padding_half, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))

    else:
        padding_img = image
    
    image_resize = cv2.resize(padding_img, dsize=(300,300))
    # st.image(image_resize,caption = 'リサイズ画像',use_column_width = True)
    image_resize = image_resize / 255
    pred = model.predict(np.expand_dims(image_resize, axis=0))
    print(pred.max())
    print(np.argmax((pred)))
    p = get_keys_from_value(true_dict, np.argmax(pred))[0]
    st.text(f'予測した犬種：{p}')
    
    del pred
# 犬種予測アプリ
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
# NEPALI BARNA RECOGNITION (NBR)
## University Project

#### Simple Web Application For Detecting Nepali Alphabets ( speech to text )

NOTE: Projects Reports and mfcc detail file are attached to this project for understanding the projects pupose and
implementation.

PROJECT DESCRIPTION:
Aimed to create neural networks architecture that can precisely detect the nepali characters spoken in an 
  audio clip and convert that to text and display to the end user

FILES/FOLDER DESCRIPTION:

1. model1_aug.h5 => CNN model. 
2. model1_aug.png => CNN model architecture image.
3. print_model.py => script to generate image(.png) of CNN model architecture from the models (.h5).
4. record_windows.py => script to record audio clip  in windows machine. configure it according to your windows folder structure.
5. datasets (or their .npy files).
6. augmented datasets (or their .npy files).
7. nepaliBarnamala/ consists of:
    (.h5) model files ready for using to predict the audio clips,
    (.npy) files of datasets,
    mfcc calculation scripts,
    datasets training scripts,
    dataset (audio) recording scripts.
8. webframe_Project/ is the web application of this project. 
WEB APP INFO:
current web application uses model1_aug.h5 model.
you can change the model and use other models (.h5 files inside nepaliBarnamala folder) to check the 
accuracy/test other models.
ALSO you can train your own models and use that model in the web application.
current .npy files contains the audio of the project members thus the application will only give accurate results.
For the voice of the project members only.

Run server and check it out in web browser.

NOTE: Recommended to record your own datasets and train them using your own CNN model architecture and use it in the web application for prediction.

FOR MORE DETAIL CONSIDER THE REPORTS ATTACHED TO THE REPOSITORY.

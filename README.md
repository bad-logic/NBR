NEPALI BARNA RECOGNITION (NBR):

NOTE: Projects Reports and mfcc detail file are attached to this project for understanding the projects pupose and
implementation.

PROJECT DESCRIPTION:

Aimed to create neural networks architecture that can precisely detect the nepali characters spoken in an 
  audio clip and convert that to text and display to the end user

FILES/FOLDER DESCRIPTION:

1. model1_aug.h5 => CNN model. 
2. model1_aug.png => CNN model architecture image.
3. print_model.py => script to generate image(.png) of CNN model architecture from the models (.h5).
4. record_windows => script to record audio clip  in windows machine. configure it according to your windows folder structure.
5. datasets=> and their .npy files.
6. augmented datasets => and their .npy files.
7. nepaliBarnamala consists of:
    i. .h5 model files ready for using to predict the audio clips.
    ii. .npy files of datasets.
    iii.mfcc calculation scripts.
    iv. datasets training scripts.
    v. dataset (audio) recording scripts.
8. webframe_Project is the web application of this project. 
==> Just run server and check it out in web browser.
==> current web application uses model1_aug.h5 model (whose architecture can be printed with the python script          print_model.py).
==> you can change the model and use other models (.h5 files inside nepaliBarnamala folder) to check the 
accuracy of the models.
===> or train your own models and use that model in the web application.
===> current .npy files contains the audio of the project members thus the application will only give accurate results.
===> for the voice of the project members only.
===> Recommended to record your own datasets and train them using your CNN model architecture and use it in the web application.

NEPALI BARNA RECOGNITION (NBR):
NOTE: Projects Report and mfcc detail file are attached to this project for understanding the projects pupose and
implementation

PROJECT DESCRIPTION:
==> Aimed to create neural networks architecture that can precisely detect the nepali characters spoken in an 
==> audio clip and convert that to text and display to the end user

FILES/FOLDER DESCRIPTION:

model1_aug.h5 => CNN model 
model1_aug.png => CNN model architecture image
print_model.py => script to generate image(.png) of CNN model architecture from the models (.h5)
record_windows => script to record audio clip  in windows machine. configure it according to your windows folder structure
datasets=> and their .npy files
augmented datasets => and their .npy files
nepaliBarnamala consists of:
    .h5 model files ready for using to predict the audio clips
    .npy files of datasets
    mfcc calculation scripts
    datasets training scripts
    dataset (audio) recording scripts
webframe_Project is the web application of this project. Just run server and check it out in web browser
==> current web application uses model1_aug.h5 model (whose architecture can be printed with the python script          print_model.py).
==> you can change the model and use other models (.h5 files inside nepaliBarnamala folder) to check the 
accuracy of the models
===> or train your own models and change use that model in the web application
===> current .npy files contains the audio of the project members thus the application will only give accurate results
===> for the voice of the project members only
===> Recommended to record your own datasets and train them using your CNN model architecture and use it in the web application

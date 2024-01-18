# Phishing-Website-Detection-Using-Data-Mining-Techniques

# Overview

Phishing is a kind of cyberattack in which perpetrators deceive targets into divulging private information, such as bank account login passwords, by means of social engineering. These phishing attempts have the potential to cause serious harm, such as identity theft, money loss, and reputational harm. Even with a variety of defenses against phishing, it might be difficult to identify malicious URLs due to the growing complexity of attackers. By examining their traits and trends, data mining technologies are frequently used to identify phishing websites. These programs categorize newly created websites using algorithms that compare and contrast elements of legitimate and phishing websites. However, the current approaches frequently have large false positive rates and poor accuracy. This emphasizes the need for more accurate and practical methods of spotting phishing websites. The purpose of this project is to create a system for precise phishing site detection via data mining techniques. In order to complete this project, a sizable and varied dataset of authentic and fraudulent websites will be gathered. Their properties will then be extracted, and machine learning techniques will be used for analysis and classification. The ultimate outcome will be a system that can accurately detect phishing websites, strengthening defenses against malicious URLs and phishing attempts.

# Instructions to run the project

## Method: Run in Local Machine

[1] Download the project files from the data mining directory, by clicking the download present [here.](https://drive.google.com/drive/folders/1Aw1vaAMupU_F-OGiN-wsMdwpvwQAlOuS?usp=sharing)

[2] A directory was created named **code** folder in which it contains 3 python files named as **data-engineering**, **train-test**, **utils**.

[3] Select folder **datamining** folder. After downloading, save the file in desired folder and unzip the file (it will be named as **datamining**).

[4] Inside the **datamining** folder the **dataset**, **readme** file, **code** folder, **requirements** text file, **figures** folder are present.

[5] Open Terminal and run the following commands:
```
cd <FILE_PATH>
cd datamining
pip install -r requirements.txt
``` 
[6] All the required libraries will be downloaded.

[7] After executing the above command now lets run the next following command 
```
python code\data-engineering.py
``` 
[8] While **data-engineering** code is running all the outputs images are stored in the **Figures** Folder

[9] After executing the above command now lets run the next following command
```
python code\train-test.py
``` 
[10] These commands might take around 15 to 20 minutes to finish execution. After execution, output is displayed to the console.


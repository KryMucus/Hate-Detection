
# Fake News detection using BoW and Naive Bayes

Description
-----------
This project is designed to process and analyze datasets from multiple Excel files, transforming them into a machine learning model input for detecting fake tweets. The main part of the project is a script using a Multinomial Naive Bayes approach to predict whether a tweet is fake or not.

The accuracy is not great by any mean, but it is in line with the algorithm used, which were imposed.

Requirements
------------
You will need Python 3.x installed on your system along with the required packages listed in the `requirements.txt` file.

Installation
------------
1. Clone the repository or download the script.
2. Navigate to the project directory.
3. Install the required packages using the command:

   ```
   pip install -r requirements.txt
   ```

Usage
-----
To run the script, use the command:

```
python project.py <Argument File>
```

The argument file should contain a list of test tweets, one tweet per line. The script will output whether each tweet is "fake" or "not fake".

The jupyter notebook is just there to provide explanations on how everything works, and showcase the iterative steps that were required to create it.

Example
-------

![Alt text](image.png?raw=true "Title")


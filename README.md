Email Spam Detection
==============================
Email Spam Detection using Machine learning, Flask and Heroku 

Heroku app deployed is at https://email-abuse-detection.herokuapp.com/

Please extract following folders: Enron1, Enron2, Enron3, Enron4 ,Enron5, Enron6 in data/processed to train the model. 

Sample training output:
'''
➜  protonmail_spam git:(master) ✗ python app.py
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 972-451-267
Predict and Train
Generating spam and ham list
Naive Bays and wordbag
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
f1_score:
0.9814601259646945
'''

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- http://www2.aueb.gr/users/ion/data/enron-spam/ 
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebook with scores and analysis
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org



import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import process_tld, abnormal_url, httpSecure, digit_count, letter_count, shortening_service, having_ip_address
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


parent_dir = os.path.join(os.path.dirname(__file__), '..')
figures_path = os.path.join(parent_dir, 'figures')
data = pd.read_csv(os.path.join(parent_dir, 'modified_dataset.csv'))
X = data.drop(['url', 'type', 'Category', 'domain'], axis=1)
y = data['Category']

discrete_features = X.dtypes == int
mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)
print("\nMi Scores...")
print(mi_scores)

scores = mi_scores.sort_values(ascending=True)
width = np.arange(len(scores))
ticks = list(scores.index)
plt.barh(width, scores)
plt.yticks(width, ticks)
plt.title("MI Scores")
plt.figure(dpi=100, figsize=(7,7))
print('\nSaving MI Scores of features.. mi-scores-of-features.jpg')
plt.savefig(os.path.join(figures_path, 'mi-scores-of-features.jpg'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print('\nShapes of training and testing data subsets after split..')
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

def train_models():
    models = [DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, KNeighborsClassifier, SGDClassifier, GaussianNB]
    test_accuracy = []
    trained_models = []
    for m in models:
        print('\n#####################################################')
        print('Model =>\033[07m {} \033[0m'.format(m))
        if m == KNeighborsClassifier:
            model_ = m(n_neighbors=5)
        else:
            model_ = m()
        model_.fit(X_train, y_train)
        pred = model_.predict(X_test)
        trained_models.append(model_)
        acc = accuracy_score(pred, y_test)
        test_accuracy.append(acc)
        print('Test Accuracy :\033[32m \033[01m {:.2f}% \033[30m \033[0m'.format(acc*100))
        print('\033[01m              Classification_report \033[0m')
        print(classification_report(y_test, pred))
        cf_matrix = confusion_matrix(y_test, pred)
        plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt= '0.2%')
        plot_name = f"confusion-matrix-{str(m).split('.')[-1][:-2]}.jpg"
        print(f'\nSaving Confusion Matrix for {m()}.. {plot_name}')
        plt.savefig(os.path.join(figures_path, plot_name))
        print('\n\033[31m###################- End -###################\033[0m')
    return models, test_accuracy

def show_train_results(test_acc):
    output = pd.DataFrame({"Model": ['Decision Tree Classifier','Random Forest Classifier',
                                'AdaBoost Classifier','KNeighbors Classifier','SGD Classifier','Gaussian NB'],
                      "Accuracy": test_acc})
    plt.figure(figsize=(10, 5))
    plots = sns.barplot(x='Model', y='Accuracy', data=output)
    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.2f'),
                    (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                    size=15, xytext=(0, 8),
                    textcoords='offset points')
    print('\nSaving bar plot of accuracy of models.. accuracy-of-models.jpg')
    plt.xlabel("Models", size=14)
    plt.xticks(rotation=20)
    plt.ylabel("Accuracy", size=14)
    plt.savefig(os.path.join(figures_path, 'accuracy-of-models.jpg'))

def hyper_params_decision_tree():
    params = {
    'criterion': 'entropy',
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

    # Create Decision Tree Classifier with specified hyperparameters
    model = DecisionTreeClassifier(**params)

    # Fit the model and make predictions
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Evaluate and display results
    acc = accuracy_score(pred, y_test)
    print('\nTest Accuracy: {:.2f}%'.format(acc * 100))
    print(f'\nClassification Report: with hyperparameters [{params}]')
    print(classification_report(y_test, pred))
    print('\nSaving Confusion Matrix.. confusion-matrix-decisiontree-hyperparams.jpg')
    cf_matrix = confusion_matrix(y_test, pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True)
    plt.savefig(os.path.join(figures_path, 'confusion-matrix-decisiontree-hyperparams.jpg'))

def hyper_params_random_forest():
    params = {
    'n_estimators': 150,
    'criterion': 'entropy',
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

    model = RandomForestClassifier(**params)

    # Fit the model and make predictions
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Evaluate and display results
    acc = accuracy_score(pred, y_test)
    print('\nTest Accuracy: {:.2f}%'.format(acc * 100))
    print('\nClassification Report: with hyperparameters [{params}]')
    print(classification_report(y_test, pred))
    print('\nSaving Confusion Matrix ')
    cf_matrix = confusion_matrix(y_test, pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True)
    plt.savefig(os.path.join(figures_path, 'confusion-matrix-randomforest-hyperparams.jpg'))

def URL_Converter(urls):
    data = pd.DataFrame()
    data['url'] = pd.Series(urls)
    data['url_len'] = data['url'].apply(lambda x: len(str(x)))
    data['domain'] = data['url'].apply(lambda i: process_tld(i))
    feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
    for a in feature:
        data[a] = data['url'].apply(lambda i: i.count(a))
    data['abnormal_url'] = data['url'].apply(lambda i: abnormal_url(i))
    data['https'] = data['url'].apply(lambda i: httpSecure(i))
    data['digits_in_url']= data['url'].apply(lambda i: digit_count(i))
    data['letters_in_url']= data['url'].apply(lambda i: letter_count(i))
    data['Shortining_Service'] = data['url'].apply(lambda x: shortening_service(x))
    data['having_ip_address'] = data['url'].apply(lambda i: having_ip_address(i))
    print(data.columns)
    X = data.drop(['url','domain'],axis=1)
    return X

urls= ['diaryofagameaddict.com',
'espdesign.com.au',
'iamagameaddict.com',
'kalantzis.net',
'slightlyoffcenter.net',
'toddscarwash.com',
'tubemoviez.com',
'ipl.hk',
'crackspider.us/toolbar/install.php?pack=exe',
'pos-kupang.com/',
'rupor.info',
'svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt',
'officeon.ch.ma/office.js?google_ad_format=728x90_as',
'sn-gzzx.com',
'sunlux.net/company/about.html',
'outporn.com',
'timothycopus.aimoo.com',
'xindalawyer.com',
'freeserials.spb.ru/key/68703.htm',
'deletespyware-adware.com',
'orbowlada.strefa.pl/text396.htm',
'ruiyangcn.com',
'zkic.com',
'adserving.favorit-network.com/eas?camp=19320;cre=mu&grpid=1738&tag_id=618&nums=FGApbjFAAA',
'cracks.vg/d1.php',
'juicypussyclips.com',
'nuptialimages.com',
'andysgame.com',
'bezproudoff.cz',
'ceskarepublika.net',
'hotspot.cz',
'gmcjjh.org/DHL',
'nerez-schodiste-zabradli.com',
'nordiccountry.cz',
'nowina.info',
'obada-konstruktiwa.org',
'otylkaaotesanek.cz',
'pb-webdesign.net',
'pension-helene.cz',
'podzemi.myotis.info',
'smrcek.com',
'spekband.com',
'm2132.ehgaugysd.net/zyso.cgi?18',
'webcom-software.ws/links/?153646e8b0a88',
'worldgymperu.com',
'zgsysz.com',
'oknarai.ru',
'realinnovation.com/css/menu.js'
'http://pashminaonline.com/pure-pashminas']

def test_models(models):
    test_data = URL_Converter(urls)
    for m in models:
        print('#############################################')
        print('######-Model =>\033[07m {} \033[0m'.format(m))
        pred = m.predict(test_data)
        print(pred)

if __name__ == "__main__":
    print("\n[STARTING] ML Modeling")
    models, test_acc = train_models()
    end = time.time()
    print("\n[FINISHED] ML Modeling")
    print("\nTraining Results..")
    show_train_results(test_acc)
    hyper_params_decision_tree()
    hyper_params_random_forest()


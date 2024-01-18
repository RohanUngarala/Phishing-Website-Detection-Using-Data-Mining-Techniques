import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

def main():
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    os.makedirs(os.path.join(parent_dir, 'figures'), exist_ok=True)      
    data = pd.read_csv(os.path.join(parent_dir, "malicious_phish.csv"))
    figures_path = os.path.join(parent_dir, 'figures')
    print('\nSample dataset')
    print(data.head())
    print('\nDataset Info')
    print(data.info())

    print("\nChecking for null values...")
    print(data.isnull().sum())

    print("\nTarget Label Counts...")
    count = data.type.value_counts()
    print(count)

    sns.barplot(x=count.index, y=count)
    plt.xlabel('Types')
    plt.ylabel('Count')
    print('\nSaving count plot target labels.. count_plot.jpg')
    plt.savefig(os.path.join(figures_path, 'count_plot.jpg'))

    data['url'] = data['url'].replace('www.', '', regex=True)

    rem = {"Category": {"benign": 0, "defacement": 1, "phishing":2, "malware":3}}
    data['Category'] = data['type']
    data = data.replace(rem)
    data['url_len'] = data['url'].apply(lambda x: len(str(x)))
    data['domain'] = data['url'].apply(lambda i: process_tld(i))

    feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
    for a in feature:
        data[a] = data['url'].apply(lambda i: i.count(a))

    data['abnormal_url'] = data['url'].apply(lambda i: abnormal_url(i))

    sns.countplot(x='abnormal_url', data=data)
    print('\nSaving count plot of records with abnormal urls.. abnormal-url-counts.jpg')
    plt.savefig(os.path.join(figures_path, 'abnormal-url-counts.jpg'))

    data['https'] = data['url'].apply(lambda i: httpSecure(i))

    sns.countplot(x='https', data=data)
    print('\nSaving count plot of records based on http/https.. http-count-plot.jpg')
    plt.savefig(os.path.join(figures_path, 'http-count-plot.jpg'))

    data['digits_in_url']= data['url'].apply(lambda i: digit_count(i))
    data['letters_in_url']= data['url'].apply(lambda i: letter_count(i))
    data['shortening_service'] = data['url'].apply(lambda x: shortening_service(x))

    sns.countplot(x='shortening_service', data=data)
    print('\nSaving count plot of records with shortened urls.. shortening-count-plot.jpg')
    plt.savefig(os.path.join(figures_path, 'shortening-count-plot.jpg'))

    data['having_ip_address'] = data['url'].apply(lambda i: having_ip_address(i))
    print("\nRecords with and without IP addresses...")
    print(data['having_ip_address'].value_counts())

    plt.figure(figsize=(10, 10))
    sns.heatmap(data[data.describe().columns].corr(), linewidths=.5)
    plt.xticks(rotation=90)
    print('\nSaving heatmap.jpg ..!')
    plt.savefig(os.path.join(figures_path, 'heatmap.jpg'))

    print('\nSaving data engineered dataset locally.. modified_dataset.csv')
    data.to_csv(os.path.join(parent_dir, 'modified_dataset.csv'), index=False)

if __name__ == "__main__":
    print("[STARTING] Data Engineering")
    main()
    print("\n[FINISHED] Data Engineering")


import csv
import pickle
import sys
import numpy as np 
import nltk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
# from sklearn.naive_bayes import GaussianNB


def loadTrainData(symbol):
    PATH = './twitter/stocktwits/'
    ticker = symbol
    result = pickle.load(open(PATH + 'result/' + ticker + '_ratio.p', 'rb'))
    X = []
    y = []
    # print result
    interest = dict()
    with open(PATH + 'result/' + ticker + '_close.csv', 'rb') as googleclose:
        csvreader = csv.reader(googleclose)
        for row in csvreader:
            interest[row[0]] = float(row[4])
            # if float(row[7]) >= 0:
            #     interest[row[0]] = 1
            # else:
            #     interest[row[0]] = -1

    # print interest
    for i in range(2, len(result)):
    # for row in result:
        # if result[i][0] in interest:
        if result[i][0] in interest: 
            # print(row, interest[row[0]])
            # X.append([result[i - 2][0], result[i - 2][1], result[i - 2][2], 
            #         result[i - 1][0], result[i - 1][1], result[i - 1][2], 
            #         result[i][0], result[i][1], result[i][2]])
            # X.append([result[i - 2][1], result[i - 2][2], 
            #     result[i - 1][1], result[i - 1][2], 
            #     result[i][1], result[i][2]])
            # X.append([result[i - 9][1], result[i - 8][1], result[i - 7][1],
            #             result[i - 6][1], result[i - 5][1], result[i - 4][1], result[i - 3][1], 
            #             result[i - 2][1], result[i - 1][1], result[i][1]])
            # X.append([result[i - 2][1], result[i - 1][1], result[i][1],
            #             result[i - 2][2], result[i - 1][2], result[i][2]])  
            # X.append([result[i - 1][1], result[i][1]])
            # X.append([result[i][1], result[i][2]])
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            #               This strategy works best for twitter                  #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # if ticker == '@Amazon':
            #     # print result[i][1], result[i][2]
            #     X.append([result[i][1]]) # Good for Amazon
            # elif ticker == 'DJIA' or ticker == 'DJIA2' or ticker == '@Microsoft':
            #     X.append([result[i][2], result[i][1]])
            # else:
            #     X.append([result[i - 2][1], result[i - 1][1], result[i][1]]) # Good for Google and United
            #     # X.append([result[i][2]])
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            #               This strategy works best for stocktwits               #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # if ticker == '@Amazon' or ticker == '@United':
            X.append([result[i][2]])
            # else:
            #     X.append([result[i][2], result[i - 1][2], result[i - 2][2]])
            y.append(interest[result[i][0]])
    # print X, y
    return X, y

def dump_classifier(classifier, symbol):
    f = open(symbol + '_predictPrice_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

def main(argv):
    X, y = loadTrainData(argv[1])
    # X, y = loadTrainData('@Google')
    # print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.33, random_state = 42)
    # print (X_train, y_train)
    # print (X_test, y_test)
    classifier = SVR()
    # classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    print classifier.score(X_test, y_test)
    dump_classifier(classifier, argv[1])
    for i in range(len(X_test)):
        print classifier.predict(X_test[i])
        print y_test[i]

if __name__ == '__main__':
	main(sys.argv)
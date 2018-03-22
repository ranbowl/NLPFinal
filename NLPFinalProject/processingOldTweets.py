import sys, datetime
import csv
import pickle
import nltk
import re
from textblob import TextBlob
from nltk.stem import PorterStemmer

def main(argv):

    # PATH = './twitter/stocktwits/'
    PATH = './twitter/oldTweets/'
    ticker = argv[1]
    made_classifier = True # Keep True after first sentiment model training

    print('dealing with features... ')
    def word_feats(tmp, word_features):
            # document_words = set(tmp)
            features = {}
            for word in word_features:
                features['contains(%s)' % word] = (word in tmp)
            return features
    def get_word_features(tweets):
        all_words = []
        for (words, sentiment) in tweets:
          all_words.extend(words)
        wordlist = nltk.FreqDist(all_words)
        word_features = []
        for feature in wordlist:
            if wordlist[feature] > 500:
                word_features.append(feature)
        print(len(word_features))
        return word_features
    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
        	if word in document_words:
        		features['contains(%s)' % word] = (word in document_words)
        smileys = [':)',':-)',':o)',':P',':D',':o',':(',':-(',':o(',':-[',":'(",':o[',':[']
        for smiley in smileys:
        	if smiley in document_words:
        		features['contains(%s)' % smiley] = (smiley in document_words)
        return features
    def process_words(tweet): 
        words_filtered = []
        #Remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', tweet)
        #Remove citations
        tweet = re.sub(r'@[a-zA-Z0-9]*', '', tweet)
        #Remove tickers
        tweet = re.sub(r'\$[a-zA-Z0-9]*', '', tweet)
        #Remove numbers
        tweet = re.sub(r'[0-9]*','',tweet)
        tweet = re.sub('\n+','\n', tweet)
        tweet = re.sub('\t','',tweet)
        # #REMOVE PUNCTUATION
        # words = filter(lambda x: x not in string.punctuation,words)
        #PORTER STEMMER
        porter = PorterStemmer()
        # tknzr = TweetTokenizer()
        # words = tknzr.tokenize(words)
        # print words
        for e in tweet.split():
            e = e.lower()
            # e = porter.stem(e)
            words_filtered.append(e)
        return words_filtered
    train_pos_tweets=[]
    train_neg_tweets=[]
    test_pos_tweets=[]
    test_neg_tweets=[]
    with open('./twitter/twitter-train-pos.txt','r') as f:
        for line in f:
            train_pos_tweets.append((line,'pos'))
    f.close()
    with open('./twitter/twitter-train-neg.txt','r') as f:
        for line in f:
            train_neg_tweets.append((line,'neg'))
    f.close()
    with open('./twitter/twitter-test-pos.txt','r') as f:
        for line in f:
            test_pos_tweets.append((line,'pos'))
    f.close()
    with open('./twitter/twitter-test-neg.txt','r') as f:
        for line in f:
            test_pos_tweets.append((line,'neg'))
    f.close()
    train_tweets = []
    test_tweets = []
    for (words, sentiment) in train_pos_tweets + train_neg_tweets:
        words_filtered = process_words(words)
        train_tweets.append((words_filtered, sentiment))
    for (words, sentiment) in test_pos_tweets + test_neg_tweets:
        words_filtered = process_words(words)
        test_tweets.append((words_filtered, sentiment))
    word_features = get_word_features(train_tweets+test_tweets)
    print("Word Features Generated")

    if made_classifier:
        print("Loading classifier...")
        f = open('twitter_classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
    else:
        training_set = nltk.classify.apply_features(extract_features, train_tweets)
        test_set = nltk.classify.apply_features(extract_features, test_tweets)
        print("Training and Test Sets Created")
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        print("Model generated")
        f = open('twitter_classifier.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()

    poscount = {}
    negcount = {}
    tbposcount = {}
    tbnegcount = {}

    print('counting sentiment... ')
    with open(PATH + ticker + '.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row['date'], row['text'])
            tweet = row['text']
            tweet2 = tweet.decode('utf-8')
            analysis = TextBlob(tweet2).sentiment
            # Textblob counts below
            if analysis.polarity > 0:
                if row['date'] not in tbposcount:
                    tbposcount[row['date']] = 1
                else:
                    tbposcount[row['date']] += 1
            if analysis.polarity < 0:
                if row['date'] not in tbnegcount:
                    tbnegcount[row['date']] = 1
                else:
                    tbnegcount[row['date']] += 1
            # our sentiment counts below
            words_filtered = process_words(tweet)
            unsup_feat = word_feats(words_filtered, word_features)
            res = classifier.classify(unsup_feat)
            if res == 'pos':
                if row['date'] not in poscount:
                    poscount[row['date']] = 1
                else:
                    poscount[row['date']] += 1
            elif res == 'neg':
                if row['date'] not in negcount:
                    negcount[row['date']] = 1
                else:
                    negcount[row['date']] += 1

    # print(poscount['2017-04-19'])
    # print(negcount['2017-04-19'])
    # print(tbposcount['2017-04-19'])
    # print(tbnegcount['2017-04-19'])
    
    print('formatting result...')
    result = []
    result2 = dict()
    ratio = []
    ratio2 = dict()
    for date in poscount:
        pos = poscount[date]
        if date in negcount:
            neg = negcount[date]
        else:
            neg = 0
        if date in tbposcount:
            tbpos = tbposcount[date]
        else:
            tbpos = 0
        if date in tbnegcount:
            tbneg = tbnegcount[date]
        else:
            tbneg = 0
        # print date 
        # print pos
        # print neg
        # print tbpos
        # print tbneg
        result.append([date, pos, neg, tbpos, tbneg])
        result2[date] = [pos, neg, tbpos, tbneg]
        if neg == 0:
            neg = 1
        if tbneg == 0:
            tbneg = 1
        ratio.append([date, float(pos) / float(neg), float(tbpos) / float(tbneg)])
        ratio2[date] = ([float(pos) / float(neg), float(tbpos) / float(tbneg)])
    # print result
    # print result2
    print('dump result...')
    pickle.dump(result, open(PATH + 'result/' + ticker + '_result.p', 'wb' ))
    print('dump result2...')
    pickle.dump(result2, open(PATH + 'result/' + ticker + '_result2.p', 'wb' ))
    print('dump ratio...')
    pickle.dump(ratio, open(PATH + 'result/' + ticker + '_ratio.p', 'wb'))
    print('dump ratio2...')
    pickle.dump(ratio2, open(PATH + 'result/' + ticker + '_ratio2.p', 'wb'))

if __name__ == '__main__':
    	main(sys.argv)

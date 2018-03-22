import nltk
import pickle
import tweepy
from textblob import TextBlob

consumer_key= 'TK1J7ldeLmRV3pBQDXB3XlGWC'
consumer_secret= 'rnvocBX4vOF6dwZ3KL1CRBNhk1ZIhYyMqIEf9SLSRhHMKZJDhe'
access_token='854562630517870593-OVBVi017LmlDDBD4Kfz89Z7s1ONvJjs'
access_token_secret='CAGTIVrDKBsixUuBclX812vPufdtcgzArposRyg8NIyGt'

made_classifier = True 

def get_word_features(tweets):
        all_words = []
        for (words, sentiment) in tweets:
          all_words.extend(words)
        wordlist = nltk.FreqDist(all_words)
        word_features = []
        for feature in wordlist:
            if wordlist[feature] > 10000:
                word_features.append(feature)
        print(len(word_features))
        return word_features

# def extract_features(document):
#         document_words = set(document)
#         features = {}
#         for word in word_features:
#             features['contains(%s)' % word] = (word in document_words)
#         return features

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

# for (words, sentiment) in train_pos_tweets + train_neg_tweets:
#     words_filtered = [e.lower() for e in words.split()]
#     train_tweets.append((words_filtered, sentiment))

# for (words, sentiment) in test_pos_tweets + test_neg_tweets:
#     words_filtered = [e.lower() for e in words.split()]
#     test_tweets.append((words_filtered, sentiment))

for (words, sentiment) in train_pos_tweets + train_neg_tweets:
    	words_filtered = []
	#Remove hyperlinks
	words = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', words)
	#Remove citations
	words = re.sub(r'@[a-zA-Z0-9]*', '', words)
	#Remove tickers
	words = re.sub(r'\$[a-zA-Z0-9]*', '', words)
	#Remove numbers
	words = re.sub(r'[0-9]*','',words)
	words = re.sub('\n+','\n', words)
	words = re.sub('\t','',words)
	# #REMOVE PUNCTUATION
	# words = filter(lambda x: x not in string.punctuation,words)
	#PORTER STEMMER
	porter = PorterStemmer()
	# tknzr = TweetTokenizer()
	# words = tknzr.tokenize(words)
	# print words
	for e in words.split():
		e = e.lower()
		# e = porter.stem(e)
		words_filtered.append(e)
	train_tweets.append((words_filtered, sentiment))
for (words, sentiment) in test_pos_tweets + test_neg_tweets:
	words_filtered = []
	#Remove hyperlinks
	words = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', words)
	#Remove citations
	words = re.sub(r'@[a-zA-Z0-9]*', '', words)
	#Remove tickers
	words = re.sub(r'\$[a-zA-Z0-9]*', '', words)
	#Remove numbers
	words = re.sub(r'[0-9]*','',words)
	words = re.sub('\n+','\n', words)
	words = re.sub('\t','',words)
	# #REMOVE PUNCTUATION
	# words = filter(lambda x: x not in string.punctuation,words)
	#PORTER STEMMER
	porter = PorterStemmer()
	# tknzr = TweetTokenizer()
	# words = tknzr.tokenize(words)
	# print words
	for e in words.split():
		e = e.lower()
		# e = porter.stem(e)
		words_filtered.append(e)
	test_tweets.append((words_filtered, sentiment))

print("Tweets filtered")

word_features = get_word_features(train_tweets+test_tweets)
print("Word Features Generated")

if made_classifier:
    print("Loading classifier...")
    f = open('twitter_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
else:
    print("Training classifier...")
    
    training_set = nltk.classify.apply_features(extract_features, train_tweets)
    test_set = nltk.classify.apply_features(extract_features, test_tweets)
    print("Training and Test Sets Created")

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Model generated")

    f = open('twitter_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

print("Model loaded")

# accuracy = nltk.classify.accuracy(classifier,test_set)

# print(accuracy)




# with open('1.txt','r') as f:
#     for line in f:
#         unsup_tweets.append(line)
# f.close()

# def word_feats(tmp, word_features):
#         # document_words = set(tmp)
#         features = {}
#         for word in word_features:
#             features['contains(%s)' % word] = (word in tmp)
#         return features


def word_feats(document_words, word_features):
    features = {}
    document_words = set(document_words)
    for word in word_features:
        if word in document_words:
            features['contains(%s)' % word] = (word in document_words)
        smileys = [':)',':-)',':o)',':P',':D',':o',':(',':-(',':o(',':-[',":'(",':o[',':[']
        for smiley in smileys:
            if smiley in document_words:
                features['contains(%s)' % smiley] = (smiley in document_words)
    return features

# pos , neg = 0, 0
# with open('twitter_unsup.txt','r') as f:
#     for line in f:
#         words_filtered = [e.lower() for e in line.split()]
#         unsup_feat = word_feats(words_filtered, word_features)
#         res = classifier.classify(unsup_feat)
#         # print res
#         if res == 'pos':
#             pos += 1
#         elif res == 'neg':
#             neg += 1
# f.close()
# print pos
# print neg

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitter_api = tweepy.API(auth)

symbol = "@United"

tickers = [symbol + " since:2017-04-20 until:2017-04-21", 
            symbol + " since:2017-04-19 until:2017-04-20",
            symbol + " since:2017-04-18 until:2017-04-19",
            symbol + " since:2017-04-17 until:2017-04-18",
            symbol + " since:2017-04-16 until:2017-04-17", 
            symbol + " since:2017-04-15 until:2017-04-16",
            symbol + " since:2017-04-14 until:2017-04-15",
            symbol + " since:2017-04-13 until:2017-04-14"]

def filter_words(tweet):
    words_filtered = []
    #Remove hyperlinks
    words = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', words)
    #Remove citations
    words = re.sub(r'@[a-zA-Z0-9]*', '', words)
    #Remove tickers
    words = re.sub(r'\$[a-zA-Z0-9]*', '', words)
    #Remove numbers
    words = re.sub(r'[0-9]*','',words)
    words = re.sub('\n+','\n', words)
    words = re.sub('\t','',words)
    # #REMOVE PUNCTUATION
    # words = filter(lambda x: x not in string.punctuation,words)
    #PORTER STEMMER
    porter = PorterStemmer()
    # tknzr = TweetTokenizer()
    # words = tknzr.tokenize(words)
    # print words
    for e in words.split():
        e = e.lower()
        # e = porter.stem(e)
        words_filtered.append(e)
    return words_filtered

for ticker in tickers:
    # fetch tweets for user-entered stock ticker
    tweets = twitter_api.search(ticker, count=1)
    positive_tweets, negative_tweets, neural_tweets = 0, 0, 0
    pre_array = []
    pre_label = []
    page = 0
    pos, neg, total = 0, 0, 0
    for tweets in tweepy.Cursor(twitter_api.search, q = ticker, count = 100, result_type = "recent").pages():
        page = page + 1
        for tweet in tweets:
            total += 1
            analysis = TextBlob(tweet.text).sentiment
            if analysis.polarity > 0:
                positive_tweets += 1
                pre_array.append(tweet.text.encode('utf-8'))
                pre_label.append(1)
            if analysis.polarity < 0:
                negative_tweets += 1
                pre_array.append(tweet.text.encode('utf-8'))
                pre_label.append(0)
            words_filtered = filter_words(tweet.text.encode('utf-8'))
            unsup_feat = word_feats(words_filtered, word_features)
            res = classifier.classify(unsup_feat)
            # print res
            if res == 'pos':
                pos += 1
            elif res == 'neg':
                neg += 1
        if(page > 1) :
            break
    print ticker
    print total
    print("By TextBlob: ")
    print positive_tweets
    print negative_tweets
    print("By us: ")
    print pos
    print neg
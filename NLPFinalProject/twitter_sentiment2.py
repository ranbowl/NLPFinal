import pickle
import re
import string
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from textblob import TextBlob
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

# def extract_features(document):
#         document_words = set(document)
#         features = {}
#         for word in word_features:
#             features['contains(%s)' % word] = (word in document_words)
        
#         smileys = [':)',':-)',':o)',':P',':D',':o',':(',':-(',':o(',':-[',":'(",':o[',':[']
#         for smiley in smileys:
#             features['contains(%s)' % smiley] = (smiley in document_words)
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
# 	words_filtered = []
# 	words = re.sub('\n+','\n', words)
# 	words = re.sub('\t','',words)
# 	for e in words.split():
# 		e = e.lower()
# 		if "@" in e:
# 			continue
# 		words_filtered.append(e)
# 	train_tweets.append((words_filtered, sentiment))

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
# print train_tweets


# for (words, sentiment) in test_pos_tweets + test_neg_tweets:
# 	words_filtered = []
# 	words = re.sub('\n+','\n', words)
# 	words = re.sub('\t','',words)
# 	for e in words.split():
# 		e = e.lower()
# 		if "@" in e:
# 			continue
# 		words_filtered.append(e)
# 	test_tweets.append((words_filtered, sentiment))
print("Tweets filtered")

word_features = get_word_features(train_tweets+test_tweets)
print("Word Features Generated")

training_set = nltk.classify.apply_features(extract_features, train_tweets)
test_set = nltk.classify.apply_features(extract_features, test_tweets)
print("Training and Test Sets Created")

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Model generated")

f = open('twitter_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

# f = open('twitter_classifier.pickle', 'rb')
# classifier = pickle.load(f)
# f.close()

# print("Model loaded")

accuracy = nltk.classify.accuracy(classifier,test_set)

print(accuracy)

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

unsup_tweets = []
with open('2.txt','r') as f:
    for line in f:
        unsup_tweets.append(line)
f.close()

newdata = []
for words in unsup_tweets:
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
  newdata.append(words_filtered)

pos , neg = 0, 0
for data in newdata :
	unsup_feat = word_feats(data, word_features)
	res = classifier.classify(unsup_feat)
	if res == 'pos':
		pos += 1
	elif res == 'neg':
		neg += 1
print pos
print neg


# # with open('1.txt','r') as f:
# #     for line in f:
# #         unsup_tweets.append(line)
# # f.close()
# def word_feats(tmp, word_features):
#         # document_words = set(tmp)
#         features = {}
#         for word in word_features:
#             features['contains(%s)' % word] = (word in tmp)
#         return features

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

# positive_tweets, negative_tweets = 0, 0
# for tweet in unsup_tweets:
# 	analysis = TextBlob(tweet).sentiment
# 	if analysis.polarity > 0:
# 		positive_tweets += 1
# 	if analysis.polarity < 0:
# 		negative_tweets += 1
# print positive_tweets
# print negative_tweets

import numpy as np
import got
import datetime
from datetime import timedelta
import sys,getopt,got,datetime,codecs

addition_mode = False # Keep false
specify_date_mode = False # Keep False

def main(argv):
    PATH = "./twitter/oldTweets/"
    ticker = argv[1]
    if ticker == 'DJIA':
        searchquery = '#dowjones OR #djia OR #DOW OR #stocks OR #stock OR #stockmarket AND up OR down OR good OR bad OR rise OR fall'
    else:
        searchquery = ticker
    tweetsPerDay = int(argv[2])
    numberOfDays = int(argv[3])

    tweets = []
    date = datetime.date.today()
    if(addition_mode or specify_date_mode):
        date = date - datetime.timedelta(days=9)
    print("Get %d tweets %dDays before %s" % (tweetsPerDay, numberOfDays, unicode(date)))
    # print(date)
    if(addition_mode):
        outputFile = codecs.open(PATH + ticker + ".csv", "a", "utf-8")
    else:
        outputFile = codecs.open(PATH + ticker + ".csv", "w+", "utf-8")
    if(addition_mode == False):
        outputFile.write('username,date,retweets,favorites,text,geo,mentions,hashtags,id,permalink')
    for i in range(numberOfDays): # control get how many days data before today
        print date;
        date = date - datetime.timedelta(days=1)
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(searchquery).setSince(unicode(date - timedelta(days=1))).setUntil(unicode(date)).setMaxTweets(tweetsPerDay)
        def receiveBuffer(tweets):
            for t in tweets:
                outputFile.write(('\n%s,%s,%d,%d,"%s",%s,%s,%s,"%s",%s' % (t.username, t.date.strftime("%F"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)))
            outputFile.flush();
            print 'More %d saved on file...' % len(tweets)
        got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)
    outputFile.close()

if __name__ == '__main__':
	main(sys.argv)
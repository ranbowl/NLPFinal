# Stock prediction based on social network sentiment
Natual Language Processing Spring 2017

XL - YJ

**Basic usage:**  
python getOldTweets.py @Amazon 200 30  
    *get random tweets for Amazon, 200 tweets per day, for the past 30 days*  
python processingOldTweets.py @Amazon  
    *analyzing tweets for Amazon and count the positive negative tweets and calculate ratio*  
python trainSentiToPredict.py @Amazon  
    *train stock prediction model and show accuracy*  
python predictTwitter.py @Amazon  
    *this python script will retrieve new tweets for Amazon and predict the trend for next day*  

python getStocktwits.py  
    *it will automatically collect tweets from Stocktwits for stocks in "names" array*  
    *but you need to control the id number to get old tweets*  
    *due to the rate limits, you may need many machines with different IPs*  
    *processing and training are the same with Twitter just remember to change the PATH*  
    


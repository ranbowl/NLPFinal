import json
import datetime
import codecs

PATH = "./twitter/stocktwits/"
FILENAME = PATH + "stocktwits.json"

names = ["GOOG", "MSFT", "AMZN", "UAL"]
tickers = {"GOOG" : "@Google", "MSFT" : "@Microsoft", "AMZN" : "@Amazon", "UAL" : "@United"}

def read_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    data = read_from_file(FILENAME)
    # print data
    for symbol in names:
        ticker = tickers[symbol]
        print ticker
        outputFile = codecs.open(PATH + ticker + ".csv", "w+", "utf-8")
        outputFile.write('username,date,retweets,favorites,text,geo,mentions,hashtags,id,permalink')
        msgs = data[symbol]
        for msg in msgs:
            text = msg['body']
            date = datetime.datetime.strptime(msg['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            date2 = date.strftime("%F")
            outputFile.write(('\n%s,%s,,,"%s",,,,%s,' % (msg['user']['username'], date2, unicode(text), msg['id'])))
            # print date2
            # print text

        # print msgs



if __name__ == "__main__":
    main()

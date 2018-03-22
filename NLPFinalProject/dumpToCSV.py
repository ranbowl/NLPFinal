import csv
import pickle
import sys

def main(argv):
    PATH = './twitter/stocktwits/'
    ticker = argv[1]

    result = pickle.load(open(PATH + 'result/' + ticker + '_ratio.p', 'rb'))

    # print result
    close = dict()
    with open(PATH + 'result/' + ticker + '_close.csv', 'rb') as googleclose:
        csvreader = csv.reader(googleclose)
        for row in csvreader:
            close[row[0]] = row[4]

    with open(PATH + 'result/' + ticker + '_ratiocsv.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['date', 'usratio', 'tbratio', 'close'])
        for row in result:
            if row[0] in close:
                newrow = row
                newrow.append(close[row[0]])
                csvwriter.writerow(newrow)
            else: 
                csvwriter.writerow(row)


if __name__ == '__main__':
    	main(sys.argv)

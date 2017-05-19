import numpy as np
import ujson
import operator
from datetime import datetime


""" PREPARES RELEVANCY.PY OUTPUT FOR INDICO BY FILTERING TOP 25% MOST RELEVANT """ 

def main():

    # rel_files = ['brexit_reddit_rel', 'brexit_twitter_rel', 'india_reddit_rel', 'india_twitter_rel']
    # data_files = ['indi_input_bre_red', 'indi_input_bre_twi', 'indi_input_ind_red', 'indi_input_ind_twi']
    # output_folder = ['brexit-reddit', 'brexit-twitter', 'india-reddit', 'india-twitter', 'output']

    rel_files = ['india_reddit_rel']
    data_files = ['indi_input_india_reddit']
    output_folder = ['output']

    threshold = 0.25
    totalCount = 0
    missedCount = 0
    for i in range(0, 1):

        label = rel_files[i].split("_")
        label = label[0] + '_' + label[1] + '_' + 'filtered.txt'
        data = ujson.loads(open('output/{}.txt'.format(data_files[i])).read())
        rel = ujson.loads(open('output/{}.txt'.format(rel_files[i])).read())

        keys = list(data.keys())
        for key in keys:
            try:
                data[key][1] = shorten(data[key][1])
            except Exception:
                pass

        sorted_items = sorted(rel.items(), key=operator.itemgetter(1))[::-1]
        sorted_items = sorted_items[:int(len(sorted_items)*threshold)]
        count = 0
        counter = 0
        filter_data = {}
        for item in sorted_items:
            id = item[0]
            counter += 1
            try:
                filter_data[id] = data[id]
            except Exception:
                # print("missing key: {}".format(id))
                count += 1

        dates = {}
        for d in data.values():
            date = d[1]
            if date not in dates:
                dates[date] = 0

        total_days = len(list(dates.keys()))

        for d in filter_data.values():
            date = d[1]
            dates[date] += 1

        values = np.array(list(dates.values()))
        values[values>0] = 1
        filter_days = sum(values)

        print("Count for {} is {}".format(rel_files[i], counter*(1/threshold)))
        print("Total days {} vs Filtered Days {}, loss= {}".format(total_days,filter_days, total_days-filter_days))
        # print("dates: {}".format(list(dates.keys())))
        totalCount += counter*(1/threshold)
        missedCount += count

        ujson.dump(filter_data, open('output/{}'.format(label), 'w'))

    # print("Total count {}, missed count {}".format(totalCount, missedCount))


def shorten(dtetime):
    dte = datetime.strptime(dtetime, '%m/%d/%y %H:%M').strftime('%m-%d-%Y')
    return dte


if __name__ == "__main__":

    main()

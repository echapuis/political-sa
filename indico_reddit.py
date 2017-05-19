import indicoio
import sys
import csv
import ujson

""" Calls Inidico API for Political Analysis """

# Place inidico API Key here (apply online)
indicoio.config.api_key = ''

ids = []
texts = []
dates = []
iteration = 2
batch_size = 1


def political_analysis(path):  # , path_keys):

    rel_file = ujson.loads(open(path).read())

    count = 0
    count2 = 0
    count3 = 0
    for key in rel_file:
        count3 += 1
        # if(count >= batch_size * iteration):
        # 	break
        if (count >= batch_size * (iteration - 1)):
            value = rel_file[key]
            texts.append(value[0])
            dates.append(value[1])
            ids.append(key)

            count2 += 1
        count += 1

    output_indico = indicoio.political(texts)

    return output_indico


def group_by_dates(output_indico):
    liberal_count = {}
    conservative_count = {}

    i = 0
    for output in output_indico:
        date = dates[i]
        threshold = 0.5
        # print output

        if (output['Conservative'] + output['Libertarian'] >= threshold):
            if (date in conservative_count):
                conservative_count[date] += 1
            else:
                conservative_count[date] = 1
        else:
            if (date in liberal_count):
                liberal_count[date] += 1
            else:
                liberal_count[date] = 1
        i += 1

    all_days = set(conservative_count.keys()).union(set(liberal_count.keys()))

    final_list = []

    for day in all_days:
        conservative = 0
        liberal = 0

        if (day in conservative_count):
            conservative = conservative_count[day]
        if (day in liberal_count):
            liberal = liberal_count[day]

        final_list.append([day, conservative, liberal])

    return final_list


if __name__ == "__main__":
    path = sys.argv[1]
    political_sent = political_analysis(path)

    file_name = "india_reddit_political.csv"
    ofile = open(file_name, "wb")
    writer = csv.writer(ofile, delimiter=',')

    for i in xrange(len(ids)):
        values = []
        values.append(ids[i])
        values.append(dates[i])

        for k, v in political_sent[i].items():
            values.append(v)

        writer.writerow(values)


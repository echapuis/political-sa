import pickle
from datetime import datetime, date, timedelta
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.dates import MO, WeekdayLocator, DateFormatter
import argparse
import numpy as np

""" PLOTS FINAL GRAPHS """ 

input_dir = ''
output_dir = ''
threshold = 0.5

show = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters specify where/how files are saved. \n"
                                                 "This program takes as input a csv,tsv, or json file and outputs stuff.")
    parser.add_argument('input_folder', type=str, default=input_dir)
    parser.add_argument('output_dir', type=str, default='')
    args = parser.parse_args()
    return args

def read_as_dict(dataset, event):
    if dataset == 'twitter':
        cons = pd.read_csv(os.path.join(input_dir, '{}_{}_cons.csv'.format(event,dataset)), sep=',', header=-1).as_matrix()
        lib =  pd.read_csv(os.path.join(input_dir, '{}_{}_lib.csv'.format(event,dataset)), sep=',', header=-1).as_matrix()
        cons_dict = {}
        lib_dict = {}

        for i in range(cons.shape[0]):
            if cons[i,2] >= 0.5:
                if cons[i,1] not in cons_dict:
                    cons_dict[cons[i,1]] = 1
                else:
                    cons_dict[cons[i,1]] += 1
            else:
                if cons[i,1] not in lib_dict:
                    lib_dict[cons[i,1]] = 1
                else:
                    lib_dict[cons[i,1]] += 1

        for i in range(lib.shape[0]):
            if lib[i,2] >= 0.5:
                if lib[i,1] not in lib_dict:
                    lib_dict[lib[i,1]] = 1
                else:
                    lib_dict[lib[i,1]] += 1
            else:
                if lib[i,1] not in cons_dict:
                    cons_dict[lib[i,1]] = 1
                else:
                    cons_dict[lib[i,1]] += 1

        pickle.dump(cons_dict, open(os.path.join(output_dir,'{}_{}_cons.dict'.format(event, dataset)), 'wb'))
        pickle.dump(lib_dict, open(os.path.join(output_dir,'{}_{}_lib.dict'.format(event, dataset)), 'wb'))

    else: #Reddit
        data = pd.read_csv(os.path.join(input_dir, '{}_{}_political.csv'.format(event, dataset)), sep=',')
        cols = list(data.columns.values)
        data = data.as_matrix()
        cons_dict = {}
        lib_dict = {}

        for i in range(data.shape[0]):

            if data[i,3] > data[i,5]: # LIBERAL
                if data[i, 1] not in lib_dict:
                    lib_dict[data[i, 1]] = 1
                else:
                    lib_dict[data[i, 1]] += 1
            else:
                if data[i, 1] not in cons_dict:
                    cons_dict[data[i, 1]] = 1
                else:
                    cons_dict[data[i, 1]] += 1

        pickle.dump(cons_dict, open(os.path.join(output_dir, '{}_{}_cons.dict'.format(event, dataset)), 'wb'))
        pickle.dump(lib_dict, open(os.path.join(output_dir, '{}_{}_lib.dict'.format(event, dataset)), 'wb'))

def plot_graph(event, dataset, normalize = False, saveAs=''):
    normalize = normalize
    MA = 7
    # if event == 'brexit' and dataset == 'twitter': MA = 1
    loc = WeekdayLocator(byweekday=MO)
    fmt = DateFormatter('%b %d')


    cons = pickle.load(open(os.path.join(output_dir,'{}_{}_cons.dict'.format(event,dataset)), 'rb'))
    lib = pickle.load(open(os.path.join(output_dir,'{}_{}_lib.dict'.format(event,dataset)), 'rb'))

    x1 = sorted([(datetime.strptime(str(a[0]), '%m-%d-%Y'),a[1]) for a in list(cons.items())])
    date_set = sorted(list(set(x1[0][0] + timedelta(x) for x in range((x1[-1][0] - x1[0][0]).days))))
    y1 = []
    i = 0
    for date in date_set:
        if date == x1[i][0]:
            y1.append(x1[i][1])
            i += 1
        else:
            y1.append(0)
    y1 = np.convolve(y1, np.ones((MA,)) / MA, mode='valid')
    x1 = date_set[:len(y1)]

    x2 = sorted([(datetime.strptime(str(a[0]), '%m-%d-%Y'), a[1]) for a in list(lib.items())])
    date_set = sorted(list(set(x2[0][0] + timedelta(x) for x in range((x2[-1][0] - x2[0][0]).days))))
    y2 = []
    i = 0
    for date in date_set:
        if date == x2[i][0]:
            y2.append(x2[i][1])
            i += 1
        else:
            y2.append(0)
    y2 = np.convolve(y2, np.ones((MA,)) / MA, mode='valid')
    x2 = date_set[:len(y2)]

    if normalize:
        size = min(len(y1), len(y2))
        y1 = y1[:size]
        y2 = y2[:size]
        x1 = x1[:size]
        x2 = x2[:size]
        sum = y1 + y2
        y1 = np.divide(y1, sum)
        y2 = np.divide(y2, sum)


    # print(x1)
    # print(y1)

    if event == 'brexit':
        event = 'Brexit'
        ax_range = [datetime.strptime('01-01-2016', '%m-%d-%Y'), datetime.strptime('06-15-2016', '%m-%d-%Y')]
        legend = ['Leave', 'Stay']
    else:
        event = 'the India Election'
        ax_range = [datetime.strptime('01-01-2014', '%m-%d-%Y'), datetime.strptime('04-01-2014', '%m-%d-%Y')]
        legend = ['Conservative', 'Liberal']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.axis((x1[0], x1[-1], 0.7,1.3))
    ax.plot(x1, y1, x2, y2)
    if normalize:
        ax.axhline(y=.5,linestyle='--', color='black')
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlim(ax_range)
    ax.legend(legend)
    ax.set_ylabel('Number of Comments/Tweets')
    ax.set_xlabel('Date')
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
    if dataset == 'twitter': dataset = 'Twitter'
    else: dataset = 'Reddit'

    ax.set_title('{} Sentiment over Time for {}'.format(dataset, event))
    plt.tight_layout(pad=2)

    if saveAs != 'meh':
        plt.savefig(saveAs, bbox_inches='tight')
    if show:
        plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    input_dir = args.input_folder
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    events = ['brexit', 'india']
    datasets = ['twitter', 'reddit']

    for event in events:
        for dataset in datasets:
            try:
                read_as_dict(dataset, event)
                plot_graph(event, dataset, normalize=False,
                           saveAs=os.path.join(output_dir,'{}_{}.png'.format(event, dataset)))
            except:
                # print("{} {} failed.".format(event, dataset))
                pass
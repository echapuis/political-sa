# Political Sentiment Analysis

This project performs political sentiment analysis using Twitter and Reddit data to predict the outcomes of the 2016 British Referendum (Brexit) and the Indian Parliamentary Election of 2014. Details of our implementation and method can be found in our paper (paper.pdf), and some visualizations and results are also included in our poster presentation (poster.pdf).

This work was the product of a collaboration between myself, Arjun Chintapalli, Joy Kimmel, Gowri Nayar, M.S. Suraj, and Jia Yi Yan. This was the final project of our CSE 6242 Data & Visual Analytics graduate course.

My role in this project primarily involved development of the code for the data pre-processing, text vectorization, vector clustering, and data filtering portions. 

## INSTALLATION: 

Please install all necessary packages given in the import lines; download GetOldTweets from github and you'll also need to get an API key from https://indico.io/non-commercial to complete step 8. 

## EXECUTION: 

Sample reddit data on the Indian Election has been included for demonstration purposes. If you decide to use this data to test the application, skip step 1.

### 1. (Optional) Gather Data 

Twitter: getOldTweets downloaded from github: https://github.com/Jefferson-Henrique/GetOldTweets-python

Reddit: search_reddit.py, uses praw API, change search parameters

### 2. Word2Vec / Clustering

Vectorizes the data and performs K-Means clustering for additional tasks. Example execution:

~~~
python3 text_clean.py -sv data/india-reddit.json "output"
~~~

Additionally, other options can be specified. Use help command for more details.

### 3. Text to Cluster

Converts the text data to vectors of cluster counts based on the number of words that appear in each cluster for the text. Example execution (assuming previous example execution):

~~~
python3 filter_data.py "output/TC_saveFolder/wordlists.list" "output/clusters.csv"
~~~

### 4. Parse Counts

Calculates relevancy of data based on cluster allocations and generates a histogram. Example execution (assuming previous example execution):

~~~
python3 code/parse_counts.py 'output/clusters.csv' 'output/clusters_freq_count.counts' 'India Reddit' 'india_reddit'
~~~

### 5. Filter by Relevancy

Filters the data based on relevancy. Example execution (assuming previous example execution):

~~~
python3 code/relevancy.py 'data/india-reddit.json' 'output/' 'output/india_reddit_rel.txt' 'india_reddit'
~~~

### 6. Sort for Indico

Example execution (assuming previous example execution):

~~~
python3 code/sort_for_indico.py
~~~

### 7. Send to Indico  # use API applied for

Example execution (assuming previous example execution):

~~~
python3 indico_twitter.py brexit_twitter_filtered.txt [@britinfluence,#theknoweu,#theinvisableman,#MyImageOfTheEU,#leafchaos,#ImagineEurope,@sayyes2europe,#euin,#brexitfears,#betteroffin,#yestoeu,#remaineu,#yes2eu,#britin,@StrongerInPress,#leadnotleave,#ukineu,#saferbritain,#Bremain,#votein,#INtogether,#LabourIn,#greenerin,@StrongerIn,#remain,#voteremain,#StrongerIn] [@JuneExit,#fudgeoff,#eurenegotiation,#eunegotiation,#NoEu,#euout,#britainout,#notoeu,@Grassroots_out,@euromove,#grassrootsout,@Vote_LeaveMedia,@LeaveEUOfficial,#no2eu,#betteroffout,#loveeuropeleaveeu,#wrongthenwrongnow,#beleave,#voteout,#voteleave,#TakeControl,@vote_leave,#leaveeu] [@Choice4Britain,@EUinEUout,#eunegotiation#EdEUref,@BrexitWatch,@whatukthinks,#eureform,#EUpoll,@eureferendum,#UKandEU,@lsebrexitvote,#eukay,@UKandEU,#UKreferendum,#UKRef,#EUpol,#europeanunion,#projectfact,#referendum,#ref,#Davidcameron,#projectfear,#eureferendum,#Brexit,#euref]

python3 indico_twitter.py india_twitter_filtered.txt [@AamAadmiParty,@ArvindKejriwal,@RahulGandhi,#AAP,#Kejriwal] [@BJP4India’s,@NarendraModi,#NaMo,#BJP,#Modi,#NaMoinGoa] [@INCIndia,#YourVote2014,Lok Sabha Elections 2014,#Elections2014]

python indico_reddit.py brexit_reddit_filtered.txt

python indico_reddit.py india_reddit_filtered.txt
~~~

### 8. Plot Results

Example execution (assuming previous example execution):

~~~
python3 code/make_charts.py 'charts-input' 'charts-output'
~~~

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: amin
"""

# =============================================================================
# Importing flat files from the web: your turn!
# You are about to import your first file from the web! 
# The flat file you will import will be 'winequality-red.csv' from the 
# University of California, Irvine's Machine Learning repository. 
# The flat file contains tabular data of physiochemical properties of red wine, 
# such as pH, alcohol content and citric acid content, along with wine quality rating.
# =============================================================================

# Import package
from urllib.request import urlretrieve

# Import pandas
# Assign url of file: url
url = "https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv"

# Save file locally
urlretrieve(url, 'winequality-red.csv')

import pandas as pd

# Read file into a DataFrame and print its head
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.head())

# =============================================================================
#  If you just wanted to load a file from the web into a DataFrame without 
#  first saving it locally, you can do that easily using pandas. 
#  In particular, you can use the function pd.read_csv() with the URL as the 
#  first argument and the separator sep as the second argument.
# =============================================================================

# Import packages
import matplotlib.pyplot as plt
import pandas as pd

# Assign url of file: url
url = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'

# Read file into a DataFrame: df
df = pd.read_csv(url, sep=';')

# Print the head of the DataFrame
print(df.head())

# Plot first column of df
pd.DataFrame.hist(df.ix[:, 0:1])
plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
plt.ylabel('count')
plt.show()

# =============================================================================
# Importing non-flat files from the web - EXCEL
# =============================================================================

# Import package
import pandas as pd

# Assign url of file: url
url = 'http://s3.amazonaws.com/assets.datacamp.com/course/importing_data_into_r/latitude.xls'

# Read in all sheets of Excel file: xl.  
# In order to import all sheets you need to pass None to the argument sheetname

xl = pd.read_excel(url, sheetname=None)

# Print the sheetnames to the shell. These will be the keys of the dictionary 
print(xl.keys())

# Print the head of the first sheet (using its name, NOT its index)
print(xl['1700'].head())

# =============================================================================
# Performing HTTP requests in Python using urllib
# =============================================================================

# Import packages
from urllib.request import urlopen, Request

# Specify the url
url = "http://www.datacamp.com/teach/documentation"

# This packages the request: request
request = Request(url)

# Sends the request and catches the response: response
response = urlopen(request)

# Print the datatype of response
print(type(response))

# Extract the response: html
html = response.read()

# Print the html
print(html)

# Be polite and close the response!
response.close()

# =============================================================================
# Performing HTTP requests in Python using requests
# =============================================================================

# Import package
import requests

# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"

# Packages the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response: 
text = r.text

# Print the html
print(text)

# =============================================================================
# Parsing HTML with BeautifulSoup
# In this interactive exercise, you'll learn how to use the 
# BeautifulSoup package to parse, prettify and extract information from HTML. 
# You'll scrape the data from the webpage of Guido van Rossum, Python's very own 
# Benevolent Dictator for Life.
# =============================================================================

# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()

# Print the response
print(pretty_soup)

# =============================================================================
# Turning a webpage into data using BeautifulSoup: getting the text
# =============================================================================

# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response as html: html_doc
html_doc = r.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Get the title of Guido's webpage: guido_title
guido_title = soup.title

# Print the title of Guido's webpage to the shell
print(guido_title)

# Get Guido's text: guido_text
guido_text = soup.text

# Print Guido's text to the shell
print(guido_text)

# =============================================================================
# Turning a webpage into data using BeautifulSoup: getting the hyperlinks
# =============================================================================

# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url
url = 'https://www.python.org/~guido/'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Extracts the response as html: html_doc
html_doc = r.text

# create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Print the title of Guido's webpage
print(soup.title)

# Find all 'a' tags (which define hyperlinks): a_tags
#Use the method find_all() to find all hyperlinks in soup, remembering that 
#hyperlinks are defined by the HTML tag <a>; store the result in the variable a_tags.

a_tags = soup.find_all('a')

# Print the URLs to the shell
#The variable a_tags is a results set: your job now is to enumerate over it, 
#using a for loop and to print the actual URLs of the hyperlinks; to do this, 
#for every element link in a_tags, you want to print() link.get('href')

for link in a_tags:
    print(link.get('href'))

# =============================================================================
# Loading and exploring a JSON
# =============================================================================

import json

# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)

# Print each key-value pair in json_data
#Use a for loop to print all key-value pairs in the dictionary json_data. 
#Recall that you can access a value in a dictionary using the syntax: dictionary[key].
    
for k in json_data.keys():
    print(k + ': ', json_data[k])

# =============================================================================
import json
 
with open('snakes.json', 'r') as json_file:
    json_data = json.load(json_file)

type(json_data)

#Exploring JSONs in Python

for key, value in json_data.items():
    print(key + ':', value) 

# =============================================================================
# API requests
# =============================================================================
# Import requests package
import requests

#Assign to the variable url the URL of interest in order to query 'http://www.omdbapi.com' 
#for the data corresponding to the movie The Social Network. 
#The query string should have two arguments: apikey=ff21610b and t=the+social+network. 
#You can combine them as follows: apikey=ff21610b&t=the+social+network

# Assign URL to variable: url

url = 'http://www.omdbapi.com?apikey=ff21610b&t=the+social+network'


# Package the request, send the request and catch the response: r
r = requests.get(url)

# Print the text of the response
print(r.text)

# =============================================================================
# JSON–from the web to Python
# =============================================================================

# Import package
import requests

# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=ff21610b&t=social+network'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary  by applying the json() method to the response object r: json_data

json_data = r.json()

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

# =============================================================================
# Checking out the Wikipedia API
# =============================================================================
#You're doing so well and having so much fun that we're going to throw one more API at you:
#the Wikipedia API (documented here). You'll figure out how to find and extract information from 
#the Wikipedia page for Pizza. What gets a bit wild here is that your query will 
#return nested JSONs, that is, JSONs with JSONs, but Python can handle that because 
#it will translate them into dictionaries within dictionaries.
    
# Import package
import requests

# Assign URL to variable: url
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'

# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()

# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)

#The variable pizza_extract holds the HTML of an extract from Wikipedia's 
#Pizza page as a string; use the function print() to print this string to the shell.

# =============================================================================
# The Twi!er API and Authentication
# =============================================================================

#Twitter has a number of APIs
#1. REST APIs - allows the user to read and write twitter data
#2. The streaming APIs - offers several streaming endpoints. We will use the 
#public stream. From the public stream, we will use the GET statuses/sample API
# which returns a small random sample of all streams.
#
#Twitter firehose is a more powerful and provides access to all streaming tweets. 

# =============================================================================
# API Authentication
# The package tweepy is great at handling all the Twitter 
# API OAuth Authentication details for you. All you need to do is pass it your 
# authentication credentials
# =============================================================================

# Import package
import tweepy, json

# Store OAuth authentication credentials in relevant variables
access_token = "395891297-EQGBYZ8TJn08AhNk3RBIC75AcATNFqgjs31Zl319"
access_token_secret = "ahVJ0gd5QqexSSPyT6iyCeMBpeCQ6sw95dMMDgXbPLwrf"
consumer_key = "oGa5dI3nF98IQ40ElkUaiGoHT"
consumer_secret = "7fPOGKbgl1QjJmvapJMTKZjs4oImGLCMVthuBSG0yuW8IQ18KO"

#Complete the passing of OAuth credentials to the OAuth handler auth by 
#applying to it the method set_access_token(), along with arguments access_token 
#and access_token_secret

# Pass OAuth details to tweepy's OAuth handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("tweets.txt", "w")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
        print(status)

# Initialize Stream listener
l = MyStreamListener()

# Create your Stream object with authentication
stream = tweepy.Stream(auth, l)


# Filter Twitter Streams to capture data by the keywords:
stream.filter(track=['clinton','trump','sanders','cruz'])

# =============================================================================
# Load and explore your Twitter data
# Now that you've got your Twitter data sitting locally in a text file, 
# it's time to explore it! This is what you'll do in the next few interactive exercises. 
# In this exercise, you'll read the Twitter data into a list: tweets_data
# =============================================================================

# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'

# Initialize empty list to store tweets: tweets_data
tweets_data = []

# Open connection to file
tweets_file = open(tweets_data_path, "r")

# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)

# Close connection to file
tweets_file.close()

# Print the keys of the first tweet dict
print(tweets_data[0].keys())

# =============================================================================
# Twitter data to DataFrame
# Now you have the Twitter data in a list of dictionaries, tweets_data, where 
# each dictionary corresponds to a single tweet. Next, you're going to extract 
# the text and language of each tweet. The text in a tweet, t1, is stored as the 
# value t1['text']; similarly, the language is stored in t1['lang']. 
# Your task is to build a DataFrame in which each row is a tweet and the 
# columns are 'text' and 'lang'.
# =============================================================================

#Use pd.DataFrame() to construct a DataFrame of tweet texts and languages; 
#to do so, the first argument should be tweets_data, a list of dictionaries. 
#The second argument to pd.DataFrame() is a list of the keys you wish to have as columns

# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])

# Print head of DataFrame
print(df.head())

# =============================================================================
# A little bit of Twitter text analysis
# Now that you have your DataFrame of tweets set up, you're going to do a bit 
# of text analysis to count how many tweets contain the words 'clinton', 
# 'trump', 'sanders' and 'cruz'. In the pre-exercise code, we have defined the 
# following function word_in_text(), which will tell you whether the first argument 
# (a word) occurs within the 2nd argument (a tweet)
# =============================================================================

import re

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]

# Iterate through df, counting the number of tweets in which
# each candidate is mentioned

for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])

# =============================================================================
# Plotting your Twitter data
# Now that you have the number of tweets that each candidate was mentioned in, 
# you can plot a bar chart of this data. 
# You'll use the statistical data visualization library seaborn
# =============================================================================

# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(color_codes=True)

# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']

# Plot histogram
ax = sns.barplot(cd, [clinton, trump, sanders, cruz])
ax.set(ylabel="count")
plt.show()

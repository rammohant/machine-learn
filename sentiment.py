# Tara Ram Mohan 
# CMSC 416 Programming Assignment 4: Sentiment Analysis
# April 25, 2022
###########################################################
# Problem: This Python program sentiment.py adapts mu sentiment.py program to perform (positive/negative) sentiment analysis over tweets. It takes in 2 file names as input (a training file containing  sentiment tagged text and a test file containing text to be  sentiment tagged, and the name of the log file). 
# It uses the bag of words feature representation and decision list with 3 additional features to learn a model from sentiment-train.txt and apply that to each of the sentences found in sentiment-test.txt in order to assign a sentiment to the word line. 
# The accuracy of our sentiment tagger will be determined using the scorer.py file and then compared to that of the most frequent sentiment baseline.
# Accuracy of most frequent baseline: 0.6896551724137931
# Accuracy:  0.728448275862069
# Confusion Matrix:
#                    Predicted: positive  Predicted: negative
# Actual: positive                  140                   43
# Actual: negative                   20                   29
###########################################################
# Usage: To start the program, run the following command: 
# python3 (sentiment.py) (sentiment-train.txt) (sentiment-test.txt) (my-model.txt) > (my-sentiment-answers.txt)
# The program will train using (sentiment-train.txt), which will be used to tag sentiments to the (sentiment-test.txt).The output filename will be defined by (my-sentiment-answers.txt) in the same format as found in line-key.txt.  
# The file my-model.txt is intended to be used as a log file in debugging your program. 
# Example command: python3 sentiment.py sentiment-train.txt sentiment-test.txt my-model.txt > my-sentiment-answers.txt
# All of the following are portions (not the entire file example) of the actual file.
# Example input sentiment-train.txt: 
  # <corpus lang="en">
  # <lexelt item="sentiment">
  # <instance id="620821002390339585">
  # <answer instance="620821002390339585" sentiment="negative"/>
  # <context>
  # Does @macleansmag still believe that Ms. Angela Merkel is the "real leader of the free world"?  http://t.co/isQfoIcod0 (Greeks may disagree
  # </context>
  # </instance>
# Example input sentiment-test.txt:
# <instance id="620979391984566272">
  # <answer instance="620979391984566272">
  # <context>
  # On another note, it seems Greek PM Tsipras married Angela Merkel to Francois Hollande on Sunday #happilyeverafter http://t.co/gTKDxivf79
  # </context>
  # </instance>
  # <instance id="621340584804888578">
  # <answer instance="621340584804888578">
  # <context>
# Example final my-model.txt: 
  # Feature: new
  # Log Likelihood: 27.89338538039604
  # Predicted Sense: positive
# Example output my-sentiment-answers.txt: 
  # <answer instance=620979391984566272 sentiment="positive"/>
  # <answer instance=621340584804888578 sentiment="positive"/>
  # <answer instance=621351052047028224 sentiment="positive"/>
# ###########################################################
# Algorithm: 
# (1) The program will read all the files outlined in the user input. This includes the training data in pos-train.txt and the text to be tagged in post-test.txt. 
# (2) For each text (training and test), convert the contents of the corpus to a string and removes unnecessary tags.
# TRAINING ---------------------------------------------------------------
# (3) Using the training string of lines, we extract the sentiment and context information from each sentence and create 2 arrays. 
# (4) Then, we create the bag of words features for our decision list. Here, a feature includes each word in all sentence contexts that is not the ambiguous word or a stop word. 
# First, we iterate through all sentiments and contexts. If a sentiment is not in the sentiment unigram dictionary, we add it. Then, for each word in context, we add it to the corresponding sentiment in the sentiment dictionary sentiment_dict[sentiment][word] if it doesn't already exist and then increment by one. This produces a sentiment-word frequency dictionary, which can then be used to create a decision list. 
# To create the decision list, we first create a list of all the features in the sentiments dictionary. We iterate through all the unique features. For each bag-of-words feature, we calculate the probability that the feature is associated with each possible sentiment (i.e. product and phone). This probability is simply calculated as the frequency in sentiment_dict[sentiment][word] since the denominator (frequency of word) acts to normalize 2 probabilies, which is not necessary since the denominator will be the same for both probabilities. The higher probability is used to assign the appropriate sentiment to the feature. 
# Then, these probabilities are used to calculate log-likelihood as absolute value of the log of the frequency of sentiment 1 and feature i together divided by the frequency of sentiment 2 and feature i together. 
# We then produce a decision list of each feature (word), prediction, and corresponding log-likelihood, which is then ordered in ascending order by log-likelihood
# TESTING ---------------------------------------------------------------
# (5) To sentiment tag the testing data, we first extract the instance id and context from each sentence. We loop through all the instances and contexts in order to create a list of all possible words in the testing data. We only add words to our test set if they are not a stop word. The default sentiment prediction is defined as the most frequent sentiment. 
# Then, for each item in the decision list, we get the feature, sentiment, and log likelihood, all three of which is outputted to the logging file for debugging. 
# We have 5 features in addition to the decision list features:
# (1) Negation words (i.e. not, no, never, not, nope, nothing) -> negative sentiment
# (2) Positive words (i.e. good, great, wonderful, best) -> positive sentiment
# (3) Negative emoticons (:( ) -> positive sentiment
# (4) Positive emoticons (:D or :) or :P) -> positive sentiment
# (5) Exclamation points -> positive sentiment
# (6) Capitalization of word (with length greater than 3) -> positive sentiment
# (7) Decision list: In the bag of words feature representation, each word (excluding stop words and the ambiguous word) in training is identified as features in the decision list. We implement a decision list by checking to see if the current feature is in the list of test words. If so, we assign the sentiment of the current feature to the current instance id and print that answer to the test answer file. 
# In order to compare my results to that of the most frequent sentiment baseline, I created an additional method that would take in the raw test and train data to calcualte the most frequent sentiment baseline and apply that to all of the test instances. The output of this will run by scorer.py to determine the baseline accuracy. 
# ###########################################################
import sys
import re
import math
from collections import OrderedDict

stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are","as","at","be","because","been","before","being","below","between","both","but","by","cannot","could","did","do","does","doing","down","during","each","few","for","from","further","had","has","have","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","it","it's","its","itself","let's","me","more","most","my","myself","of","off","on","once","only","or","other","ought","our","ours"," ourselves","out","over","own","same","she","she'd","she'll","she's","should","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","we","we'd","we'll","we're","we've","were","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","would","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]

# Function that takes in a filename and iterates through each line in the file, converting it into a singular string. 
# It removes unnecessary tags
def get_clean_string(file):
  
  # Delete all newlines and trailing spaces and create one big string
  with open(file, 'r+') as f:
    lines = []
    for line in f:
      lines.append(line.strip()) # remove extra white space at the beginning and end of each line

  # Combine all lines
  lines = ' '.join(line for line in lines)

  # Remove unnecessary tags from lines
  lines = re.sub(r'<s>|<\/s>|<@>|<p>|<\/p>','',lines)
  
  return lines

# Function that extracts information, including the instance_id, sentiment, context (sentence contents) from each sentence 
def get_info(lines, type): 
  
  # Create an instances array
  instances = re.findall(r'<instance id="(.*?)">', lines)
  
  # Create a context (sentence contents) array to generate bag of words
  context_raw = re.findall(r'<context>(.*?)<\/context>',lines)
  contexts = [re.sub(r'[.?,;"\']', '', context).split() for context in context_raw]
  
  # If we are reading the contents of training data, create and return array of sentiments, instances, and contexts
  if (type == 'train'):
    sentiments = re.findall(r'sentiment="(.*?)"\/>', lines)
    return sentiments, instances, contexts

  # If testing data, return just instances and contexts
  return instances, contexts

# Function that calculates log-likelihood as the absolute value of the log of the probability of sentiment 1 given feature i divided by the probability of sentiment 2 given feature i
# This probability simplies to the absolute value of the log of the frequency of sentiment 1 and feature i together divided by the frequency of sentiment 2 and feature i together
# This produces a decision list of each feature (word), prediction, and corresponding log-likelihood in ascending order by log-likelihood
def calculate_log_likely(sentiment_labels, sentiment_dict): 
  
  # Create decision dictionary
  decision_list = {}

  # Create an array of all the features in the sentiments dictionary
  all_features = []
  for item in sentiment_dict.values(): 
    all_features = all_features + list(set(word for word in item.keys()))

  # Loop through all unique features
  for feature in set(all_features):
    
    max_probability = 0
    prediction = ''
    probabilities = []
    
    # Loop through all unique sentiments to alculate the probability that the feature is associated with each possible sentiment (i.e. product and phone)
    for sentiment in sentiment_labels:
      
      # If the feature isn't in the sentiment dictionary (aka not in training), set probability to a really small number to prevent devision by 0 
      if feature not in sentiment_dict[sentiment]:
        probability = 0.00000000001 
      # Else, add probability of feature with sentiment to list
      else: 
        probability = sentiment_dict[sentiment][feature]

      # Add to list of all probabilities for current feature 
      probabilities.append(probability)

      # If the probability is higher than the current max_probability, update the max_probability value and the predicted sentiment for the current feature
      
      if probability > max_probability:
        max_probability = probability
        prediction = sentiment

    # Get numerator and denominator for the log-likelihood calculation
    prob_s1 = probabilities[0]
    prob_s2 = probabilities[1]
    
    # Calculate log-likelihood 
    log_likelihood = abs(math.log( prob_s1 / prob_s2 ))

    # Creates a decision list of each feature (word), prediction, and corresponding log-likelihood
    decision_list[feature + '|' + prediction] = log_likelihood

    # Orders the decision list in ascending order by log-likelihood
    o_decision_list = OrderedDict(sorted(decision_list.items(), key=lambda item: item[1], reverse=True))
  
  return o_decision_list

# Function that creates a decision list of bag of words features using te training data
# This produces a decision list of each feature (word), prediction, and corresponding log-likelihood in ascending order by log-likelihood
def create_decision_list(lines):

  # Extract sentiments and context from training data
  sentiments, instances, contexts = get_info(lines, 'train')

  # Get unique sentiments (i.e. phone, product)
  sentiment_labels = set(sentiments)
  
  # Determine most frequent sentiment
  global most_frequent_sentiment 
  most_frequent_sentiment = max(sentiment_labels, key = sentiments.count)
  
  # Create sentiment-feature (bag of words unigram) dictionary 
  sentiment_dict = {}

  # Loop through all sentiments and contexts
  for sentiment, context in zip(sentiments, contexts):
    
    # If the sentiment is not in the dictionary, add it 
    if sentiment not in sentiment_dict:
      sentiment_dict[sentiment] = {}
      
    # Loop to each word in the context of the current sentence
    for word in context: 

      # If the word is not a stop word
      if word not in stop_words:
        # If the feature hasn't been assigned a sentiment, add it
        if word not in sentiment_dict[sentiment]:
          sentiment_dict[sentiment][word] = 0
          
        # Increment count of the feature with that sentiment by 1
        sentiment_dict[sentiment][word] += 1

  # Calculate the log likelihood for each possible [sentiment][tag] pairing and generate a  decision list with ranked features
  decision_list = calculate_log_likely(sentiment_labels, sentiment_dict)

  # Return an ordered decision list
  return decision_list

# Function that uses the ordered decision_list to add sentiments to the test data and outputs log data to the model file  
def predict_testing(decision_list, lines, model):
  
  # Opens the file at the model filepath to be written to 
  with open(model, 'w') as f:

    # Get all instances and contexts from the test data
    instances, contexts = get_info(lines,'test')
    
    # Loop through all contexts, and instances
    for instance, context in zip(instances, contexts):
      # All possible words in the testing data
      test_words = []

      # For each word in each context sentence
      for word in context:
        
        # If the word is not yet in the list of test words AND not a stop word, add it
        if (word not in test_words) and (word not in stop_words):
          test_words.append(word)
      
      # Default prediction
      predicted = most_frequent_sentiment

      # Loop through each feature decision in the decision list starting from highest log likelihood
      for pair in decision_list:
        
        # Split pair into feature and sense 
        feature, sense = pair.split('|')

        # Get log_likelihood for model 
        log_likelihood = decision_list[pair]

        # Output to log file
        f.write(f'Feature: {feature}\n')
        f.write(f'Log Likelihood: {log_likelihood}\n')
        f.write(f'Predicted Sense: {sense}\n\n')
        
        # DECISION LIST: If the feature exists within the possible features/words in training, assign the associated sense
        
        # Feature 1: Negation words (i.e. not, no, never, not, nope, nothing) -> negative sentiment
        r1 = re.compile(r'\bnot\b|\bno\b|\bnever\b|n\'t|\bnope\b|\bnothing\b')
        # Feature 2: Positive words (i.e. good, great, wonderful, best) -> positive sentiment
        r2 = re.compile(r'\bgood\b|\bgreat\b|\bwonderful\b|\bbest\b')
        # Feature 3: Negative emoticons (:( ) -> positive sentiment
        r3 = re.compile(r':\(')
        # Feature 4: Positive emoticons (:D or :) or :P) -> positive sentiment
        r4 = re.compile(r':D|:\)|:p')
        # Feature 5: Exclamation points (indicate excitement) -> positive sentiment
        r5 = re.compile(r'!+')
        # Feature 6:  Capitalization of word (with length greater than 3) -> positive sentiment
        r6 = re.compile(r'[A-Z]{3,}')

        if list(filter(r6.match, test_words)):
          predicted = 'positive'
          break
        if list(filter(r3.match, test_words)):
          predicted = 'negative'
          break
        elif list(filter(r4.match, test_words)):
          predicted = 'positive'
          break
        elif list(filter(r5.match, test_words)): # 0.6637931034482759
          predicted = 'positive'
          break
        elif list(filter(r1.match, test_words)):
          predicted = 'negative'
          break
        elif list(filter(r2.match, test_words)):
          predicted = 'positive'
          break
        elif feature in test_words:
          predicted = sense
          break
        # Otherwise, the most frequent sentiment will be assigned to the feature
            
      # Print answers to answer file
      print(f'<answer instance={instance} sentiment="{predicted}"/>')

# Function that calculates the most frequent sentiment baseline and tagging it to all test set 
def calculate_mfs_baseline(train_lines, test_lines):

  # Get sentiments from training data
  train_sentiments, instances, contexts = get_info(train_lines, 'train')

  # Get all instances from testing data
  test_instances, contexts = get_info(test_lines,'test')

  # Get sentiment labels to calculate most_frequent sentiment from training
  sentiment_labels = set(train_sentiments)
  most_frequent_sentiment = max(sentiment_labels, key = train_sentiments.count)

  # Assign most frequent sentiment to every sentence in test file
  for instance in test_instances:
    print(f'<answer instance={instance} sentiment="{most_frequent_sentiment}"/>')

if __name__ == '__main__':
  
  # Step 1: Read in command line args
  training_file = sys.argv[1] # training file containing part of speech tagged text
  test_file = sys.argv[2] # file containing text to be part of speech tagged
  model = sys.argv[3]
  
  # Step 2: Create 2 corpus containing contents of training and testing files
  training_lines = get_clean_string(training_file)
  test_lines = get_clean_string(test_file)

  # Step 3: Generate decision list using training data
  decision_list = create_decision_list(training_lines)

  # Step 4: Assign sentiment to testing
  predict_testing(decision_list, test_lines, model)

  # EXTRA: Compare your results to that of the most frequent sentiment baseline. Comment out Step 3 and 4 and then run this and then scorer.py to get most frequent sentiment baseline accuracy.
  # calculate_mfs_baseline(training_lines, test_lines)
  # Accuracy of most frequent baseline: 0.6896551724137931

  

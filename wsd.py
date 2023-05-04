# Tara Ram Mohan 
# CMSC 416 Programming Assignment 3: Word Sense Disambiguation
# March 29, 2022
###########################################################
# Problem: This Python program called wsd.py implements a Decision List classifier to perform word sense disambiguation. It takes in 2 file names as input (a training file containing word sense tagged text and a test file containing text to be word sense tagged, and the name of the log file). 
# It uses the bag of words feature representation and decision list to learn a model from line-train.txt and apply that to each of the sentences found in line-test.txt in order to assign a sense to the word line. 
# The accuracy of our word sense tagger will be determined using the scorer.py file and then compared to that of the most frequent sense baseline.
# Accuracy of most frequent baseline: 0.42857142857142855
# Accuracy:  0.8492063492063492
# Confusion Matrix:
#                   Predicted: phone  Predicted: product
# Actual: phone                  66                  13
# Actual: product                 6                  41
###########################################################
# Usage: To start the program, run the following command: 
# python3 (wsd.py) (line-train.txt) (line-test.txt) (my-model.txt) > (my-line-answers.txt)
# The program will train using (line-train.txt), which will be used to tag word senses to the (line-test.txt).The output filename will be defined by (my-line-answers.txt) in the same format as found in line-key.txt.  
# The file my-model.txt is intended to be used as a log file in debugging your program. Y
# Example command: python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt
# All of the following are portions (not the entire file example) of the actual file.
# Example input line-train.txt: 
  # <instance id="line-n.w9_10:6830:">
  # <answer instance="line-n.w9_10:6830:" senseid="phone"/>
  # <context>
  #  <s> The New York plan froze basic rates, offered no protection to Nynex against an economic downturn that sharply cut demand and didn't offer flexible pricing. </s> <@> <s> In contrast, the California economy is booming, with 4.5% access <head>line</head> growth in the past year. </s> 
  # </context>
  # </instance>
# Example input line-test.txt:
  # <instance id="line-n.w8_059:8174:">
  # <context>
  # <s> Advanced Micro Devices Inc., Sunnyvale, Calif., and Siemens AG of West Germany said they agreed to jointly develop, manufacture and market microchips for data communications and telecommunications with an emphasis on the integrated services digital network. </s> <@> </p> <@> <p> <@> <s> The integrated services digital network, or ISDN, is an international standard used to transmit voice, data, graphics and video images over telephone <head>lines</head> . </s> 
  # </context>
  # </instance>
# Example final my-model.txt: 
  # Feature: corp
  # Log Likelihood: 27.525660600270722
  # Predicted Sense: product
# Example output my-line-answers.txt: 
  # <answer instance="line-n.w8_059:8174:" senseid="product"/>
  # <answer instance="line-n.w7_098:12684:" senseid="product"/>
  # <answer instance="line-n.w8_106:13309:" senseid="product"/>
  # <answer instance="line-n.w9_40:10187:" senseid="phone"/> 
# ###########################################################
# Algorithm: 
# (1) The program will read all the files outlined in the user input. This includes the training data in pos-train.txt and the text to be tagged in post-test.txt. 
# (2) For each text (training and test), convert the contents of the corpus to a string and removes unnecessary tags.
# TRAINING ---------------------------------------------------------------
# (3) Using the training string of lines, we extract the sense and context information from each sentence and create 2 arrays. 
# (4) Then, we create the bag of words features for our decision list. Here, a feature includes each word in all sentence contexts that is not the ambiguous word or a stop word. 
# First, we iterate through all senses and contexts. If a sense is not in the sense unigram dictionary, we add it. Then, for each word in context, we add it to the corresponding sense in the sense dictionary sense_dict[sense][word] if it doesn't already exist and then increment by one. This produces a sense-word frequency dictionary, which can then be used to create a decision list. 
# To create the decision list, we first create a list of all the features in the senses dictionary. We iterate through all the unique features. For each feature, we calculate the probability that the feature is associated with each possible sense (i.e. product and phone). This probability is simply calculated as the frequency in sense_dict[sense][word] since the denominator (frequency of word) acts to normalize 2 probabilies, which is not necessary since the denominator will be the same for both probabilities. The higher probability is used to assign the appropriate sense to the feature. 
# Then, these probabilities are used to calculate log-likelihood as absolute value of the log of the frequency of sense 1 and feature i together divided by the frequency of sense 2 and feature i together. 
# We then produce a decision list of each feature (word), prediction, and corresponding log-likelihood, which is then ordered in ascending order by log-likelihood
# TESTING ---------------------------------------------------------------
# (5) To sense tag the testing data, we first extract the instance id and context from each sentence. We loop through all the instances and contexts in order to create a list of all possible words in the testing data. We only add words to our test set if they are not a stop word. The default sense prediction is defined as the most frequent sense. 
# Then, for each item in the decision list, we get the feature, sense, and log likelihood, all three of which is outputted to the logging file for debugging. 
# Decision list: In the bag of words feature representation, each word (excluding stop words and the ambiguous word) in training is identified as features in the decision list. We implement a decision list by checking to see if the current feature is in the list of test words. If so, we assign the sense of the current feature to the current instance id and print that answer to the test answer file. 
# In order to compare my results to that of the most frequent sense baseline, I created an additional method that would take in the raw test and train data to calcualte the most frequent sense baseline and apply that to all of the test instances. The output of this will run by scorer.py to determine the baseline accuracy. 
# ###########################################################
import sys
import re
import math
from collections import OrderedDict

stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours"," ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]

# Function that takes in a filename and iterates through each line in the file, converting it into a singular string. 
# It removes unnecessary tags
def get_clean_string(file):
  
  # Delete all newlines and trailing spaces and create one big string
  with open(file, 'r+') as f:
    lines = []
    for line in f:
      lines.append(line.strip().lower()) # remove extra white space at the beginning and end of each line

  # Combine all lines
  lines = ' '.join(line for line in lines)

  # Remove unnecessary tags from lines
  lines = re.sub(r'<s>|<\/s>|<@>|<p>|<\/p>','',lines)
  
  return lines

# Function that extracts information, including the instance_id, sense, context (sentence contents) from each sentence 
def get_info(lines, type): 
  
  # Create an instances array
  instances = re.findall(r'<instance id="(.*?)">', lines)
  
  # Create a context (sentence contents) array to generate bag of words
  context_raw = re.findall(r'<context>(.*?)<\/context>',lines)
  contexts = [re.sub(r'[.!?,;"\']', '', context).split() for context in context_raw]
  
  # If we are reading the contents of training data, create and return array of senses, instances, and contexts
  if (type == 'train'):
    senses = re.findall(r'senseid="(.*?)"\/>', lines)
    return senses, instances, contexts

  # If testing data, return just instances and contexts
  return instances, contexts

# Function that calculates log-likelihood as the absolute value of the log of the probability of sense 1 given feature i divided by the probability of sense 2 given feature i
# This probability simplies to the absolute value of the log of the frequency of sense 1 and feature i together divided by the frequency of sense 2 and feature i together
# This produces a decision list of each feature (word), prediction, and corresponding log-likelihood in ascending order by log-likelihood
def calculate_log_likely(sense_labels, sense_dict): 
  
  # Create decision dictionary
  decision_list = {}

  # Create an array of all the features in the senses dictionary
  all_features = []
  for item in sense_dict.values(): 
    all_features = all_features + list(set(word for word in item.keys()))

  # Loop through all unique features
  for feature in set(all_features):
    
    max_probability = 0
    prediction = ''
    probabilities = []
    
    # Loop through all unique senses to alculate the probability that the feature is associated with each possible sense (i.e. product and phone)
    for sense in sense_labels:
      
      # If the feature isn't in the sense dictionary (aka not in training), set probability to a really small number to prevent devision by 0 
      if feature not in sense_dict[sense]:
        probability = 0.00000000001 
      # Else, add probability of feature with sense to list
      else: 
        probability = sense_dict[sense][feature]

      # Add to list of all probabilities for current feature 
      probabilities.append(probability)

      # If the probability is higher than the current max_probability, update the max_probability value and the predicted sense for the current feature
      
      if probability > max_probability:
        max_probability = probability
        prediction = sense

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

  # Extract senses and context from training data
  senses, instances, contexts = get_info(lines, 'train')

  # Get unique senses (i.e. phone, product)
  sense_labels = set(senses)
  
  # Determine most frequent sense
  global most_frequent_sense 
  most_frequent_sense = max(sense_labels, key = senses.count)
  
  # Create sense-feature (bag of words unigram) dictionary 
  sense_dict = {}

  # Loop through all senses and contexts
  for sense, context in zip(senses, contexts):
    
    # If the sense is not in the dictionary, add it 
    if sense not in sense_dict:
      sense_dict[sense] = {}
      
    # Loop to each word in the context of the current sentence
    for word in context: 

      # If the word is not a stop word
      if word not in stop_words:
        # If the feature hasn't been assigned a sense, add it
        if word not in sense_dict[sense]:
          sense_dict[sense][word] = 0
          
        # Increment count of the feature with that sense by 1
        sense_dict[sense][word] += 1

  # Calculate the log likelihood for each possible [sense][tag] pairing and generate a  decision list with ranked features
  decision_list = calculate_log_likely(sense_labels, sense_dict)

  # Return an ordered decision list
  return decision_list

# Function that uses the ordered decision_list to add word senses to the test data and outputs log data to the model file  
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
      predicted = most_frequent_sense
  
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
        if feature in test_words:
          predicted = sense
          break
        # Otherwise, the most frequent sense will be assigned to the feature
          
      # Print answers to answer file
      print(f'<answer instance={instance} senseid="{predicted}"/>')

# Function that calculates the most frequent sense baseline and tagging it to all test set 
def calculate_mfs_baseline(train_lines, test_lines):

  # Get senses from training data
  train_senses, instances, contexts = get_info(train_lines, 'train')

  # Get all instances from testing data
  test_instances, contexts = get_info(test_lines,'test')

  # Get sense labels to calculate most_frequent sense from training
  sense_labels = set(train_senses)
  most_frequent_sense = max(sense_labels, key = train_senses.count)

  # Assign most frequent sense to every sentence in test file
  for instance in test_instances:
    print(f'<answer instance={instance} senseid="{most_frequent_sense}"/>')

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

  # Step 4: Assign word sense to testing
  predict_testing(decision_list, test_lines, model)

  # EXTRA: Compare your results to that of the most frequent sense baseline. Comment out Step 3 and 4 and then run this and then scorer.py to get most frequent sense baseline accuracy.
  # calculate_mfs_baseline(training_lines, test_lines)
  # Accuracy of most frequent baseline: 0.42857142857142855

  

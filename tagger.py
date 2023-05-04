# Tara Ram Mohan 
# CMSC 416 Programming Assignment 2: Tagger
# March 15, 2022
###########################################################
# Problem: This Python program called tagger.py will take 2 files as input (a training file containing part of speech tagged text and a test file containing text to be POS tagged). 
# It will implement the "most likely tag" baseline by assign each word in train with the POS tag that maximizes P(tag|word). 
# Using these tags, it will POS tag the test file and return it in a text file whose name is specified in the input arguments. 
# The accuracy of our POS tagger will be determined using the scorer.py file
###########################################################
# Usage: To start the program, run the following command: 
# python tagger.py (train_file.txt) (test_file.txt) > (tagged_test_file.txt) where (train_file.txt) and (test_file.txt) are actual file names of the training data and the text that you want to be tagged. 
# The program will train using train_file.txt to implement the "most likely tag" baseline, which will be used to tag the test_file.The output filename will be defined by tagged_test_file.txt.
# Example command: python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt 
# All of the following are portions (not the entire file example) of the actual file.
# As long as each word/tag pairing is separated by some whitespace, the formatting doesn't matter
# Example input train_file: 
    # [ Pierre/NNP Vinken/NNP ]
    # ,/, 
    # [ 61/CD years/NNS ]
    # old/JJ ,/, will/MD join/VB 
    # [ the/DT board/NN ]
# Example input test_file:
    # No , 
    # [ it ]
    # [ was n't Black Monday ]
    # . 
    # But while 
    # [ the New York Stock Exchange ]
    # did n't 
    # [ fall ]
# Example output tagged_test_file: 
    # No/DT ,/, it/PRP was/VBD n't/RB Black/NNP Monday/NNP ./. But/CC while/IN the/DT New/NNP York/NNP Stock/NNP Exchange/NNP did/VBD n't/RB fall/NN    
# ###########################################################
# Algorithm: 
# (1) The program will read all the files outlined in the user input. This includes the training data in pos-train.txt and the text to be tagged in post-test.txt. 
# (2) For each text (training and test), convert the contents of the corpus to a string, remove brackets since we don't use them in POS tagging. and then splits the string into a list containing each word/tag pairing
# (3) Then, by iterating through the training data in a for loop, we create an array of lists (groups) each containing word and tag in each unique word/tag pairing from train_str 
# (4) Using that array of lists, we create 3 dictionaries (word frequency dictionary, tag-word frequency dictionary, and dictionary containing all tags for each word) that will used for the P(tag|word) conditional probability calculation later.
# To create the word frequency dictionary, we iterate through each group in training. If the word in the group already exists, we increment the frequency by 1. If not, we add the word to the dictionary with a frequency of 1
# To create the word/tag frequency dictionary, we iterate through each group in training and define the word and tag values. 
# If the word/tag tuple is already in the dictionary, increment frequency by 1. If the tuple doesn't exist in the dictionary yet, add to dictionary with frequency of 1.
# (5) For each word in the training data, assign it the POS tag that maximizes P(tag|word). 
# To do so,  we iterates through all the possible word/tag pairings in training. For all the possible tags of a given word, calculate P(tag|word). 
# If the probability of this tag is higher than that of the current max tag, set current tag to max tag and update max probability. 
# After you iterate through all possible tags of a given word, set current max tag to the value of the word key in the max_dict
# (6) Now we have finished iterating through the training data, the program creates an array of lists (groups) each containing a unique word from test_str. 
# Iterating through each group in this array, if the word was in the training data and is therefore in alltags_dict, get the POS tag that maximizes P(tag|word) for that word. 
# Else, any word found in the test data but not in training data (i.e. an unknown word) is an 'NN' tag. Add the tag to the list/group containing current word and then add list to finalized array containing all words and corresponding tags
# Raw accuracy of my most likely tagger on the given test file is: 0.8453998310572998
# After this is completed, implement the tagger with each of my 5 rules one at a time within step 6 and see how those rules affect your accuracy. The rules and accuracy are as follows: 
# Rule 1: If the word is a number (or fraction), the tag will be a cardinal number (CD)
# Accuracy after Rule 1: 0.8529670561734478
# Rule 2: If the word is tagged as a noun and the first letter is capitalized, then it is a proper noun (NNP)
# Accuracy after Rule 2: 0.8495530057722089
# # Rule 3: If a word is hyphenated, then it is an adjective 
# Accuracy after Rule 3: 0.8484091229058145
# Rule 4: If a word is a proper noun and ends with an s, then it is a plural proper noun
# Accuracy after Rule 4: 0.8410178797691117
# Rule 5: If a word is a noun but ends with -ed, then it is a verb in past tense
# Accuracy after Rule 5: 0.8474588202168098
# (7) After the test file has been tagged, we output tagged word pairings into provided file by iterating through each group and connecting the word-tag pairings using a '/' and adding it into an output array.
###########################################################

import sys, re, os, random

# Function that takes in a filename and iterates through each line in the file, converting it into a singular string. 
# It then removes brackets since we don't use them for POS tagging and uses the split() function to return a list containing each word/tag pairing 
# The returned 'output_text' list will look something like this: ['stemming/VBG', 'market/NN' , ',/,'....]
def convert_to_text(file): 
    output_text = ''
    
    # Turn corpus into a singular string
    with open(file, 'r+') as f:
        lines = []
        for line in f:
            lines.append(line.strip()) # remove extra white space at the beginning and end of each line
        output_text = ' '.join(lines)
    
    # Ignore brackets for POS tagging 
    output_text = re.sub(r'[\[\]]', '', output_text)
    
    # Split string into list of word/tag pairings using default whitespace
    return output_text.split()

# Function that creates frequency dictionaries for words and tags named word_dict and tag_dict
# The word frequency dictionary will be used for the P(tag|word) conditional probability calculation later.
# word_dict will look something like this: { 'the' : 47, 'Dutch' : 100, .... }
def create_freq_dicts(groups):
    word_dict = {}
    tag_dict = {}
    
    # Iterate through each group in training
    for group in groups:
        word = group[0]
        tag = group[1]
        
        # If the word in the group already exists, we increment the frequency by 1. 
        if word in word_dict:
            word_dict[word] += 1
        # If not, we add the word to the dictionary with a frequency of 1
        else:
            word_dict[word] = 1
        
        # Repeat for tag_dict
        # NOTE: We don't use tag_dict for anything
        if tag in tag_dict:
            tag_dict[tag] += 1
        else:
            tag_dict[tag] = 1
              
    return word_dict, tag_dict

# Function that creates frequency dictionary for word-tag pairings named word_tag_dict
# The word/tag frequency dictionary will be used for the P(tag|word) conditional probability calculation later.
# word_tag_dict will look something like this: { ('the','DT'): 11, ('Dutch', 'NNP'): 45, .... }
def create_word_tag_dict(groups):
    word_tag_dict = {} # Dictionary for word|tag pairings
    
    for group in groups:
        
        # Assign values for word and tag variables for the current group
        word = group[0]
        tag = group[1]
        
        # If the word/tag tuple is already in the dictionary, increment frequency by 1. 
        if (word,tag) in word_tag_dict:
                word_tag_dict[(word,tag)] += 1
        # If the tuple doesn't exist in the dictionary yet, add to dictionary with frequency of 1.
        else:
            word_tag_dict[(word,tag)] = 1
            
    return word_tag_dict

# Function that creates a dictionary named alltags_dict of lists containing all possible tags for each word
# The alltags_dict dictionary will be used for the P(tag|word) conditional probability calculation later.
# alltags_dict  will look something like this: { 'the':['DT','JJ'] , 'group':['NN',....] .... }
def create_alltag_dict(word_tag_dict):
    alltags_dict = {}
    
    # Iterate through all the possible word/tag pairings in the word/tag frequency dictionary
    for group in word_tag_dict.keys():
        
        # Assign values for word and tag variables for the current group
        word = group[0]
        tag = group[1]
        
        # If the word is already in the alltags dictionary, just add the tag to the list of tags
        if word in alltags_dict:
            alltags_dict[word].append(tag)
        
        # If not, add the word and set its value to a list containing the tag
        else:
            alltags_dict[word] = [tag]
    return alltags_dict

# Function that calculates and returns P(tag|word) 
def calculate_probability(word, tag):
    
    # Make sure that the word tag tuple existed in training 
    if (word, tag) not in word_tag_dict:
        return 0
    
    # Calculate probability using conditional probability P(tag|word) = P(tag & word) / P(word)
    probability = float(word_tag_dict[(word, tag)])/float(word_dict[word])
    
    return probability

# Function that creates a dictionary where for each word in the training data, we assign it the POS tag that maximizes P(tag|word).
def create_max_dict(groups, alltags_dict):
    
    max_dict = {}
    
    # Iterates through all the possible word/tag pairings in training 
    for group in groups:
        max = 0
        tag = ''
        word = group[0]
        
        # For all the possible tags of a given word
        for curr_tag in alltags_dict[word]:
            
            # Calculate P(tag|word)
            probability = calculate_probability(word, curr_tag)
            
            # If the probability of this tag is higher than that of the current max tag, set current tag to max tag and update max probability
            if probability >= max:
                max = probability
                tag = curr_tag
                
        # After you iterate through all possible tags of a given word, set current max tag to the value of the word key in the max_dict
        max_dict[word] = tag
    
    return max_dict

# Function that POS tags array of lists containing words from testing 
def tag_words(groups, max_dict, alltags_dict):
    tagged_test = []
    
    for group in groups:
        word = group[0]
        tag = ''
        
        # If the word was in the training data and is therefore in alltags_dict, get the POS tag that maximizes P(tag|word) for that word
        if word in alltags_dict:
            tag = max_dict[word]
        # Else, any word found in the test data but not in training data (ie an unknown word) is an NN
        else:
            tag = 'NN'
        
        # Raw accuracy:  0.8453998310572998

        # Note that accuracy was calculated by adding rule 1 see accuracy, removing rule 1 and then adding rule 2 to see accuracy...
        # Rule 1: If the word is a number (or fraction), the tag will be a cardinal number (CD)
        # Accuracy after Rule 1: 0.8529670561734478
        # if (re.search(r'(\d+[.\/]?\d?)',word)): 
        #     tag = 'CD'
        
        # Rule 2: If the word is tagged as a noun and the first letter is capitalized, then it is a proper noun (NNP)
        # Accuracy after Rule 2: 0.8495530057722089
        # if (tag == 'NN' and word.isupper()):
        #     tag = 'NNP'
        
        # # Rule 3: If a word is hyphenated, then it is an adjective 
        # Accuracy after Rule 3: 0.8484091229058145
        # if (re.search(r'\b\w+\b-\b\w+\b',word)):
        #     print("HA")
        #     tag = 'JJ'
        
        # # Rule 4: If a word is a proper noun and ends with an s, then it is a plural proper noun
        # Accuracy after Rule 4: 0.8410178797691117
        # if (tag == 'NNP' and re.search(r'\b\w+s\b',word)):
        #     print("HA")
        #     tag = 'NNPS'
        
        # # Rule 5: If a word is a noun but ends with -ed, then it is a verb in past tense
        # Accuracy after Rule 5: 0.8474588202168098
        # if (tag == 'NN' and re.search(r'\b\w+ed\b',word)):
        #     print("HA")
        #     tag = 'VBD'
        
        # Add tag to list containing current word 
        group.append(tag)
        
        # Add list to tagged_test array 
        tagged_test.append(group)
        
    return tagged_test

        
# Main method where program starts
if __name__ == '__main__':
    
    # Step 1: Read in command line args
    training_file = sys.argv[1] # training file containing part of speech tagged text
    test_file = sys.argv[2] # file containing text to be part of speech tagged

    # Step 2: Convert both the contents of the train and test files into one string
    train_str = convert_to_text(training_file)
    test_str = convert_to_text(test_file)

    # Step 3: Creates an array 'train_groups' of lists (groups) each containing word and tag in each unique word/tag pairing from train_str 
    # train_groups will look something like this: ( ['the','DT'], ['Dutch','NP'], ... ])
    train_groups = []  
    for pairing in train_str:
        group = pairing.split('/')
        
        # Accounts for  "ambiguous" tags where we only use the first part of speech tag and ignore the rest
        temp = group[1].split('|')
        group[1] = temp[-1] 
        train_groups.append(group)

    # Step 4: Create 3 dictionaries for P(tag|word) equation calculation
    word_dict, tag_dict = create_freq_dicts(train_groups) # dictionaries of unigram frequencies of words and tags
    word_tag_dict = create_word_tag_dict(train_groups) # dictionary of the frequency of each word-tag pairing
    alltags_dict = create_alltag_dict(word_tag_dict) # dictionary of lists of all possible tags for each word
    
    # Step 5: For each word in the training data, assign it the POS tag that maximizes P(tag|word) 
    max_dict = create_max_dict(train_groups,alltags_dict) # dictionary containing every word and the tag that maximizes it
    
    # Creates an array of lists (groups) each containing a unique word from test_str 
    test_groups = []
    for word in test_str:
        test_groups.append([word])
      
    # Step 6: Add tags to every word group in the test string  
    tagged_test = tag_words(test_groups, max_dict, alltags_dict)

    # Step 7: Output tagged word pairings into provided file
    words = []
    for trip in test_groups:
        words.append(trip[0] + '/' + trip[1])
    output = ' '.join(words)
    
    print(output)
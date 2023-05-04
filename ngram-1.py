# Tara Ram Mohan 
# CMSC 416 Programming Assignment 2: Ngram
# February 22, 2021
###########################################################
# Problem: This Python program will learn an n-gram language model given an arbitrary number of plain text files from which it will generate a given number of sentences based on that N-gram model.
###########################################################
# Usage: Run the following command: python3 ngram4.py [n] [m] [input-file/s] where n is the number of words in a 'gram', m is the number of output sentences, and input-file/s should be a list of one or more file names that contain the text you are building your ngram model from. Note that m and n must be positive integer values. The program will then output m sentences. 
# Example: The command 'ngram.py 3 8 book1.txt book2.txt' should result in 8 randomly generated sentences based on a tri-gram model learned from these 2 files. 
# Expected output: 
# This program generates random sentences based on an Ngram model.
# Command line settings : ngram.py   3   8
# (1) Oh, thanks, im afraid, said the mock turtle went on at last.
# (2) The chief difficulty alice found at first he thought his thoughts.
# (3) She cried.
# (4) Alice said nothing.
# (5) He might have pitied her so little, weak voice.
# (6) Nobody took any notice of him.
# (7) When the rabbit sends in a tone of this work.
# (8) Old and forgotten fancies of his pockets, his arms folded, frowning a little lunch first.

###########################################################
# Algorithm: 
# (1) The program will read all the files outlined in the user input and turn them into one string for the corpus.
# (2) The program converts the entire string to lowercase and and adds start and end tags. To add tags, we define a start variable with n-1 <s> using a 'for' loop. We then use regex to substitute .!? with an endtag <end>. Then, we push end tag and n-1 start variable <start> to end of corpus string 
# (3) To create the actual ngrams, we must split on white space where each word is an element in the ngram array. 
# (4) For each element in corpus_array, we will populate create an ngram and history dictionary, which will be used later to calculate conditional probability. In both, we will check if (length(ngram_array)  == n ). If true, for hisotry dictionary, we will check if the history is already in the dictionary, and if not, we will add it and set frequency to one. If it is, we will increase the frequency for that key. We repeat this with the ngram dictionary but using a nested if statement, checking to see if the history and word pairing is in the dictionary. 
# (5) In order to generate a sentence, we will add words to a sentence until we pick an end tag <e>. We start with an empty history. From there, we use the ngram dictionary to find the frequency of each word given this history. For each word, we calculate conditional probability P[history][word] / P[word]. If the probability is greater than a randomly generated number, we pick that word. Else, we continue to the next word, add the probability of that word to the running probability sum, and recheck if sum > random. This process repeats until we read an end tag in our sentence array. 
# (6) We output m sentences created using step 5. 
# For unigrams, we simply create a frequency table of each individual word in the corpus. To generate the sentence, we repeat step (5) but probability = density of a word in the corpus, which calculated by dividing the total number of words in the corpus but the frequency of the selected word. 
###########################################################

import sys
import re
import random

# Function that converts raw_text into lowercase and removes all punctuation other than . ! ?
def clean(raw_text):

  # Removes all punctuation other than . ! ?
  # Note contractions are combined into one word (EX. you'll = youll)
  raw_text = re.sub(r'[^\'\w\s.!?,-]', '', raw_text)
  raw_text = re.sub(r'\n', ' ', raw_text)

  # Convert all text to lower case
  return raw_text.lower()

# Function that replaces tags and adds proper capitalization and spacing AFTER sentence has been generated
def reformat(sentence): 

  # Change start tag to blank 
  sentence = sentence.replace('<s>', '')

  # Change end tag to punctuation
  sentence = sentence.replace(' <e>', '.')

  # Add proper punctuation and spacing
  sentence = sentence.replace(' i ', ' I ')
  sentence = sentence.replace(' , ', ', ')
  sentence = sentence.strip()

  return sentence.capitalize()
  
# Function that adds start and end tags to the raw_text
def tag(raw_text,n):
  
  # Define n-1 start variable 
  start_tag = ''
  for i in range(n - 1):
      start_tag = start_tag + ' <s> ' # make sure to add whitespace for delimiter

  # Push start variable <s> to the beginning of the raw_text string 
  raw_text = start_tag + raw_text

  # Push end tag and n-1 start variable <start> to end of raw_text string 
  end_tag = ' <e> '
  tag = end_tag + start_tag

  # Turn comma into a token
  raw_text = raw_text.replace(', ', ' , ')

  # Accounts for nonpunctuation periods
  raw_text = raw_text.replace('Mr.', 'Mr ')
  raw_text = raw_text.replace('Mrs.', 'Mrs ')
  raw_text = raw_text.replace('Ms.', 'Ms ')
  raw_text = raw_text.replace('Dr.', 'Dr ')
  raw_text = raw_text.replace('. . . ', '... ')
  raw_text = raw_text.replace('. . ', '.. ')

  # Add end tags to .!?
  raw_text = raw_text.replace('. ', tag)
  raw_text = raw_text.replace('! ', tag)
  raw_text = raw_text.replace('? ', tag)
  
  raw_text = re.sub(r'  ', ' ', raw_text)

  return raw_text

# Function to convert raw text into a corpus array
def create_corpus_arr(raw_text, n):

  # Remove punctatuation and convert text to lowercase
  raw_text = clean(raw_text)

  # Add start and end tags
  raw_text = tag(raw_text,n)

  # For each element in raw_text, you will populate n-gram array. 
  token_arr = []
  for token in raw_text.split(' '):
    if token != '':
      token_arr.append(token)

  # Create ngram sequences of n word and store in ngrams
  ngrams = []
  for i in range(n):
    ngram_item = token_arr[i:]
    ngrams.append(ngram_item)

  # Create pairings
  corpus_arr = zip(*ngrams)

  # Combines all the ngram arrays into strings and adds them to the final 
  corpus_arr_final = []
  for ngram in corpus_arr:
    ngram_item = ' '.join(ngram)
    corpus_arr_final.append(ngram_item)

  return corpus_arr_final

# Function used to create frequency dictionary for unigrams 
def create_freq_dict(ngrams,n):
  freq_dict = {}

  # For all ngrams in ngram object
  for i in range(len(ngrams)):
    current_word = ngrams[i] #maybe this is wrong
    if current_word in freq_dict:
      freq_dict[current_word] += 1
    else: 
      freq_dict[current_word] = 1

# Function used to created the ngram and history dictionaries
def create_dictionaries(ngrams, n):
  ngram_dict = {}
  history_dict = {}

  # For all ngram arrays in ngrams object
  for i in range(len(ngrams)):

    # If string is less then n, break out of for loop
    word_list = ngrams[i].split()

    # If string length is less then n (smaller than ngram), break out of for loop
    if len(word_list) == n: 
       # Pop last element in ngram array 
      word = str(word_list[-1])
  
      # Create history string with remaining elements in ngram array
      history_list = word_list[:(n - 1)]
      history = ' '.join(history_list)
  
      # Add occurance to ngram table
      # If the selected history is already in history dictionary
      if history in ngram_dict:  
        # If the selected history is already in history dictionary, increment frequency by 1
        history_dict[history] += 1
        
        # AND if word + history pairing is already in history dictionary, increment frequency by 1
        if word in ngram_dict[history]:
            ngram_dict[history][word] += 1 
        # Else, add pairing occurance with a frequency value of 1
        else: 
            ngram_dict[history][word] = 1
      else:
        # If the selected history is NOT already in history dictionary, add to history dictionary and set frequency value to 1  
        history_dict[history] = 1

        # If the selected history is NOT already in history dictionary, add pairing to dictionary
        ngram_dict[history] = {} 
        ngram_dict[history][word] = 1
      
  return ngram_dict, history_dict
  
# Function used to create sentences for non-unigram ngrams
def generate_sentence(ngram_dict, history_dict, n):
  curr_word = []
  sentence = ''
  history = ''

  # Add n-1 start tag to curr_word array
  for i in range(n - 1):
    curr_word.append('<s>')

  # Continue adding words until you pick end tag 
  while '<e>' not in curr_word:

    # Create string of history 
    history = ' '.join(curr_word)
    
    # Get frequency of words based on history
    word_list = ngram_dict[history]

    # Get frequency of history 
    freq_hist = history_dict[history]
    
    counter = 0
    
    # Get random number 
    rand = random.uniform(0, 1)

    for word in word_list:
      
      # Calculate conditional probability = P[history][word] / P[word]
      probability = word_list[word] / freq_hist
      
      # Add to running counter value
      counter = counter + probability

      # If counter > random, then pick that word
      if counter > rand:

        # Store current word
        curr_word.append(word)

        # Add previous word to sentence
        sentence = sentence + curr_word.pop(0) + ' '
        break 

  # Add final word to sentence
  sentence += ' '.join(curr_word)

  # Reformat sentence for print
  sentence = reformat(sentence)
  
  return sentence
  
# Function that is used to call the generate sentence function to create the desired number of sentences
def output_sentences(m, ngram_dict, history_dict, n):

  # Create m sentences and print it out with numbered list
  for m in range(m):
    sentence = generate_sentence(ngram_dict, history_dict, n)
    print('(' + str(m+1) + ') ' + sentence)
    
def create_unigram_freq_dict(ngrams):
  freq_dict = {}

  # For all words 
  for current in ngrams:

    # If already in dictionary, increment frequency by 1 
    if current in freq_dict:
      freq_dict[current] += 1

    # If not, add to dictionary with frequency of 1
    else: 
      freq_dict[current] = 1

  return freq_dict
      
# Function that is used to generate sentences specifically for unigrams
def create_unigram_sentence(freq_dict):
  curr_word = []
  sentence = ''
  total_count = sum(freq_dict.values()) - freq_dict['<e>']

  # Add 1 start tag to curr_word
  curr_word.append('<s>')
    
  # Continue adding words until you pick end tag 
  while '<e>' not in curr_word:

    counter = 0
    
    # Get random number 
    ran_num = random.uniform(0, 1)
    
    for word in freq_dict:
      
      # Calculate conditional probability = P[word] / total_count
      probability = freq_dict[word] / total_count

      # Add to running counter
      counter = counter + probability

      # If counter > random, then pick that word
      if ran_num < counter:

        # Store current word
        curr_word.append(word) 

        # Add previous word to sentence
        sentence = sentence + curr_word.pop(0) + ' '
        break

  # Add final word to sentence
  sentence += ' '.join(curr_word)

  # Reformat sentence for print
  sentence = reformat(sentence)

  return sentence
  
def output_unigram_sentences(freq_dict, m):

  # Create m sentences and print it out with numbered list
  for m in range(m):
    sentence = create_unigram_sentence(freq_dict)
    print('(' + str(m+1) + ') ' + sentence)
  
if __name__ == '__main__':

  # Step 1: Read in command line args
  n = int(sys.argv[1])
  m = int(sys.argv[2])
  input_files = sys.argv[3:]
  
  print('This program generates random sentences based on an Ngram model.')
  print('Command line settings : ngram.py ', str(n), ' ', str(m))

  # Step 2: Convert all input files into one string
  raw_text = ''
  for curr_file in input_files:
    with open(curr_file, 'r') as fp:
        data = fp.read()
        raw_text += data

  # If unigram:
  if n == 1:
    # Create all possible ngrams
    ngrams = create_corpus_arr(raw_text, n)
    
    # Create frequency dictionary
    freq_dict = create_unigram_freq_dict(ngrams)
    # Print out unigram sentences
    output_unigram_sentences(freq_dict, m)
      
  # If any other ngram:
  elif n > 1:
    # Create all possible ngrams
    ngrams = create_corpus_arr(raw_text, n) 

    # Create ngram dictionary [history][word] and history dictionary [history]
    ngram_dict, history_dict = create_dictionaries(ngrams, n)

    # Print out ngram sentences
    output_sentences(m, ngram_dict, history_dict, n)

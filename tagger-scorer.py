# Tara Ram Mohan 
# CMSC 416 Programming Assignment 2: Scorer
# March 15, 2022
###########################################################
# Problem: This Python program called scorer.py will take 2 files as input (a pos tagged output generated from tagger.py and the gold standard "key" data). 
# It will report the overall accuracy of your tagging based on the gold standard and provide a confusion matrix as outputted into the STDOUT defined file. 
###########################################################
# Usage: To start the program, run the following command: 
# python scorer.py (tagged_test_file.txt) (key_file.txt) > (scorer_report.txt) where (tagged_test_file.txt) is the tagged test file generated from tagger.py and (key_file.txt) is the gold standard key. 
# (scorer_report.txt) will be the name of the output file that contains the generated accuracy and confusion matrix.
# Example command: python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-tagging-report.txt
# All of the following are portions (not the entire file example) of the actual file.
# As long as each word/tag pairing is separated by some whitespace, the formatting doesn't matter
# Example input tagged_test_file: 
    # No/DT ,/, it/PRP was/VBD n't/RB Black/NNP Monday/NNP ./. But/CC while/IN the/DT New/NNP York/NNP Stock/NNP Exchange/NNP did/VBD n't/RB fall/NN    
# Example input key_file:
    # No/RB ,/, 
    # [ it/PRP ]
    # [ was/VBD n't/RB Black/NNP Monday/NNP ]
    # ./. 
    # But/CC while/IN 
    # [ the/DT New/NNP York/NNP Stock/NNP Exchange/NNP ]
    # did/VBD n't/RB 
    # [ fall/VB ]
# Example output scorer_report: 
    #Accuracy: 0.846438124736027
    # Confusion Matrix:
    #                       Actual: DT  Actual: ,  Actual: PRP  Actual: VBD  \
    # Predicted: DT              4747          0            0            0   
    # Predicted: ,                  0       3070            0            0   
    # Predicted: PRP                0          0         1042            0   
    # Predicted: VBD                0          0            0         1498   
    # Predicted: RB                 0          0            0            0   
    # Predicted: NNP                0          0            0            0   
# ###########################################################
# Algorithm: 
# (1) The program will read all the files outlined in the user input. This includes my pos tagged output and the gold standard text output
# (2) For each text file, the program iterates through each line in the file, converting it into a singular string and removing brackets.
# Then, the program converts the string into an array of tags by splitting the string by default whitespace into groups and then splitting the groups into word and tag using '/' and adding the tag into the final array. 
# This step will produce 2 arrays containing the actual tags from my pos tagged output and the expected tags from the gold standard text output
# (3) The program calculates and outputs accuracy given actual and expected tags. 
# To do so, we iterate through the actual tags array counting the total number of tags as (TP + TN + FP + FN). For each 'actual' tag that matches the corresponding 'expected' tag, we incremeent (TP + TN) by 1. 
# Then, we calculate accuracy as (TP + TN) / (TP + TN + FP + FN)
# (4) The program creates confusion matrix given actual and expected tags. First, it create a list of indices containing each unique tag, which will be used as labels.
# Then, the actual confusion matrix will be created using sklearn.metrics, the array of actual tags, and the array of expected tags and then converted into a pandas dataframe to be outputted.
###########################################################

import sys, re
import pandas as pd 
from sklearn.metrics import confusion_matrix


# Function that takes in a filename and iterates through each line in the file, converting it into a singular string. 
# It then removes brackets since we don't use them for POS tagging and return a string containing each word/tag pairing. 
# The returned 'output_text' list will look something like this: 'stemming/VBG market/NN ,/, ...
def convert_to_text(file): 

    output_text = ''
    with open(file, 'r+') as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        output_text = ' '.join(lines)
    
    # Ignore brackets for POS tagging 
    output_text = re.sub(r'[\[\]]', '', output_text)

    return output_text


# Function that takes in the raw string from convert_to_text as input and creates an array of tags from each word-tag pairing
# This will be implemented on both the actual output from tagger.py and the gold standard
# The returned 'tags_list' will look something like this: ['DT', ',' , 'PT' ... ]
def convert_to_groups(output_text):

    # Split string into list of word/tag pairings using default whitespace
    groups = output_text.split()
    tags_list = []
    
    # For each word/tag pairing
    for curr_group in groups:
        
        # Split word and tag 
        group = curr_group.split('/')
        
        # Accounts for  "ambiguous" tags where we only use the first part of speech tag and ignore the rest
        temp = group[1].split('|')
        group[1] = temp[-1]
        
        # Add just the tag to the tag list
        tags_list.append(group[1])
        
    return tags_list

# Function that will take in the accuracy using the following equation: accuracy = (TP + TN) / (TP + TN + FP + FN)
def calculate_accuracy(actual, expected):
    tp_tn = 0
    tp_tn_fp_fn = 0
    
    # For each tag in my 'actual' tagged output
    for i in range(len(actual)):
        
        # Add to total num of tags (TP + TN + FP + FN)
        tp_tn_fp_fn += 1 
        
        # If expected and actual are the same, add to correct output count (tp+tn)
        if actual[i] == expected[i]:
            tp_tn+=1
            
    return tp_tn/tp_tn_fp_fn
  
# Function that creates 
def create_cm(actual,expected):
    
    indices = list()
    
    # Create a list of indices containing each unique tag 
    for curr in actual:
      if curr not in indices:
        indices.append(curr)
      
    # Using sklearn.metrics, create a confusion matrix
    cm = confusion_matrix(actual,expected, labels = indices)

    # Format the confusion matrix appropriately with the right column and row headers
    cm_df = pd.DataFrame(cm,
                     index = ['Predicted: ' + i for i in indices], 
                     columns = ['Actual: ' + i for i in indices] )
  
    return indices, cm, cm_df
    
if __name__ == '__main__':
  
    # Step 1: Read in command line args
    my_file = sys.argv[1] # My pos tagged output
    gold_file = sys.argv[2] # Gold standard key output

    # Step 2: Convert contents of each file into a string and get an array of just tags 
    actual_tags = convert_to_groups(convert_to_text(my_file))
    expected_tags = convert_to_groups(convert_to_text(gold_file))

    # Step 3: Calculate and output accuracy given actual and expected tags
    print('Accuracy:', calculate_accuracy(actual_tags, expected_tags), end='\n')
    
    # Step 4: Create confusion matrix given actual and expected tags
    indices, cm, cm_df = create_cm(actual_tags, expected_tags)

    # Step 5: Output confusion matrix 
    pd.set_option("display.max_rows", None, "display.max_columns", None) # Show ALL columns of the matrix instead of middle bit getting cut off
    print('Confusion Matrix:\n',cm_df)

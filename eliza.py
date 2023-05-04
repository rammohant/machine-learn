# Tara Ram Mohan 
# CMSC 416 Project 1: ELIZA
# February 8, 2021
# Sources consulted: 
# Weizenbaum Paper - https://dl.acm.org/doi/10.1145/365153.365168
# Online ELIZA Chatbot - https://web.njit.edu/~ronkowit/eliza.html
###########################################################
# Problem: This program immitates a very basic Rogerian psychotherapist chatbot. It takes in your input (discussion of your problems) and asks follow-up questions to prompt introspective thinking.
###########################################################
# Usage: Run eliza.py. ELIZA will immeditately introduce itself and ask questions to prompt user input. Answer the questions realistically and appropriately in non-compound, singular sentence responses. User should not ask questions to ELIZA, should not use contractions, and should speak only in the present tense. The program will continue indefinitely until the user forcefully ends the program. Below are example inputs and outputs. 
# Acceptable EX1: 
# [Tara] "My sister hates me." 
# [ELIZA] "Are you close with your family" 
# Acceptable EX2: 
# [Tara] "i am depressed!" 
# [ELIZA] "Have you considered getting a real therapist to talk about these negative emotions with? You know I'm not real, right?"
###########################################################
# Algorithm: 
# In the main method, (1) the program calls a function where ELIZA introduces itself and asks for user's name in the first user input. Note that every user input is cleaned by removing all punctation and converting to lowercase in the 'clean' method. Using the get_name function, the program searches for the keywords 'am' or 'is' in the input,and if found, stores the first word immediately following the keywords as the name. If not, stores the entire user input as name. Using user's name, ELIZA says "Hi" and asks first question. 
# Following the first user input, (2) the program cleans the next user input and generates a reply. To do so, the cleaned input goes through the 'reply' function, which searches through the keys in the 'rules' dictionary for a matching regex expression. If a matching expression is found, then a random output option is selected from the corresponding value in the dictionary. (3) In the matching expression, if there are personal pronounds (i.e. "I", "you", "myself"), these are switched from first person perspective to second person perspective and vice person. To do so, the raw matching expression is split into individual words in an array using the 'switch_perspective' function. Iterating through the words array, if any word matches a key found in the 'perspective' dictionary, it is replaced with the corresponding value in the array. The finalized words array is then combined into a sentence and returned. (4) The finalized matching expression then replaces the temporary placeholder in the random output option selected in the 'reply' function. This reply is then printed out in the main method.
# Unles the user types 'quit', STEP (2) through (4) loop in the main method as long as there is user input. 
###########################################################
import re,random

###########################################################
# DICTIONARIES 
###########################################################

# Dictionary used by 'switch_perspective' method to change the perspective from that of the user to that of ELIZA. The dictionary key (user input) will replaced with the value in the ELIZA output. 
perspective = {
  # nouns
  'i' : 'you',
  'you' : 'me',
  'we' : 'you all',
  'me' : 'you',
  'us' : 'you all',
  # possessive pronouns
  'my' : 'your',
  'our' : 'your',
  'your' : 'my',
  'myself' : 'yourself',
  'yourself' : 'myself',
  # verbs (to be) 
  'are' : 'am',
  'am' : 'are'
}

# Dictionary used by 'reply' method to match the user input with a regex expression and identify capturing groups for ELIZA output. The key is the user input, and the value includes all options for ELIZA output in an array. 
perspective = {
  # nouns
  'i' : 'you',
  'you' : 'me',
  'we' : 'you all',
  'me' : 'you',
  'us' : 'you all',
  # possessive pronouns
  'my' : 'your',
  'our' : 'your',
  'your' : 'my',
  'myself' : 'yourself',
  'yourself' : 'myself',
  # verbs (to be) 
  'are' : 'am',
  'am' : 'are'
}

# Dictionary used by 'reply' method to match the user input with a regex expression and identify capturing groups for ELIZA output. The key is the user input, and the value includes all options for ELIZA output in an array. 
# NOTE: Rules 6 and 8-14 all transform statements into questions to account for more general statements 
rules = {
  # Rule 1 for all inputs containing the noun 'mom' or 'mother' singular or plural. Returns a 'yo momma' joke just to add a little humor to ELIZA
  r'.*(\bmoms?\b|\bmothers?\b).*$':
    [
      'Yo mamma so ... sorry sorry, I got distracted. Tell me more about your family.'
    ]
  ,
  # Rule 2 for all inputs containing nouns relating to family members since family can be a key talking point for most people
  r'(moms?\b|\bdads?\b|\bmothers?\b|\bfathers?\b|\brothers?\b|\bsisters?\b|\bstepmoms?\b|\bstepmother?\b|\bstepdads?\b|\bstepfather?\b|\bstepbrothers?\b|\bstepsisters?\b|\bgrandparents?\b\baunts?\b|\buncles?\b|\bgrandmas?\b|\bgrandpas?\b|\bgrandmothers?\b|\bgrandfathers?\b).*$' :
    [
      'Tell me more about your family.',
      'Are you close with your family?',
      'How often do you reach out to your family?'
    ]
  ,
  # Rule 3 for all inputs where the user indicates that they have strong negative emotions (i.e. depressed, suicidal, cry, lonely, etc.) relating to mental health. Input must include 'I' or 'my' (i.e. (i|my) ) and a strong negative emotion keyword to show that the user themselves is experiencing these emotions
  r'^.*(i|my).*(\bdepressed\b|\bdepression\b|\bsuicidal\b|\bsuicide\b|\bcry\b|\blonely\b).*$' :
    [ 
      'Have you considered getting a real therapist to talk about these negative emotions with? You know I\'m not real, right?',
      'Have you talked to someone about these negative feelings before?'    
    ]
  ,
  # Rule 4 for all inputs containing other emotion keywords (i.e. angry, happy, sad, etc.) experienced by the user (i.e. (i|my) ) 
  r'^.*(i|my).*(\bhappy\b|\bsad\b|\bangry\b|\bconfused\b|\benergized\b|\bannoyed\b|\bstressed\b).*$':
    [
      'What is causing you to feel TEMP?', 
      'What normally causes you to feel TEMP',
      'Are you experiencing any other emotions?'
    ]
  ,
  # Rule 4 for all inputs containing body image/weight related keywords (i.e. body, weight, fat, etc.) experienced by the user (i.e. (i|my) ) 
  r'^.*(i|my).*(\bbody\b|\bweight\b|\bheavier\b|\bfat\b|\bfatter\b|\bskinnier\b|\bskinny\b).*$':
    [
      'Do you have issues with your body image?', 
      'Why do you think you perceive yourself this way?'
    ]
  ,
  # Rule 5 for all inputs containing absolute words (i.e. No, Yes, Never, etc.). ELIZA asks a follow-up question to see if the user actually feels that strongly or is just exaggerating. 
  r'(\bno\b|\byesb\b|.*\bnever\b|.*\balways\b|.*\butterly\b|.*\bcompletely\b|.*\bcertainly\b|\bwithout\sa\sdoubt\b).*$':
    [
      'You seem pretty certain. Why is that?',
      'Are you sure?',
      'Why is it so clear to you?',
    ]
  ,
  # Rule 17 for extremely strong emotion words (i.e. hate, detest, love, etc.). ELIZA asks a follow-up question to see if the user actually feels that strongly or is just exaggerating. 
  r'.*(\bhate\b|\bloathe\b|\bdetest\b|\bdespise\b|\blove\b|\bso\smuch\b|.*\breally\b|.*\bvery\b|.*\bextremely\b).*$':
    [
      'Wow, \'TEMP\' is a strong sentiment. Do you really mean it?',
      'Why do you feel so strongly about that?'
    ]
  ,
  # Rule 6 for all inputs talking about an object possessed by the owner or in other words, an object with the 'our' or 'me' possessive adjectives. First capturing group captures the noun being 'possessed' by the user
  r'(\bour\b|\bmy\b)\s(\w+)':
    [
      'What do you like about your TEMP?',
      'How does your TEMP make you feel?', 
      'Tell me more about your TEMP.',
      'Do you have any strong memories associated with your TEMP?'
    ]
  ,
  # Rule 7 for all inputs containing the word 'you'
  r'\byou\b':
    [
      'Don\'t think you can change the subject to me that easily.',
      'We are here to discuss you, not me. Nice try though.',
      'Interesting. Why did you bring me up? We are here to talk about you.',
      'Let\'s talk about you.'
    ]
  ,
  # Rule 7 for all inputs containing time-related phrases
  r'(a\slong\stime|a\swhile\ago|way\sback)':
    [
      'Tell me about when this started.', 
      'How long has it been?'
    ]
  ,
  # Rule 8 for all inputs containing the word 'am'. First capturing group () captures all user input after the word 'am'. EX. I am (going to cry)
  r'\bam\b\s(.*)$':
    [
      'Do you believe it is normal to be TEMP?',
      'How long have you been TEMP?',
      'Why do you think you are TEMP?',
      'Did you come to me because you are TEMP?', 
    ]
  ,
  # Rule 9 for all inputs containing the words 'i was'. First capturing group () captures all user input after the word 'was'. EX. I was (going to cry)
  r'\bi\swas\b\s(.*)$':
    [
      'Tell me more about how you were TEMP.',
      'Why do you think you were TEMP?'
    ]
  ,
  # Rule 10 for all inputs containing the word 'is'. First capturing group () captures all user input after the word 'is'. EX. He is (going to cry)
  r'\bis\b\s(.*)$':
    [
      'Does it please you to believe they are TEMP?',
      'What caused you to feel that they are TEMP?'
    ]
  ,
  # Rule 11 for all inputs containing the word 'i'. First capturing group () captures all user input after the word 'i'. EX. I (cried)
  r'\bi\b\s(.*)$':
    [
      'Why do you think you TEMP?',
      'Tell me more about how you TEMP.'
    ]
  ,  
  # Rule 12 for all inputs containing the word 'they'. First capturing group () captures all user input after the word 'they'. EX. They (cried)
  r'\bthey\b\s(.*)$':
    [
      'Why do you think they TEMP?',
      'Can you do anything to change that they TEMP?',
      'How do you feel when they TEMP?', 
      'Does it please you to believe that they TEMP?',
      'What makes you think they TEMP?',
      'Why is it significant that they TEMP?'
    ]
  ,
  # Rule 12 for all inputs containing the word 'he'. First capturing group () captures all user input after the word 'he'. EX. he (cried)
  r'\bhe\b\s(.*)$':
    [
      'Why do you think he TEMP?',
      'Can you do anything to change that he TEMP?',
      'How do you feel when he TEMP?', 
      'Does it please you to believe that he TEMP?'
      'What makes you think they TEMP?',
      'Why is it important that he TEMP?'
    ]
  ,
  # Rule 13 for all inputs containing the word 'she'. First capturing group () captures all user input after the word 'she'. EX. She (cried)
  r'\bshe\b\s(.*)$':
    [
      'Why do you think she TEMP?',
      'Can you do anything to change that she TEMP?',
      'How do it make you feel when she TEMP?', 
      'Does it please you to believe that she TEMP?',
      'What makes you think she TEMP?',
      'Why is it significant that she TEMP?'
    ]
  ,
  # Rule 14 for all inputs containing the word 'we'. First capturing group () captures all user input after the word 'we'. EX. we (cried)
  r'\bwe\b\s(.*)$':
    [
      'Why do you think you all TEMP?',
      'Can you do anything to change that you all TEMP?',
      'How do it make you feel when you all TEMP?', 
      'Does it please you to believe that you all TEMP?',
      'What makes you think you all TEMP?',
      'Why is it so important to you that you all TEMP?'
    ]
  ,
  # Rule 15 for all inputs containing the word 'it'. First capturing group () captures all user input after the word 'it'. EX. It (cried)
  r'\bit\b\s(.*)$':
    [
      'Why is it so important to you?',
      'Do you have any strong memories with it?'
    ]
  ,
  # Rule 16 acts as a catch all for all other inputs
  r'.*?':
    [
      'Interesting...', 
      'Fascinating...',
      'I don\'t quite understand, can you rephrase that?',
      'I am not following. Can you explain a little differently?',
      'Could you elaborate on that?', 
      'Tell me more...', 
      'Please go on', 
      'Please continue'
    ]
}
###########################################################
# FUNCTIONS 
###########################################################

# The 'clean' function takes a cleaned user input as a parameter. It then removes all punctation and converts user input into lowercase to be used in other steps.
def clean(input):

  # Uses regex to search if the input contains any punctation (regex expression [^\w\s]+ includes any character other than word or space characters) at the end of the line. If found, stores all input (.*?) prior to that punctuation in 'output_punct'
  output_punct = re.search(r'(.*?)[^\w\s]+$', input)
  if output_punct is not None:
      output_punct = output_punct.group(1)
  else: 
      output_punct = input

  # Converts output_punct into a string and converts the entire string to lowercase
  output_punct = str(output_punct)
  output_lc = output_punct.lower()

  return output_lc

# STEP 1a: The 'get_name' function takes in cleaned user input as a parameter and returns the stripped user's name. 
def get_name(input):

  # Uses regex to search if input contains 'am' (EX. I am Tara) using .*\bam\b\s. If found, it stores the following first word from the ([^\s]+) capturing group in 'output'
  if re.search(r'\bam\b', input) is not None: 
    name = re.search(r'.*\bam\b\s([^\s]+)\s?.*$', input) 
    output = name.group(1)
  # Uses regex to search if input contains 'is' (EX. My name is Tara) using .*\bis\b\s. If found, it stores the following first word from the ([^\s]+) capturing group in 'output'
  elif re.search(r'\bis\b', input) is not None: 
      name = re.search(r'.*\bis\b\s([^\s]+)\s?.*$', input) 
      output = name.group(1)
  # If neither 'am' or 'is' is found in user input, stores the entire input as the name output
  else: 
      output = input

  # Capitalizes the first letter of the name output and returns it
  return output.capitalize() 

# The 'switch_perspective' method takes in an array of all the words in the response output and replaces first person personal pronouns with second person and vice versa. Uses the 'perspective' dictionary to do so.
def switch_perspective(match): 

  # Split match into individual words and store in 'words' array
  words = re.split(r'\s', match)

  # Iterates through 'words' array and searches through all the words for words that depend on first/second person perspective, such as 'you', 'i', 'us', etc. 
  for index, word in enumerate(words):
    for original, switched in perspective.items():
      # If a personal pronoun is found, this method replaces that word with the appropriate one.
      if word == original:
          words[index] = switched

  # Combine all the words in the now switched 'words' string array into a sentence
  match_finalized = ' '.join(str(word) for word in words)

  return match_finalized

# The 'response' function takes a cleaned user input as a parameter and returns an appropriate reply using the 'rules' dictionary.
def reply(input):

  # Search through the keys in the 'rules' dictionary for a matching regex expression
  for regex, response in rules.items():
    
    match = re.search(regex, input)
    # If a matching expression is found
    # NOTE: The priority is from top of dictionary to bottom of the dictionary. In other words, the top most rule that matches to the user input will generate an output. 
    if match is not None:

      # NOTE: all my rules, EXCEPT Rule 6, only have one capturing group, meaning group_match = 2 for rule 6 and group_match = 1 for all others 
      group_match = len(match.groups())
      match = match.group(group_match)
      
      # STEP 3: Switch personal pronouns from first person to second person perspective and vice versa
      match = switch_perspective(match)

      # A random reply option is selected from the corresponding value to the key/rule in the dictionary
      output = random.choice(response)

      # STEP 4: Replace the 'TEMP' placeholder with the reflected match key
      output = output.replace('TEMP', match)

      return output

# The 'start_eliza' function introduces ELIZA and then asks for users name to start conversation with ELIZA
def start_eliza():

  # ELIZA introduces itself and asks for user's name
  print('[ELIZA] Hi there! I am your psychotherapist, ELIZA. What is your name?')

  # Print header to label user input
  print("[User] ", end = ' ')

  # Clean user input and derive name
  name_input = clean(input())
  name = get_name(name_input)

  # ELIZA says "Hi" to user and asks first introspective question
  print('[ELIZA] Hi, ' + str(name) + '. What would you like to talk about today?')

  # Create name header to label all user input lines
  name_header = '[' + str(name) + '] '

  return name_header

if __name__ == '__main__':

  # STEP 1: Introduce ELIZA and ask for user's name
  name_header = start_eliza()

  # As long as there is user input, the program takes and cleans user input and prints a response to console
  loop_go = True
  while (True and loop_go):
    # Print name header to label all user input lines
    print(name_header, end = ' ')

    # STEP 2: The program cleans the next user input 
    userInput = clean(input())
    
    # If you user types quit, the program will end
    if (re.search(r'^\bquit\b$',userInput)):
      loop_go = False 
      print('Thanks for talking to me today. I get lonely sometimes, too. Have a good one!')
    else: 
      # Generates a ELIZA reply using the 'reply' function
      print('[ELIZA] ', reply(userInput))


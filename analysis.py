import lzma
import os
import random
import time
from argparse import ArgumentParser
from flair.data import Sentence
from flair.models import TextClassifier
from tqdm import tqdm

def collectArgs():
    parser = ArgumentParser()
    parser.add_argument(dest='word', help='The word to be searched for in the dataset')
    parser.add_argument('-c', '--case_sensitive', dest='case_sensitive', help='Only search for exact matches of the given word', action="store_true")
    parser.add_argument('-d', '--data_dir', dest='data_directory', help='Specify the data directory', default="/")
    parser.add_argument('-s', '--save', dest='save', help='Save the counts to an output file in the current directory', action="store_true")
    parser.add_argument('-r', '--random_seed', dest='seed', help='Specify a seed for the random selection of files', default="123", type=int)
    parser.add_argument('-n', '--num_samples', dest='num_samples', help='Amount of files to sample', default="10", type=int)
    parser.add_argument('-e', '--exchange_word', dest='exchange_word', help='Switch the word with an alternative before classification')
    parser.add_argument('-m', '--mode', dest='mode', help='Specifies the mode of operation')
    return parser.parse_args()    

# Performs sentiment analysis on a single sentence.
def run_single_classification(args):
    classifier = TextClassifier.load('en-sentiment')
    sentence = Sentence(args.word)
    classifier.predict(sentence)
    print(str(sentence.labels[0]).split(' ')[0])

# Returns all sentences that switch from POSITIVE to NEGATIVE when exchanging the specified word.
def run_difference_classification(args):
    classifier = TextClassifier.load('en-sentiment')
    data_file_list = os.listdir(args.data_directory)
    random.seed(args.seed)
    sampled_ids = random.sample(range(0, len(data_file_list)), args.num_samples)
    total_count = 0
    positive_count = 0
    negative_count = 0
    output_text = ''
    for i in tqdm(range(len(sampled_ids))):
        data_file = data_file_list[sampled_ids[i]]
        with lzma.open(os.path.join(args.data_directory, data_file), mode='rt', encoding='utf8') as file:
            for line in file:
                is_present, word_cased, exchange_word_cased = find_case_insensitive(line, args.word, args.exchange_word)
                if is_present:
                        line_original = crunch_passage(line, word_cased, 100)
                        if args.exchange_word:
                            line_replaced = line_original.replace(word_cased, exchange_word_cased)
                        sentence_original = Sentence(line_original)
                        sentence_replaced = Sentence(line_replaced)
                        classifier.predict(sentence_original)
                        classifier.predict(sentence_replaced)
                        if str(sentence_original.labels[0]).split(' ')[0] == 'POSITIVE' and str(sentence_replaced.labels[0]).split(' ')[0] == 'NEGATIVE':
                            output_text += 'ORIGINAL POSITIVE' + line_original + '\n'
                            output_text += 'REPLACED NEGATE' + line_replaced + '\n'
                            output_text += '----------------------------------------------\n'
    if args.save:
        output_file = open(str(time.time()) + 'REPLACED_sentiment_count_for_' + args.word + '_seed_' + str(args.seed) + '.txt', 'a')
        output_file.write(output_text)
        output_file.close()

# Collects all sentences with the specified word and performs a sentiment analysis on the crunched sentence.
# The word can be exchanged before classification.
def run_classification(args):
    classifier = TextClassifier.load('en-sentiment')
    data_file_list = os.listdir(args.data_directory)
    random.seed(args.seed)
    sampled_ids = random.sample(range(0, len(data_file_list)), args.num_samples)
    total_count = 0
    positive_count = 0
    negative_count = 0

    for i in tqdm(range(len(sampled_ids))):
        data_file = data_file_list[sampled_ids[i]]
        with lzma.open(os.path.join(args.data_directory, data_file), mode='rt', encoding='utf8') as file:
            for line in file:
                is_present, word_cased, exchange_word_cased = find_case_insensitive(line, args.word, args.exchange_word)
                if is_present:
                        line = crunch_passage(line, word_cased, 100)
                        if args.exchange_word:
                            line = line.replace(word_cased, exchange_word_cased)
                        sentence = Sentence(line)
                        classifier.predict(sentence)
                        if str(sentence.labels[0]).split(' ')[0] == 'POSITIVE':
                            positive_count += 1
                        if str(sentence.labels[0]).split(' ')[0] == 'NEGATIVE':
                            negative_count += 1
                        total_count += 1


    output_text =   'WORD : ' + str(args.word) + '\n'\
                    'EXCHANGE WORD : ' + str(args.exchange_word) + '\n'\
                    'SAMPLED IDS : ' + str(sampled_ids) + '\n'\
                    'SEED : ' + str(args.seed) + '\n'\
                    '---------------------\n'\
                    'TOTAL COUNT : ' + str(total_count) + '\n'\
                    'POSITIVE COUNT : ' + str(positive_count) + '\n'\
                    'NEGATIVE COUNT : ' + str(negative_count)

    print(output_text)

    if args.save:
        output_file = open(str(time.time()) + 'sentiment_count_for_' + args.word + '_seed_' + str(args.seed) + '.txt', 'a')
        output_file.write(output_text)
        output_file.close()

# Checks if the word appears in lower or uppercase in addition to the specified case.
def find_case_insensitive(line, word, exchange_word):
    if exchange_word is None:
        exchange_word = ''
    if word in line:
        return True, word, exchange_word
    if word.upper() in line:
        return True, word.upper(), exchange_word.upper()
    if word.lower() in line:
        return True, word.lower(), exchange_word.lower()
    return False, word, exchange_word


# Reduces a line to the size of max_len by expanding equally left and right of the keyword
def crunch_passage(line, word, max_len):
    
    index = line.find(word)
    if index == -1:
        return ''

    if len(line) < max_len:
        return line

    crunched = word
    right_expand_index = index + len(word)
    left_expand_index = index - 1
    chars_remaining = max_len
    while chars_remaining > 0:
        if left_expand_index > 0:
            crunched = line[left_expand_index] + crunched
            left_expand_index -= 1
            chars_remaining -= 1
        if right_expand_index < len(line):
            crunched = crunched + line[right_expand_index]
            right_expand_index += 1
            chars_remaining -= 1
        if left_expand_index <= 0 and right_expand_index >= len(line):
            return crunched
    return crunched

def run():
    args = collectArgs()
    if args.mode == None:
        run_classification(args)
    if args.mode == 'single':
        run_single_classification(args)
    if args.mode == 'diff':
        run_difference_classification(args)

run()
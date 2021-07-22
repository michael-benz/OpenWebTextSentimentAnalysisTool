# OpenWebTextSentimentAnalysisTool
A tool for performing sentiment analysis on the OpenWebText corpus available at https://skylion007.github.io/OpenWebTextCorpus/. Checks the corpus for passages with the specified word, crunches to passage and returns POSITIVE or NEGATIVE for that part of the passage. You can also specify an exchange word, to discover if changing that word alone changes the sentiment.

## Requirements
* Python version: 3.6
* Flair for Sentiment analysis: pip install flair

## Usage
* Download the OpenWebText corpus and unzip. You are left with zipped subsets named urlsf_subset<int><int>-<int>_data.xz. This is the directory you want to specify.
* Run python analysis.py <YOUR WORD> -d <DIRECTORY>
* use -s or --save to save to output to a file in the current directory
* -r specifies a random seed and -n the number of dataset subset samples
* -e specifies the exchange word
* -m single classifies only the input word or sentence and doesnt use the corpus
* -m diff gets the sentences where the exchange of the word would change the sentiment from POSITIVE to NEGATIVE

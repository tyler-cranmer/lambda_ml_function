# import re
# import nltk
# import os

# from bs4 import BeautifulSoup

# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# nltk_data_path = '/tmp/nltk_data'

# nltk.download('stopwords', download_dir=nltk_data_path)
# nltk.download('punkt', download_dir=nltk_data_path)
# nltk.download('wordnet', download_dir=nltk_data_path)

# nltk.data.path.append(nltk_data_path)

# def clean_text(text: str) -> str:
#     REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
#     STOP_WORDS = set(stopwords.words('english'))
#     BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
#     URL_PATTERN = re.compile(r'https?://\S+')
#     word_net_lem = WordNetLemmatizer()

#     text = re.sub(r'task name:\s*', '', text)  # remove task name:
#     text = re.sub(r'task:\s*', '', text)  # remove task:
#     text = URL_PATTERN.sub('url', text)  # remove url
#     text = BeautifulSoup(text, 'html.parser')
#     text = text.get_text()  # removes html text
#     text = REPLACE_BY_SPACE_RE.sub(' ', text)  #removes special characters
#     text = BAD_SYMBOLS_RE.sub('', text)  # removes any weird symbols.
#     text = re.sub(' +', ' ', text)  # removes empty spaces
#     text = text.lstrip()
#     text = text.rstrip()
#     # deletes stopwords from text
#     text = ' '.join([word for word in text.split()
#                     if word not in (STOP_WORDS)])
#     text = nltk.word_tokenize(text)  # tokenizes the text
#     # lemmatization of words and combines them to text
#     text = ' '.join([word_net_lem.lemmatize(words) for words in text])
#     return text

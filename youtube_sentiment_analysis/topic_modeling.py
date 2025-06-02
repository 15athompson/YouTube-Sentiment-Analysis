import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 3]

def perform_lda_topic_modeling(comments, num_topics=5, num_words=10):
    preprocessed_comments = [preprocess_text(comment) for comment in comments]
    dictionary = corpora.Dictionary(preprocessed_comments)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_comments]

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    topics = []
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        topics.append({
            'topic_id': idx,
            'words': [word.split('"')[1] for word in topic.split('+')]
        })

    return topics
import matplotlib.pyplot as plt
from nltk.lm.models import KneserNeyInterpolated
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.lm import MLE
from nltk.corpus import stopwords as STOPWORDS_NLTK
import gensim
from gensim.models import Word2Vec
import  numpy as np
import requests
from PIL import Image
import pickle #turşu
import re
from scipy import special
from operator import itemgetter
from math import log
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten, padded_everygram_pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
import math
import multiprocessing
from time import time



def create_WordCloud (docs, dimension_size, wordcloud_outputfile, mode = "TF", stopwords = True ):

     # use all documents in the given turşu file 
    """
    The function will create the word cloud and save it in the provided output file in png format

    Docs = List of documents
    dimension_size = dimension size
    wordcloud_outputfile = full path of the output file
    mode = The term weighting option (either TF or TFIDF weighting, default is TF)
    stopwords = The stopwords option (either to remove the stopwords or not, default is to
    keep them)
    """


    
    words = [word_tokenize(i) for i in docs] #words = 
    #print("type of words is: ", type(words)) #list of (list of string)
    #print(words)
    t = time()
    words_modified = []
    if stopwords == True:
        stopWords = set(STOPWORDS_NLTK.words('turkish'))
        stopWords.update(['ve', 'işte', 'bu', 'bir', 'bile' , 'var' , 'dedi' ,'onu', 'o','e','olan' ,'olarak', 'ancak' ,'kadar' ,'ın','n','da','artık','nın' ,'göre','ile','ise','için','değil', 'vs', 'acaba', 'ama','ayrıca','aslında','az','bazı','belki', 'biri' ,'birkaç', 'birşey', 'şey', 'biz', 'çok' ,'çünkü', 'daha' , 'de', 'defa', 'diye' , 'eğer' , 'en','gibi' , 'hem' , 'hep' , 'hepsi' , 'her' , 'hiç' , 'kez', 'ki', 'mı' , 'mi', 'mu' , 'mü' , 'nasıl' , 'ne' , 'neden' , 'nerde', 'nerede','nereye','niçin','niye','o','sanki','siz' , 'şu' , 'tüm' , 'veya' , 'ya' , 'yani'])
        #print( "type of stopWords is : " , type(stopWords))
        #print(stopWords)
        for w in words:
            #print("type of w: " , type(w)) #list
            for word in w:
                #print("type of word: " , type(word)) #string
                if word.lower() not in list(stopWords):
                    #print(word.lower())
                    words_modified.append(word.lower())
            
        #print("Stop words has been removed")
    else:
        for w in words:
            for word in w:
                words_modified.append(word.lower())

    comment_words = ' '

    for word in words_modified:
        comment_words += word + " "
    #print("comment_words is: " , comment_words)
    
    #pic = np.array (Image.open(requests.get('https://i.imgyukle.com/2020/12/02/YPTCoq.png',stream=True).raw))
    #in order to have a shaped wordcloud you can write mask = pic below. TA said no.
    wordcloud = WordCloud (width = 800 , height = 800, background_color = 'white', min_font_size= 10).generate(comment_words)
    # mask helps us to change the shape of the cloud

    #now plotting the image
    plt.figure(figsize = (10,10), facecolor = 'white' , edgecolor = 'blue')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    #plt.show()
    
    print('Time to create the wordcloud: {} mins'.format(round((time() - t) / 60, 3)))
    # saving the png
    wordcloud.to_file(wordcloud_outputfile)
    
    
def create_ZiphsPlot(docs, zips_outputfile):
    """
    The function will create the Zipf’s plot and save it in the provided output file in png format.
    #docs = list of documents
    zips_outputfile = full path of output file
    """
    words_modified = []
    comment_words = ' '
    words_tokenized = [word_tokenize(i) for i in docs]
    t = time()
    #words_tokenized = [word_tokenize(docs[5])]
    for w in words_tokenized:
            for word in w:
                comment_words += word.lower() + ' '
    frequency = {}
    words = re.findall(r'(\b[A-ZÇĞİÖŞÜa-zçğıöşü][a-zçğıöşü]{2,}\b)', comment_words)   
    for word in words:
        count = frequency.get(word,0)
        frequency[word] = count + 1 
   
    frequency = {key:value for key,value in frequency.items()}
    #print("frequency dict: " , frequency)

    #convert freq into numpy array
    s = frequency.values()
    #print("type of s:" , type(s)) #dict_values
    
    sorted_frequency = sorted(s, reverse = True)
    #print(sorted_frequency[0:50])
    plt.loglog(sorted_frequency)
    plt.xlabel('log(rank)')
    plt.ylabel('log(freq)')
    plt.savefig(zips_outputfile)
    plt.close()
    print('Time to plot the Ziph\'s Plot: {} mins'.format(round((time() - t) / 60, 3)))
    #plt.show()


def create_HeapsPlot(docs, heaps_outputfile):
    """
    The function will create the Heap’s plot and save it in the provided output file in png format.

    #docs = list of documents
    # heaps_outputfile = full path of output file
    """

    words_modified = []
    comment_words = ' '
    words_tokenized = [word_tokenize(i) for i in docs]
    #words_tokenized = [word_tokenize(docs[5])]
    vocab_size = 0
    unique_word_count = 0
    x_val = []
    y_val = []
    unique_words = []
    t = time()
    for w in words_tokenized:
            for word in w:
                if word not in unique_words:
                    unique_word_count += 1
                    unique_words.append(word)
                vocab_size += 1
                y_val.append(unique_word_count)
                x_val.append(vocab_size)
                
    plt.plot(x_val,y_val)
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Unique Word Count')
    plt.savefig(heaps_outputfile)
    plt.close()
    print('Time to plot the Heaps\' Plot: {} mins'.format(round((time() - t) / 60, 3)))
    #plt.show()


def create_LanguageModel(docs, model_type, ngram):
    """
    docs = list of documents
    model_type: Type of the model (MLE or KneserNeyInterpolated)
    ngram = maximum ngram size
    """  
    #DOCS = LİST OF STR
    #docs = docs[0:500]
    s =''
    for item in docs:
        s += item
    #print("type of s is: ", type(s))
    s = s.lower()
    t = time()
    tokenized_text = [word_tokenize(i.lower()) for i in docs] #do we need this ?
    #tokenized_text = [word_tokenize(docs[12])]
    #list_of_str_docs = [' '.join(elem) for elem in docs] # docs is now list of string 
    #string_docs = [' '.join(elem) for elem in list_of_str_docs] #our docs is now string
    #print(tokenized_text)
    # HER KELİMEYI LOWERCASE YAP ##################################################
    # input format - fonksıyona gıren 

    #paddedLine = list(pad_both_ends(docs, n=ngram))
    
    train_data , padded_sents = padded_everygram_pipeline(ngram , tokenized_text)

    if model_type == 'MLE':
        
        model = MLE(ngram)
        model.fit (train_data, padded_sents)
       
        print('Time to train the MLE language model: {} mins'.format(round((time() - t) / 60, 3)))
        """print(model.vocab)
        print('length of model: ' , model.vocab)
        print('model count is: ', model.counts)"""
        #print('model: ', model)   

    else: #model_type == 'KneserNeyInterpolated'
        model = KneserNeyInterpolated(ngram)
        model.fit(train_data , padded_sents)
       
        print('Time to train the Kneser Ney Interpolated langauge model: {} mins'.format(round((time() - t) / 60, 3)))
        
    #print(model)
    return model


def generate_sentence(LM3_KneserNeyInterpolated,text):
    """
    This function will take your trained language model and starting text for the sentence.
    Then it will generate a sentence by predicting the next word until the end of the
    sentence token (</s>) is seen. In this function, you will generate 5 sentences and then
    you will return the sentence which has the lowest perplexity score. You will return the
    sentence and its perplexity score.

    LM3_KneserNeyInterpolated = Trained language model
    text = text for starting the sentence

    """
    detokenize = TreebankWordDetokenizer().detokenize
    content1 = []
    content2 = []
    content3 = []
    content4 = []
    content5 = []
    token = text
    #num_words =1#################
    while True:
        token = LM3_KneserNeyInterpolated.generate(num_words = 1, text_seed  = token)
        if token == '<s>':
            continue
        if token == '</s>':
            break
        if len(content1) > 200:
            print("Word count for sentence 1 exceeded 200, terminating the sentence automatically.")
            break
        #print("token is: " +token)
        content1.append(token)
    token = text
    while True:
        token = LM3_KneserNeyInterpolated.generate(num_words = 1, text_seed  = token)
        if token == '<s>':
            continue
        if token == '</s>':
            break
        if len(content2) > 200:
            print("Word count for sentence 2 exceeded 200, terminating the sentence automatically.")
            break
        content2.append(token)
    token = text
    while True:
        token = LM3_KneserNeyInterpolated.generate(num_words = 1, text_seed  = token)
        if token == '<s>':
            continue
        if token == '</s>':
            break
        if len(content3) > 200:
            print("Word count for sentence 3 exceeded 200, terminating the sentence automatically.")
            break
        content3.append(token)
    token = text
    while True:
        token = LM3_KneserNeyInterpolated.generate(num_words = 1, text_seed  = token)
        if token == '<s>':
            continue
        if token == '</s>':
            break
        if len(content4) > 200:
            print("Word count for sentence 4 exceeded 200, terminating the sentence automatically.")
            break
        content4.append(token)
    token = text
    while True:
        token = LM3_KneserNeyInterpolated.generate(num_words = 1, text_seed  = token) 
        if token == '<s>':
            continue
        if token == '</s>':
            break
        if len(content5) > 200:
            print("Word count for sentence 5 exceeded 200, terminating the sentence automatically.")
            break
        content5.append(token)
    
    sentence1 = detokenize(content1)
    sentence2 = detokenize(content2)
    sentence3 = detokenize(content3)
    sentence4 = detokenize(content4)
    sentence5 = detokenize(content5)
    ngram_s1 = everygrams(sentence1.split(), max_len =LM3_KneserNeyInterpolated.order)
    ngram_s2 = everygrams(sentence2.split(), max_len =LM3_KneserNeyInterpolated.order)
    ngram_s3 = everygrams(sentence3.split(), max_len =LM3_KneserNeyInterpolated.order)
    ngram_s4 = everygrams(sentence4.split(), max_len =LM3_KneserNeyInterpolated.order)
    ngram_s5 = everygrams(sentence5.split(), max_len =LM3_KneserNeyInterpolated.order)
    #print(list(ngrams(content1, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')))
    #print("sentence1 is:" ,sentence1)
    perp1 = LM3_KneserNeyInterpolated.perplexity(ngram_s1) # sentence ları ngram olarak ver. everygram(ngram), veya ngram()
    perp2 = LM3_KneserNeyInterpolated.perplexity(ngram_s2)
    perp3 = LM3_KneserNeyInterpolated.perplexity(ngram_s3)
    perp4 = LM3_KneserNeyInterpolated.perplexity(ngram_s4)
    perp5 = LM3_KneserNeyInterpolated.perplexity(ngram_s5)
    #print("perplexities 1,2,3,4,5: ",perp1 , "\t", perp2 , "\t", perp3 , "\t" , perp4, "\t" , perp5 , "\n")
    """
    print("sentence1: " , sentence1)
    print("sentence2: " , sentence2)
    print("sentence3: " , sentence3)
    print("sentence4: " , sentence4)
    print("sentence5: " , sentence5)
    """

    min_perp = min(perp1, perp2, perp3, perp4, perp5)
    min_sent = ''
    if min_perp == perp1:
        min_sent = sentence1
    elif min_perp == perp2:
        min_sent = sentence2
    elif min_perp == perp3:
        min_sent = sentence3
    elif min_perp == perp4:
        min_sent = sentence4
    elif min_perp == perp5:
        min_sent = sentence5
    return min_sent,min_perp
    


def create_WordVectors(docs, dimension_size, model_type, window_size):
    """

    docs = List of documents
    dimension_size = The dimension size of the vectors
    model_type = Type of the model (Skipgram or CBOW)
    window_size = Window size

    The function will return the trained word embeddings.
    """
    
    tokenized_text = [word_tokenize(i.lower()) for i in docs]
    t = time()
    if model_type == 'Skipgram' or model_type == 'skipgram':
        model = Word2Vec( size = dimension_size, window = window_size, workers = multiprocessing.cpu_count()-1, sg= 1, min_count = 5)
    elif model_type == 'cbow':
        model = Word2Vec( size = dimension_size, window = window_size, workers = multiprocessing.cpu_count()-1, sg = 0,  min_count = 5) # workers=4 for parallel quad core work
    else:# if model_type is entered in wrong format it is automatically set to cbow
        model = Word2Vec( size = dimension_size, window = window_size, workers = multiprocessing.cpu_count()-1, sg = 0,  min_count = 5)
    #print(model)
    model.build_vocab(tokenized_text, progress_per=10000)
    
    model.train(tokenized_text, total_examples=model.corpus_count, epochs=10, report_delay=1)
    model.init_sims(replace=True)
    print('Time to create word vectors: {} mins'.format(round((time() - t) / 60, 3)))
    return model
    #https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

def use_WordRelationship(WE,example_tuple_list,example_tuple_test):
    """
        This function will take your trained word embeddings model, an example list of tuples
    of words which have the same type of relationship and example tuple with the missing
    value. In this function, you will use the example tuples to find average distance
    between pair of words, and then use this distance to predict the missing value in the
    example tuple.
    Example tuples may include words that are not in your vocabulary. In such a case, you
    have to exclude these tuples from your list of tuples before starting your calculations.

    If there are no examples left in the list of tuples or the test tuple word is not in the
    vocabulary, you have to print the following message 
    “Sorry, this operation cannot be performed!”.

    In the output you will print the top 5 candidates for the missing value as tuple together
    with its similarity score. In some cases, the test word itself could be in the top-5
    candidates list, please make sure you ignore the test word while printing the top-5
    candidates.


    WE = Trained distributed word representation model (word_embeddings)
    example_tuple_list = Example Tuple List
    example_tuple_test = Example Tuple with missing value
    """
    sum_dist = 0
    count  = 0
    #wv 
    #kelimelerin vektorlerını bulup fark vektorunu bulucam.
    #tum ıkılıler ıcın fark verktorunu bul sonra ubnun ortalmaasını bul == bu da bana bir vektor verıyor.
    #o ortalama vektoru de test kelımesıne eklıyorum veya cıkartıcam x veya y verildiyse
    #similar by vector fonskıyonui ile top5 yazdıyıorum.
    size = WE.vector_size
    #print("size is:" , size)
    difference_vector = np.empty(size, dtype=float)
    sum = np.empty(size, dtype=object)
    count = 0
    xminusy_vector = np.empty(size, dtype=float)
    for pair in example_tuple_list:
        try:
            xminusy_vector += WE.wv[pair[0]] - WE.wv[pair[1]] # x - y 
            count += 1
        except KeyError: #item is not in vocabulary
            #print("KeyError at example_tuple_list, at item:", pair)
            continue
    if count == 0:
        print("Sorry, this operation cannot be performed!1")

    average_vector = [x / count for x in xminusy_vector] #xminusy_vector / count --> BUNU DENE !!
    count2 = 0
    item = example_tuple_test
    try:
        if item[0] == '': #if x is missing 
            difference_vector = WE.wv[item[1]] + average_vector
        elif item[1] == '': #if y is missing
            difference_vector = WE.wv[item[0]] - average_vector
        count += 1
    except KeyError:
        print("Sorry, this operation cannot be performed! KeyError at example_tuple_test, at item:" , item)
    if count2 == 0:
        print("Sorry, this operation cannot be performed!2")

    similars = WE.similar_by_vector(difference_vector, topn = 5)
    for i,tuple in enumerate(similars):
        if item[0] == '':
            if item[1] == tuple[0]:
                continue
            else:
                perc = round(tuple[1]*100,3)
                num = i+1
                print(str(num) + ". most similar result is: " + tuple[0] + " with similarity : " + str(perc) +"%.")
        elif item[1] == '':
            if item[0] == tuple[0]:
                continue
            else:
                perc = round(tuple[1]*100,3)
                num = i+1
                print(str(num) + ". most similar result is: " + tuple[0] + " with similarity : " + str(perc) +"%.")
    t = time()
    print('Time to use word relationships to find missing value in the tuple: {} mins'.format(round((time() - t) / 60, 3)))
#References
#https://radimrehurek.com/gensim/models/word2vec.html
#https://www.nltk.org/api/nltk.lm.html
#https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
#https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
#https://www.kaggle.com/alvations/n-gram-language-model-with-nltk/notebook
#https://rstudio-pubs-static.s3.amazonaws.com/215309_736f5cc00eea4bb9be5a8c566da2beb6.html
#https://www.datacamp.com/community/tutorials/wordcloud-python
#https://www.geeksforgeeks.org/generating-word-cloud-python/
#https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
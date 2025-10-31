import heapq
import math
import random

import pandas as pd
from pandas import DataFrame

NUMBER_OF_DOCS = 7000
TOP_K = 5
R_CHAMPIONS = 5
RESPOND_BY_CHAMP_LIST = False
USE_HEAP = True
NUMBER_OF_CENTROIDS = 5


def main():
    load_singulars(singulars)

    # Data input:
    data = pd.read_excel('IR_Spring2021_ph12_7k.xlsx')
    test_data = DataFrame(data).head(NUMBER_OF_DOCS)
    contents = test_data.get('content')
    urls = test_data.get('url')

    # Tokenization & some normalizations to produce inverted index:
    for i in test_data.get('id'):
        tokenize(i, (contents[i - 1]).split('\n'))

    # # calculate tf-idf vector for each document:
    # for i in test_data.get('id'):
    #     cal_tfidf(i, (contents[i - 1]).split('\n'))

    remove_most_counted_words(word_count, posting_lists, posting_lists_with_tf, 5)
    build_champion_list()

    cal_idfs(posting_lists, idf_list)
    # print("pl w tf:", posting_lists_with_tf)

    # test_postings(posting_lists)
    cal_doc_vectors()
    generate_centroids(doc_vectors)

    # Getting Query:
    while True:
        query = input('\nEnter your QUERY please: (enter \"end\" to finish)\n > ')
        if query == 'end':
            break
        # respond_to_query(query, posting_lists)
        respond_by_cos_score(query, posting_lists, urls)


def generate_centroids(nodes):
    # index = 0
    for i in range(0, NUMBER_OF_CENTROIDS):
        # random_centroid = np.random.rand()

        random_centroid = random.choice(nodes)
        centroids.append(random_centroid)
    # centroids.sort()
    print("centroids: ", centroids)


def cal_doc_vectors():
    for term in posting_lists_with_tf.keys():
        for doc in posting_lists_with_tf[term]:

            if doc not in doc_vectors.keys():
                doc_vectors[doc] = {}
            dft = posting_lists[term][0]
            idf = math.log(NUMBER_OF_DOCS / dft, 10)
            # print(cal_wtd(posting_lists_with_tf[term][doc], idf))
            doc_vectors[doc][term] = cal_wtd(posting_lists_with_tf[term][doc], idf)


def cal_idfs(postings, idfs):
    for term in postings.keys():
        dft = postings[term][0]
        idfs[term] = math.log(NUMBER_OF_DOCS / dft, 10)


def tokenize(doc_id, text):
    # removing last 3 lines of text which is not needed
    text = text[:len(text) - 3]
    doc_tf = {}
    docs_len_list[doc_id] = 0

    for line in text:
        # Normalization
        line = remove_bad_chars(line)
        line = remove_numbers(line)
        line = homogenize(line)

        # tokenization
        for word in line.split(" "):
            word = word.strip()

            # # word_count update before normalization to prevent loss of data:
            # if word not in word_count.keys():
            #     word_count[word] = 1
            # else:
            #     word_count[word] += 1

            # word = remove_suffix(word)
            # word = remove_prefix(word)
            word = singularize(word)
            docs_len_list[doc_id] += 1

            # word_count update:
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word] += 1

            if word not in doc_tf.keys():
                doc_tf[word] = 1
            else:
                doc_tf[word] += 1

            # tokens, posting_lists update:
            if not (word, doc_id) in tokens:
                # tokens.append((word, doc_id))
                if word not in posting_lists.keys():
                    posting_lists[word] = (1, [doc_id])
                    # posting_lists_with_tf[word] = ([(doc_id, 1)])

                else:
                    if doc_id not in (posting_lists[word][1]):
                        (posting_lists[word][1]).append(doc_id)
                        (posting_lists[word]) = ((posting_lists[word][0]) + 1, (posting_lists[word][1]))

                    #     (posting_lists_with_tf[word]).append((doc_id, 1))
                    # else:
                    #     word_count.
                    #     (posting_lists_with_tf[word]).keys()

                    # if doc_id not in (posting_lists_with_tf[word][1][0]):
                    #     (posting_lists_with_tf[word][1]).append((doc_id, 1))
                    #     posting_lists_with_tf[word] = (1, [(doc_id, 1)])

    for term in doc_tf.keys():
        if term not in posting_lists_with_tf.keys():
            posting_lists_with_tf[term] = {}
        #     posting_lists_with_tf[term] = ([(doc_id, doc_tf[term])])
        # else:
        #     posting_lists_with_tf[term].append(doc_id, doc_tf[term])
        posting_lists_with_tf[term][doc_id] = doc_tf[term]

    # print("doc_tf:", doc_tf)
    # print("pl w tf:", posting_lists_with_tf)


def build_champion_list():
    for term in posting_lists_with_tf.keys():

        temp_champ_list = []
        for doc in posting_lists_with_tf[term].keys():

            if len(temp_champ_list) < R_CHAMPIONS:
                temp_champ_list.append(posting_lists_with_tf[term][doc])
            elif posting_lists_with_tf[term][doc] > min(temp_champ_list):
                temp_champ_list.remove(min(temp_champ_list))
                temp_champ_list.append(posting_lists_with_tf[term][doc])
        temp_champ_list.sort()

        champion_lists[term] = []
        for tf in temp_champ_list:
            for doc in list(posting_lists_with_tf[term].keys()):
                if posting_lists_with_tf[term][doc] == tf and (doc not in champion_lists[term]):
                    champion_lists[term].append(doc)
        champion_lists[term].reverse()


def cal_wtq(term, query, dft):
    tf = query.count(term)
    idf = math.log(NUMBER_OF_DOCS / dft, 10)
    wtq = math.log(1 + tf, 2) * idf
    return wtq


def cal_wtd(tf, idf):
    wtd = math.log(1 + tf, 2) * idf
    return wtd


def respond_by_cos_score(query, postings, urls):
    query = remove_bad_chars(query)
    query = remove_numbers(query)
    query_words = query.split(' ')
    # query_words = remove_suffix(query_words)
    # query_words = remove_prefix(query_words)
    # print(query_words)

    updated_query_words = []
    for word in query_words:
        new_word = singularize(word)
        # new_word = remove_suffix(new_word)
        # new_word = remove_prefix(new_word)
        updated_query_words.append(new_word)

        # if word in query_word_counts.:
        #     query_word_counts[query_word_counts.index(word)] += 1
        # else:
        #     query_word_counts.append((word, 1))

    query_words = updated_query_words

    doc_scores = {}

    for term in query_words:
        if term in postings.keys():
            term_wtq = cal_wtq(term, query_words, postings[term][0])

            if RESPOND_BY_CHAMP_LIST:
                doc_collection = champion_lists[term]
            else:
                doc_collection = posting_lists_with_tf[term].keys()

            for doc in doc_collection:
                term_wtd = cal_wtd(posting_lists_with_tf[term][doc], idf_list[term])
                if doc in doc_scores.keys():
                    doc_scores[doc] += term_wtd * term_wtq
                else:
                    doc_scores[doc] = term_wtd * term_wtq
    for doc in doc_scores.keys():
        doc_scores[doc] = doc_scores[doc] / docs_len_list[doc]

        if USE_HEAP:
            doc_scores[doc] = doc_scores[doc] * -1
    if not len(doc_scores) == 0:
        print("Top Docs by cos similarity scores respectively:\n", return_top_k_docs(doc_scores, urls))
    else:
        print("no matches found. please reinform your query!")


def return_top_k_docs(doc_scores, urls):
    vals = list(doc_scores.values())

    if USE_HEAP:
        heap_list = vals
        heapq.heapify(heap_list)

        top_k_docs = []
        for i in range(0, min(TOP_K, len(doc_scores))):
            score = heapq.heappop(heap_list)

            for doc in doc_scores.keys():
                if doc_scores[doc] == score and ((doc, urls[doc - 1]) not in top_k_docs):
                    top_k_docs.append((doc, urls[doc - 1]))
    else:
        vals.sort()

        top_k_docs = []
        for i in range(0, min(TOP_K, len(doc_scores))):
            score = vals.pop()

            for doc in doc_scores.keys():
                if doc_scores[doc] == score and ((doc, urls[doc - 1]) not in top_k_docs):
                    top_k_docs.append((doc, urls[doc - 1]))

    return top_k_docs


def homogenize(line):
    # change Arabic to Persian
    line = line.replace('ا', 'ا')
    # line = line.replace('آ', 'ا')
    line = line.replace('ب', 'ب')
    line = line.replace('ت', 'ت')
    line = line.replace('ث', 'ث')
    line = line.replace('ج', 'ج')
    line = line.replace('ح', 'ح')
    line = line.replace('خ', 'خ')
    line = line.replace('د', 'د')
    line = line.replace('ذ', 'ذ')
    line = line.replace('ر', 'ر')
    line = line.replace('ز', 'ز')
    line = line.replace('س', 'س')
    line = line.replace('ش', 'ش')
    line = line.replace('ص', 'ص')
    line = line.replace('ض', 'ض')
    line = line.replace('ط', 'ط')
    line = line.replace('ظ', 'ظ')
    line = line.replace('ع', 'ع')
    line = line.replace('غ', 'غ')
    line = line.replace('ف', 'ف')
    line = line.replace('ق', 'ق')
    line = line.replace('ك', 'ک')
    line = line.replace('ل', 'ل')
    line = line.replace('م', 'م')
    line = line.replace('ن', 'ن')
    line = line.replace('و', 'و')
    line = line.replace('ه', 'ه')
    line = line.replace('ي', 'ی')
    line = line.replace('ى', 'ی')

    return line


def remove_suffix(word):
    if word[len(word) - 2:len(word)] == 'ها' or word[len(word) - 2:len(word)] == 'ات' or word[len(word) - 2:len(
            word)] == 'تر':
        word = word[:len(word) - 2]

    elif word[len(word) - 4:len(word)] == 'ترین':
        word = word[:len(word) - 4]

    return word


def remove_prefix(word):
    if word[0:3] == 'فرا' or word[0:3] == 'فرو' or word[0:3] == 'پسا':
        word = word[4:len(word)]

    elif word[0:2] == 'هم':
        word = word[3:len(word)]

    return word


def singularize(word):
    if word in singulars.keys():
        word = singulars[word]
    return word


def remove_bad_chars(line):
    bad_chars = [';', ':', '!', "*", '.', ',', '،', '\n', '\u200c', '?', '(', ')', '؛', '٪', '%']
    line = ''.join((filter(lambda i: i not in bad_chars, line)))
    line = line.replace('-', ' ')
    line = line.replace('_', ' ')
    return line


def remove_numbers(line):
    numbers = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
    line = ''.join((filter(lambda i: i not in numbers, line)))
    return line


def remove_most_counted_words(words, postings, postings_w_tf, k):
    # removing top k stopwords
    for i in range(k):
        # list out keys and values separately
        key_list = list(words.keys())
        val_list = list(words.values())

        position = val_list.index(max(val_list))
        max_word = key_list[position]
        words.pop(max_word)
        postings.pop(max_word)
        postings_w_tf.pop(max_word)


def test_postings(postings):
    while True:
        word = input("\n\nEnter a word to get its postings-list: (enter \"end\" to finish testing)\n > ")
        if word == 'end':
            break
        elif word in postings.keys():
            print(postings[word])
        else:
            print('Not in dictionary. Try again!')


def respond_to_query(query, postings):
    query = remove_bad_chars(query)
    query = remove_numbers(query)
    query_words = query.split(' ')
    # query_words = remove_suffix(query_words)
    # query_words = remove_prefix(query_words)

    # print(query_words)

    if len(query_words) == 1:
        word = query_words[0]

        word = singularize(word)

        if word in postings.keys():
            print(' \"one word query\". DOCs:\n >', postings[word])
        else:
            print(' Not in dictionary. Try again!')

    else:
        print(' \"multiple words query\". DOCs:')

        updated_query_words = []
        for word in query_words:
            new_word = singularize(word)
            # new_word = remove_suffix(new_word)
            # new_word = remove_prefix(new_word)
            updated_query_words.append(new_word)
        query_words = updated_query_words

        # file_list = set()
        answers = []
        contains = False
        i = 0
        while i < len(query_words):
            if query_words[i] in postings:
                file_list = set((postings[query_words[i]])[1])
                contains = True
                answers.append(file_list)
                break
            i += 1

        if contains:
            for i in range(i + 1, len(query_words)):
                if query_words[i] in postings:
                    file_list = set((postings[query_words[i]])[1]) & answers[len(answers) - 1]
                    if not len(file_list) == 0:
                        answers.append(file_list)

        for i in range(len(answers) - 1, -1, -1):
            list_out = list(answers[i])
            list_out.sort()
            print('ًRank', len(answers) - i, ':\n >', list_out)
            # print('result containing:', i + 2, 'last words:', '\n > ', answers[i])


def load_singulars(singular_dict):
    singular_dict['آداب'] = 'ادب'
    singular_dict['قواعد'] = 'قاعده'
    singular_dict['معابد'] = 'معبد'
    singular_dict['ادبا'] = 'ادیب'
    singular_dict['مساجد'] = 'مسجد'
    singular_dict['آثار'] = 'اثر'
    singular_dict['قوانین'] = 'قانون'
    singular_dict['معابر'] = 'معبر'
    singular_dict['قلل'] = 'قله'
    singular_dict['مظاهر'] = 'مظهر'
    singular_dict['مناظر'] = 'منظر'
    singular_dict['امراض'] = 'مرض'
    singular_dict['کتب'] = 'کتاب'
    singular_dict['اساطیر'] = 'اسطوره'
    singular_dict['امرا'] = 'امیر'
    singular_dict['حقایق'] = 'حقیقت'
    singular_dict['آفاق'] = 'افق'
    singular_dict['اخبار'] = 'خبر'
    singular_dict['اماکن'] = 'مکان'
    singular_dict['ابیات'] = 'بیت'
    singular_dict['ادوار'] = 'دوره'
    singular_dict['ادیان'] = 'دین'
    singular_dict['دفاتر'] = 'دفتر'
    singular_dict['تصاویر'] = 'تصویر'
    singular_dict['محافل'] = 'محفل'
    singular_dict['ذخایر'] = 'ذخیره'
    singular_dict['مکاتب'] = 'مکتب'
    singular_dict['روابط'] = 'رابطه'
    singular_dict['سوانح'] = 'سانحه'
    singular_dict['علل'] = 'علت'
    singular_dict['مراکز'] = 'مرکز'


if __name__ == '__main__':
    singulars = {}
    tokens = []
    posting_lists = {}
    champion_lists = {}
    posting_lists_with_tf = {}
    idf_list = {}
    docs_len_list = {}
    word_count = {}

    centroids = []
    membership = {}
    doc_vectors = {}

    main()

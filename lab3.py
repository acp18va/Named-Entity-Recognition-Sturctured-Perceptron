# ======================================================================================================================
# Importing
# ======================================================================================================================

import copy
from collections import Counter
import itertools
import operator
from random import shuffle
from random import seed
import time
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# setting random seed
seed(123)

# Declaring labels
labels = ["O", "PER", "LOC", "ORG", "MISC"]

# declaring variables to store counts for learning graph
phi_1_learn = []
phi_merge_learn = []


# ======================================================================================================================
# Taking file input and converting it to list of tuples
# ======================================================================================================================

def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    # reading the input file
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)


# ======================================================================================================================
# Extracting features for phi_1
# ======================================================================================================================

def feature_extraction_phi_1(train_data):
    temp = []
    new_dict = {}
    # creating one list of all the tuples
    for list in train_data:
        for pair in list:
            temp.append(pair)
    # counting the frequency of the keys in the list
    train_dict = Counter(temp)
    # Creating a new dictionary with the frequencies(cw_cl_counts)
    for key, value in train_dict.items():
        word, tag = key
        new_dict.update({word + "_" + tag: train_dict[key]})
    return new_dict


# ======================================================================================================================
# phi_1 function
# ======================================================================================================================

def phi_1(list, train_dict_phi_1):
    # counting the elements of the list
    sent = Counter(list)
    dict = {}
    # making a new dictionary with the elements present in train_dict_phi_1(cl_cw_count)
    for key, val in sent.items():
        if key in train_dict_phi_1.keys():
            dict.update({key: val})
    return dict


# ======================================================================================================================
# function to extract features for phi_2 (previous tag_current_tag)
# ======================================================================================================================

def feature_extraction_phi_2(train_data):
    temp = []
    new_dict = Counter()
    # for each list of tuples adding a none element in the beginning
    for list in train_data:
        list = [("None", "None")] + list
        # creating bigrams from the list
        n_grams = zip(*[list[i:] for i in range(2)])
        # from each bigram creating previous tag_current tag features
        for tuple in n_grams:
            tag1 = tuple[0][1]
            tag2 = tuple[1][1]
            temp.append(tag1 + "_" + tag2)
    # counting the features
    train_dict_2 = Counter(temp)
    for key, value in train_dict_2.items():
        new_dict.update({key: train_dict_2[key]})

    return new_dict


# ======================================================================================================================
# phi_2 function
# ======================================================================================================================

def phi_2(sent, train_dict_phi_2):
    # we add the None tag in beginning
    sent = ['None_None'] + sent
    # creating the list of tags
    temp_tag = []
    for key in sent:
        p = key.split("_")
        temp_tag.append(p[1])
    # creating bigrams from temp_tag
    n_grams = zip(*[temp_tag[i:] for i in range(2)])
    bigram = ["_".join(n_gram) for n_gram in n_grams]
    # counting the elements in bigram
    count = Counter(bigram)
    # making a new dictionary with the elements present in train_dict_phi_2
    dict = {}
    for key, val in count.items():
        if key in train_dict_phi_2.keys():
            dict.update({key: val})
    return dict


# ======================================================================================================================
# train function for both the features models
# ======================================================================================================================

def train(train_data, train_dict_phi_1, train_dict_phi_2, listoftags, feature, iterations):
    # creating zero weights dictionary for phi_1, phi_1 and phi_2
    zero_weights = {}
    if feature == 1:
        zero_weights.update({key: 0 for key in train_dict_phi_1.keys()})
    else:
        weights = merge_dicts(train_dict_phi_1, train_dict_phi_2)
        zero_weights.update({key: 0 for key in weights.keys()})
    # declaring list_of_weights to store updated weights after each iteration
    list_of_weights = []
    for i in range(iterations):
        phi_1_error_count=0
        phi_merge_error_count = 0
        # shuffling my training data
        shuffle(train_data)
        # for each list or sentence in my training data
        for obj in train_data:
            # creating a list of correct sentences
            y_true = []
            for l in obj:
                word, tag = l
                y_true.append(word + "_" + tag)
            # calling function to create all possible combinations
            new_comb = make_combinations(obj, listoftags)
            # creating a list phi to store a counted dictionary for all possible combinations
            phi = []
            for sent in new_comb:
                # calling respective phi functions to create counted dictionaries and appending them in phi
                if feature == 1:
                    phi.append(phi_1(sent, train_dict_phi_1))
                else:
                    phi.append(merge_dicts(phi_1(sent, train_dict_phi_1), phi_2(list(sent), train_dict_phi_2)))
            # declaring a list to store the score of all the dictionaries
            sent_score = []
            # calculating the score for each sent dict in phi
            for sent_dict in phi:
                score = 0
                # for each element in dict, multiplying its frequency with its value in zero weights dictionary
                for key, val in sent_dict.items():
                    if key in zero_weights.keys():
                        # adding the score if the key present in zero weights dictionary
                        score += val * zero_weights[key]
                # appending the score of each sent
                sent_score.append(score)

            # finding maximum score from list of scores
            index, value = max(enumerate(sent_score), key=operator.itemgetter(1))
            # the predicted sentence is y_hat
            y_hat = y_hat_sent(new_comb[index])
            # comparing if the true and predicted sentence are the same
            if y_hat != y_true:
                if feature == 1:
                    phi_1_error_count += 1
                else:
                    phi_merge_error_count += 1
                # calling respective phi functions for each model
                if feature == 1:
                    phi_true = phi_1(y_true, train_dict_phi_1)
                    phi_pred = phi_1(y_hat, train_dict_phi_1)
                else:
                    phi_true = merge_dicts(phi_1(y_true, train_dict_phi_1), phi_2(list(y_true), train_dict_phi_2))
                    phi_pred = merge_dicts(phi_1(y_hat, train_dict_phi_1), phi_2(list(y_hat), train_dict_phi_2))
                # updating the values if the predicted sentence is not the same as the correct sentence
                # increasing the value of correct sentence
                for key in phi_true:
                    if key not in zero_weights:
                        continue
                    else:
                        zero_weights[key] = zero_weights[key] + phi_true[key]
                # reducing the value of wrong prediction
                for key in phi_pred:
                    if key not in zero_weights:
                        continue
                    else:
                        zero_weights[key] = zero_weights[key] - phi_pred[key]
        # adding the updated weights to list_of_weights
        list_of_weights.append(copy.deepcopy(zero_weights))
        if feature==1:
            phi_1_learn.append(phi_1_error_count)
        else:
            phi_merge_learn.append(phi_merge_error_count)
    # calling the function to average the list_of_weights
    average_weights_dict = average_weight(list_of_weights)
    # returning the averaged dictionary
    return average_weights_dict


# ======================================================================================================================
# predict function for both the feature models
# ======================================================================================================================

def predict(list_of_weights, test_set, listoftags, feature):
    # declaring the lists to hold the correct and predicted tags
    y_true_1 = []
    y_hat_1 = []
    # for each list in the testing data
    for obj in test_set:
        # making a list of correct labels
        y_true = y_true_test(obj)
        y_true_1.append(y_true)
        # creating a list of all possible combinations
        new_comb = make_combinations(obj, listoftags)
        # creating a list phi to store a counted dictionary for all possible combinations
        phi = []
        for sent in new_comb:
            # calling respective phi functions to create counted dictionaries and appending them in phi
            if feature == 1:
                phi.append(phi_1(sent, train_dict_phi_1))
            else:
                phi.append(merge_dicts(phi_1(sent, train_dict_phi_1), phi_2(list(sent), train_dict_phi_2)))
        # declaring a list to store the score of all the dictionaries
        sent_score = []
        # calculating the score for each sent dict in phi
        for sent_dict in phi:
            score = 0
            # for each element in dict, multiplying its frequency with its value in zero weights dictionary
            for key, val in sent_dict.items():
                # adding the score if the key present in zero weights dictionary
                if key in list_of_weights.keys():
                    score += val * list_of_weights[key]
            # appending the score of each sent
            sent_score.append(score)
        # finding maximum score from list of scores and its index in sent_score
        index, value = max(enumerate(sent_score), key=operator.itemgetter(1))
        # making a list od predicted labels
        y_hat = new_comb[index]
        y_hat_1.append(y_hat_test(y_hat))
    # returning the lists of predicted and correct labels
    return y_hat_1, y_true_1


# ======================================================================================================================
# test function for both feature models
# ======================================================================================================================

def test(y_predicted, y_true, feature):
    # flattening the list of lists into a list for y_true and y_predicted
    y_predicted = list(itertools.chain.from_iterable(y_predicted))
    y_true = list(itertools.chain.from_iterable(y_true))

    # Creating arrays of correct and predicted labels
    true_array = np.asarray(y_true)
    predicted_array = np.asarray(y_predicted)

    # calculating the f1_score for both feature models
    f1_micro = f1_score(true_array, predicted_array, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    # printing the result of each model
    if feature == 1:
        print("\n \n f1_score for phi_1 feature set is : ", f1_micro)
    else:
        print("\n \n f1_score for phi_1 & phi_2 feature set is : ", f1_micro)


# ======================================================================================================================
# function to create all possible combinations
# ======================================================================================================================

def make_combinations(obj, listoftags):
    words = []
    comb = []

    for tuple in obj:
        word, tag = tuple
        words.append(word)
    # for each word , attaching each label and making a list
    for word in words:
        temp = []
        for tag in listoftags:
            temp.append(word + "_" + tag)
        comb.append(temp)
    # making the list of all combination of above list
    new_comb = list(itertools.product(*comb))
    return new_comb


# ======================================================================================================================
# function to create a list of predicted elements
# ======================================================================================================================

def y_hat_sent(sent):
    # making a list of predicted elements
    list = []
    for key in sent:
        list.append(key)
    # returning the list for y_hat
    return list


# ======================================================================================================================
# function to create a list of correct tags
# ======================================================================================================================

def y_true_test(sent):
    # making a list of tags from the tuples
    tags = []
    for tuple in sent:
        word, tag = tuple
        tags.append(tag)
    # returning the correct tags
    return tags


# ======================================================================================================================
# function to create a list of predicted tags
# ======================================================================================================================

def y_hat_test(sent):
    # making a list of tags from the tuples
    tags = []
    for key in sent:
        list = key.split("_")
        tags.append(list[1])
    # returning the predicted tags
    return tags


# ======================================================================================================================
# function to merge the dictionaries
# ======================================================================================================================

def merge_dicts(*dict_args):
    # updating both the dictionaries in a single dictionary
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    # returning the merged dictionary
    return result


# ======================================================================================================================
# function to average the list of dictionaries
# ======================================================================================================================

def average_weight(new_weights):
    sums = Counter()
    counters = Counter()
    # for each dictionary in the list of dictionaries
    for itemset in new_weights:
        sums.update(itemset)
        counters.update(itemset.keys())
        # taking the average of each element
        average_weight_dict = {x: float(sums[x]) / counters[x] for x in sums.keys()}
    # returning the averaged weights dictionary
    return average_weight_dict


# ======================================================================================================================
# function to print the top features for each label in both the models
# ======================================================================================================================

def top_features(updated_weights, listoftags, feature):
    # repeating for each tag in the global list of tags
    for tag in listoftags:
        # declaring a temporary dictionary to store elements for each tag
        tags = {}
        # iterating through the averaged dictionary
        for key, value in updated_weights.items():
            # key ending with the desired tag
            if key.endswith('_' + tag):
                # update into the temporary dictionary
                tags.update({key: value})
        # sorting the dictionary and saving the top ten elements in new dict
        new = sorted(tags.items(), key=lambda key: key[1], reverse=True)[:10]
        # printing the top elements of each tag in both models
        if feature == 1:
            print("\nTop 10 weighted words for " + tag + " in phi_1:", new)
        elif feature == 2:
            print("\nTop 10 weighted words for " + tag + " in phi_1 & phi_2:", new)


# ======================================================================================================================
# function to plot the learning graph
# ======================================================================================================================

def plot(phi_1_learn, phi_merge_learn):
    plt.plot(phi_1_learn, marker='o', label='Phi_1 Error')
    plt.plot(phi_merge_learn, marker='*', label='Phi_merge Error')
    plt.xlabel('Iteration-Learning Progress')
    plt.ylabel('Number of Errors on training data')
    plt.legend()
    plt.grid()
    plt.show()


# =========================================================================================================
# MAIN
# =========================================================================================================

if __name__ == '__main__':
    # taking a timestamp to calculate total time
    start = time.time()
    # taking the training and test data to calculate the list of tuples
    train_data = load_dataset_sents('train.txt')
    test_data = load_dataset_sents('test.txt')
    # making the feature set for both the models
    train_dict_phi_1 = feature_extraction_phi_1(train_data)
    train_dict_phi_2 = feature_extraction_phi_2(train_data)

    # For phi_1 calling train, top_features, predict and test
    print("----------------Running the phi_1 Model----------------")
    updated_weights = train(train_data, train_dict_phi_1, train_dict_phi_2, labels, feature=1, iterations=6)
    top_features(updated_weights, labels, feature=1)
    y_hat_1, y_true_1 = predict(updated_weights, test_data, labels, feature=1)
    test(y_hat_1, y_true_1, feature=1)

    # For phi_1 and phi_2 calling train, top_features, predict and test
    print("\n----------------Running the phi_1 & phi_2 Model----------------")
    updated_weights_1 = train(train_data, train_dict_phi_1, train_dict_phi_2, labels, feature=2, iterations=30)
    top_features(updated_weights_1, labels, feature=2)
    y_hat_2, y_true_2 = predict(updated_weights_1, test_data, labels, feature=2)
    test(y_hat_2, y_true_2, feature=2)

    # taking timestamp to calculate time
    end = time.time()
    # printing total time taken in seconds
    print(end - start)

    # calling plot function to plot the learning graph
    plot(phi_1_learn, phi_merge_learn)



# code used from: https://github.com/chadaeun/weat_replication/blob/0753713a47333827ef9f653d85e08740834ef698/lib/weat.py#L21

import numpy as np
from sympy.utilities.iterables import multiset_permutations
from utils_bias import get_word_vectors, balance_word_vectors, balance_combinations_word_vectors
import pandas as pd


def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)


def weat_association(W, A, B):
    """
    Returns association of the word w in W with the attribute for WEAT score.
    s(w, A, B)
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: (len(W), ) shaped numpy ndarray. each rows represent association of the word w in W
    """
    
    
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)


def weat_differential_association(X, Y, A, B):
    """
    Returns differential association of two sets of target words with the attribute for WEAT score.
    s(X, Y, A, B)
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: differential association (float value)
    """
   
    return np.sum(weat_association(X, A, B)) - np.sum(weat_association(Y, A, B))


def weat_test_p_value(X, Y, A, B):
    """
    Returns one-sided p-value of the permutation test for WEAT score
    CAUTION: this function is not appropriately implemented, so it runs very slowly
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: p-value (float value)
    """
    diff_association = weat_differential_association(X, Y, A, B)
    target_words = np.concatenate((X, Y), axis=0)

    # get all the partitions of X union Y into two sets of equal size.
    idx = np.zeros(len(target_words))
    idx[:len(target_words) // 2] = 1

    partition_diff_association = []
    for i in multiset_permutations(idx):
        i = np.array(i, dtype=np.int32)
        partition_X = target_words[i]
        partition_Y = target_words[1 - i]
        partition_diff_association.append(weat_differential_association(partition_X, partition_Y, A, B))
    
    partition_diff_association = np.array(partition_diff_association)
    return np.sum(partition_diff_association > diff_association) / len(partition_diff_association)


def weat_p_value(X, Y, A, B):
    """
    Returns one-sided p-value of the permutation test for WEAT score
    CAUTION: this function is not appropriately implemented, so it runs very slowly
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: p-value (float value)
    """
    diff_association = weat_differential_association(X, Y, A, B)
    p_value = weat_test_p_value(X, Y, A, B)
    if diff_association < 0:
        p_value = 1 - p_value
       
    return p_value/2
    


def weat_score(X, Y, A, B):
    """
    Returns WEAT score
    X, Y, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between X and Y
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEAT score
    """

    x_association = weat_association(X, A, B)
    y_association = weat_association(Y, A, B)

    

    tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    
    tmp2 = np.std(np.concatenate((x_association, y_association), axis=0))
   
    return tmp1 / tmp2


def wefat_p_value(W, A, B):
    """
    Returns WEFAT p-value
    W, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: not implemented yet
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEFAT p-value
    """
    pass


def wefat_score(W, A, B):
    """
    Returns WEFAT score
    W, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between A and B
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEFAT score
    """
    tmp1 = weat_association(W, A, B)
    tmp2 = np.std(np.concatenate((cos_sim(W, A), cos_sim(W, B)), axis=0))

    return np.mean(tmp1 / tmp2)

def balance_word_vectors(A, B):
    """
    Balance size of two lists of word vectors by randomly deleting some vectors in larger one.
    If there are words that did not occur in the corpus, some words will ignored in get_word_vectors.
    So result word vectors' size can be unbalanced.
    :param A: (len(words), dim) shaped numpy ndarrary which is word vectors
    :param B: (len(words), dim) shaped numpy ndarrary which is word vectors
    :return: tuple of two balanced word vectors
    """

    diff = len(A) - len(B)

    if diff > 0:
        A = np.delete(A, np.random.choice(len(A), diff, 0), axis=0)
    else:
        B = np.delete(B, np.random.choice(len(B), -diff, 0), axis=0)

    return A, B


def balance_combinations_word_vectors(A, B):
    diff = len(A) - len(B)
    
    A_list = []
    B_list = []
    
    if diff > 0:
        for a in A:
            A_list.append(np.delete(A, A.index(a), axis=0))
        B_list.extend([B] * len(A_list))
    elif diff < 0:
        for b in B:
            B_list.append(np.delete(B, B.index(b), axis=0))
            A_list.extend([A] * len(B_list))
    else:
        A_list.append(A)
        B_list.append(B)

    return A_list, B_list


def compute_weat_score(weat_test_cases, word_vectors, bool_p=True):
    result_df = pd.DataFrame(columns=['Data Name', 'Targets', 'Attributes', 'Method', 'Score', '# of target words', '# of attribute words'])

    for data_name, data_dict in weat_test_cases.items():
        if data_dict['method'] == 'weat' and data_dict['attributes'] == 'Male attributes vs Female attributes':
            X_key = data_dict['X_key']
            Y_key = data_dict['Y_key']
            A_key = data_dict['A_key']
            B_key = data_dict['B_key']


            X = get_word_vectors(words = data_dict[X_key], model = word_vectors)
            Y = get_word_vectors(words = data_dict[Y_key], model = word_vectors)
            A = get_word_vectors(words = data_dict[A_key], model = word_vectors)
            B = get_word_vectors(words = data_dict[B_key], model = word_vectors)
            
            num_target = len(X)
            num_attr = len(A)
            
            if not len(X) == len(Y) or not len(A) == len(B):
                temp_df = compute_combinations_weat_score(data_name, word_vectors, weat_test_cases)
                result_df = result_df.append(
                        {
                            'Data Name': data_name,
                            'Targets': data_dict['targets'],
                            'Attributes': data_dict['attributes'],
                            'Method': data_dict['method'],
                            'Score': np.mean(temp_df['Score']),
                            'p-value': np.mean(temp_df['p-value']),
                            '# of target words': num_target,
                            '# of attribute words': num_attr,
                        }, ignore_index=True
                )
                
            else:
                score = weat_score(X, Y, A, B)
                if bool_p:
                    p_value = weat_p_value(X, Y, A, B)
                    result_df = result_df.append(
                            {
                                'Data Name': data_name,
                                'Targets': data_dict['targets'],
                                'Attributes': data_dict['attributes'],
                                'Method': data_dict['method'],
                                'Score': score,
                                'p-value': p_value,
                                '# of target words': num_target,
                                '# of attribute words': num_attr,
                            }, ignore_index=True
                    )
                else:
                    result_df = result_df.append(
                            {
                                'Data Name': data_name,
                                'Targets': data_dict['targets'],
                                'Attributes': data_dict['attributes'],
                                'Method': data_dict['method'],
                                'Score': score,
                                '# of target words': num_target,
                                '# of attribute words': num_attr,
                            }, ignore_index=True
                    )
    return result_df


def compute_combinations_weat_score(data_name, word_vectors, weat_test_cases):
    result_df = pd.DataFrame(columns=['Data Name', 'Targets', 'Attributes', 'Method', 'Score', '# of target words', '# of attribute words'])
    data_dict = weat_test_cases[data_name]

    X_key = data_dict['X_key']
    Y_key = data_dict['Y_key']
    A_key = data_dict['A_key']
    B_key = data_dict['B_key']

    A_words = [w for w in data_dict[A_key] if w in word_vectors]
    B_words = [w for w in data_dict[B_key] if w in word_vectors]
    X_words = [w for w in data_dict[X_key] if w in word_vectors]
    Y_words = [w for w in data_dict[Y_key] if w in word_vectors]

    
    A_list, B_list = balance_combinations_word_vectors(A_words, B_words)
    X_list, Y_list = balance_combinations_word_vectors(X_words, Y_words)
    
    for x, y in zip(X_list, Y_list):
        X = get_word_vectors(words = x, model = word_vectors)
        Y = get_word_vectors(words = y, model = word_vectors)
        for a, b in zip(A_list, B_list):
            
            A = get_word_vectors(words = a, model = word_vectors)
            B = get_word_vectors(words = b, model = word_vectors)
            
            num_target = len(X)
            num_attr = len(A)

            score = weat_score(X, Y, A, B)
            p_value = weat_p_value(X, Y, A, B)
            
            
            result_df = result_df.append(
                    {
                         'Data Name': data_name,
                         'Targets': data_dict['targets'],
                         'Attributes': data_dict['attributes'],
                         'Method': data_dict['method'],
                         'Score': score,
                         'p-value': p_value,
                         '# of target words': num_target,
                         '# of attribute words': num_attr,
                         'X': x,
                         'Y': y,
                         'A': a,
                         'B': b,
                    }, 
                ignore_index=True)
    return result_df


def most_similar(data_name, word_vectors, weat_test_cases, top):
    word_similar = dict()
    data_dict = weat_test_cases[data_name]
    
    X_key = data_dict['X_key']
    Y_key = data_dict['Y_key']
    A_key = data_dict['A_key']
    B_key = data_dict['B_key']

    words_list = [w for w in data_dict[A_key] if w in word_vectors]
    words_list.extend([w for w in data_dict[B_key] if w in word_vectors])
    words_list.extend([w for w in data_dict[X_key] if w in word_vectors])
    words_list.extend([w for w in data_dict[Y_key] if w in word_vectors])
    
    for word in words_list:
        if word in word_vectors:
            word_similar[word] = word_vectors.most_similar(word, topn=top)
            
    return word_similar

def count_words_vocab(data_name, word_vectors, weat_test_cases):
    word_count = dict()
    data_dict = weat_test_cases[data_name]
    
    X_key = data_dict['X_key']
    Y_key = data_dict['Y_key']
    A_key = data_dict['A_key']
    B_key = data_dict['B_key']

    words_list = [w for w in data_dict[A_key] if w in word_vectors]
    words_list.extend([w for w in data_dict[B_key] if w in word_vectors])
    words_list.extend([w for w in data_dict[X_key] if w in word_vectors])
    words_list.extend([w for w in data_dict[Y_key] if w in word_vectors])
    
    for word in words_list:
        if word in word_vectors:
            word_count[word] = word_vectors.get_vecattr(word, 'count')
            
    return word_count


def check_consistency_weat(x,y, attribute_list, word_vectors):
    result_df = pd.DataFrame(columns=['X', 'Y', 'A', 'B', 'WEAT', 'p-value', 'count X', 'count Y', 'count A', 'count B'])

    X = get_word_vectors(words = [x], model = word_vectors)
    Y = get_word_vectors(words = [y], model = word_vectors)
    results = []
    for a, b in attribute_list:
        A = get_word_vectors(words = [a], model = word_vectors)
        B = get_word_vectors(words = [b], model = word_vectors)
        score = weat_score(X, Y, A, B)
        p_value = weat_p_value(X, Y, A, B)
        c_x = word_vectors.get_vecattr(x, 'count')
        c_y = word_vectors.get_vecattr(y, 'count')
        c_a = word_vectors.get_vecattr(a, 'count')
        c_b = word_vectors.get_vecattr(b, 'count')
        result = {'X': x, 'Y':y, 'A': a, 'B': b, 'WEAT': score, 'p-value': p_value, 'count X': c_x, 'count Y':c_y, 'count A':c_a, 'count B':c_b}
        result_df = result_df.append(result, ignore_index=True)
    return result_df
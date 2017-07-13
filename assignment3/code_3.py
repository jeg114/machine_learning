"""
assignment2_jeg114.py: Code to solve Problem 3 (Movie recommendation)
Jesus E. Garcia
Machine Learning 
9 - 03 - 2017
"""

import csv
from pprint import PrettyPrinter as pp
from sklearn import preprocessing
import random
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing.dummy import Pool as thread_pool

def csv_import(filename, features=False):
    with open(filename) as f:
    #Ignore header line 
        f.readline()
        if features:
            return [map(int, i) for i in list(csv.reader(f))]
        else:
            return [[int(u), int(m), float(r)] for u, m, r in list(csv.reader(f))]

def square_err(predictor, predict_inputs, correct_outputs, test=False):
    pre_err = 0
    n = 0
    if type(correct_outputs[0]) == tuple:
        for i in range(len(predict_inputs)):
            if predict_inputs[i][0] != correct_outputs[i][0]:
                print "MISMATCHED USERS!"
                raise Exception
            else:
                    predict = predictor(predict_inputs[i][0][0],
                                        predict_inputs[i][1])
                    pre_err += (predict - correct_outputs[i][1])**2
                    n+=1
    else:
        for i,correct in zip(predict_inputs, correct_outputs):
            predict = predictor(i)
            pre_err += (predict - correct)**2
            n+=1
    pre_err = pre_err / n
    return pre_err, n

def cross_validation(class_gen, options_list, train_data, train_labels, 
                        K, test_data, test_labels, 
                        MULTITHREAD = True, random_n = 0, debug =False):
    data_len = len(train_data)
    val_n = data_len / K if K>0 else 0
    trai_n = val_n * (K-1)
    train = zip(train_data, train_labels)
    random.shuffle(train)
    shuffled_data, shuffled_labels = map(list, zip(*train))

    best_err = 4e15
    all_avg=[]
    for options in options_list:
        avg_err = 0
        all_runs =[]
        for i in range(K) if K > 0 else [1]:
            if K > 0:
                if random_n > 0:
                    train = zip(train_data, train_labels)
                    random.shuffle(train)
                    shuffled_data, shuffled_labels = map(list, zip(*train))
                    train_set = shuffled_data[:random_n]
                    train_labels_set = shuffled_labels[:random_n]
                    val_set = shuffled_data[random_n:]
                    val_labels_set = shuffled_labels[random_n:]
                if K == len(shuffled_data):
                    train_set = shuffled_data[0:i] + shuffled_data[i+1:-1]
                    train_label_set = shuffled_labels[0:i]\
                                        + shuffled_labels[i+1:-1]
                    val_set = shuffled_data[i:i+1]
                    val_labels_set = shuffled_labels[i:i+1]
                else:
                    base1 = 0
                    end1 = i*val_n 
                    base2 = end1 + val_n-1 
                    end2 = -1
                    train_set = shuffled_data[base1:end1] + shuffled_data[base2:end2]
                    train_label_set = shuffled_labels[base1:end1] \
                                        + shuffled_labels[base2:end2]
                    val_set = shuffled_data[end1:base2]
                    val_labels_set = shuffled_labels[end1:base2]
                data_tuple = (class_gen, options, train_data, train_labels,
                            val_set, val_labels_set)
                all_runs.append(data_tuple)
            else:
                predictor = class_gen(options, train_data, train_labels)
                avg_err,n = square_err(predictor, test_data, test_labels, True)
        if K > 0:
            if MULTITHREAD:
                thread_errs = []
                pool = thread_pool()
                errs = pool.map(train_and_check, all_runs)
                pool.close()
                pool.join()
            else:
                errs = []
                for run in all_runs:
                    errs += [train_and_check(run)]
            avg_err = sum(errs)/K
        all_avg.append(avg_err)
        if debug:
            print "Err:  ", avg_err,  " with options:", options
        if avg_err < best_err:
            best_err = avg_err
            best_opt = options
    if debug:
        print "\n Best err:  ", best_err,  " with options:", best_opt
    if K > 0:
        predictor = class_gen(best_opt, train_data, train_labels)
        train_err,n = square_err(predictor, train_data, train_labels)
        test_err,n = square_err(predictor, test_data, test_labels)
        if debug:
            print "Training on full set -> Train err: ", train_err , \
                    "Test error: ", test_err
        print "\t", best_opt["lambda"], " ", test_err
    return test_err

#Callback for threads, takes tuple of: 
# (options, train_set, train_label_set, val_set, val_labels_set)
def train_and_check(data_tuple):
    class_gen, options, train_data, \
            train_labels, val_data, val_labels = data_tuple
    predictor = class_gen(options, train_data, train_labels)
    err,n = square_err(predictor, val_data, val_labels)
#    thread_errs.append(err)
    return err

def do_transform(transform, vects):
    #Assumes 0 coordinate as 1 present and indeces given accordingly
    transformation = transform["type"]
    new_vects = []
    for vect in vects:
        if transformation == "subsets":
            new_vect = [1]
            indeces_list = transform["indices"]
            for indeces in indeces_list:
                new_vect.append(sum([vect[i] for i in indeces]))
        else:
            new_vect = vect
        if "order" in transform:
            order = transform["order"]
            interaction = transform["interaction"]
            P = preprocessing.PolynomialFeatures(order, interaction) 
            new_vect = P.fit_transform(
                            np.array(new_vect[1:]).reshape(1,-1)).tolist()[0]
        new_vects.append(new_vect)
    return new_vects

def baseline_trainer(options, train_data_items, train_label_items):
    #Data: items of dictionary of user,movie = vectors
    #Label: items of dictionary of user,movie = ratings
    #Prediction: Gen weights for a given user and predict from movie

    user_weights = {}
    lambda_param = options["lambda"]
    if lambda_param == 0:
        normalize_param = False
#        normalize_param = True
    else:
        normalize_param = options["normalize"] 
    unknown_user = options["unknown_user"]
    transform = options["transform"]
#    print transform

    #Group all movies of a user
    train_label_dict = dict(train_label_items)
    train_label ={}
    train_data = {}
    all_vects = []
    all_ratings = []
    for (user, movie), feat in train_data_items:
        if user in train_data:
            train_data[user] += [map(float, feat)] \
                                if normalize_param else [feat]
            train_label[user] += [train_label_dict[user, movie]]
        else:
            train_data[user] = [map(float, feat)] \
                                if normalize_param else [feat]
            train_label[user] = [train_label_dict[user, movie]]
        all_vects += [map(float, feat)]
        all_ratings.append(train_label_dict[user, movie])

    if normalize_param:
        scaler = preprocessing.StandardScaler().fit(np.matrix(all_vects))
        ratings_mean = sum(all_ratings)/len(all_ratings)
    
    for user in train_data:
        if normalize_param:
            feature_vects = scaler.transform(np.array(train_data[user])
                                ).tolist()
            feature_vects[0][0] = 1
            ratings_vect = map(lambda i: i-ratings_mean, train_label[user])
        else:
            feature_vects = train_data[user]
            ratings_vect = train_label[user]
        if transform:
            feature_vects = do_transform(transform, feature_vects)
        X = np.matrix(feature_vects)
        X_T = X.transpose()
        X_X_T = X_T * X
        if lambda_param == 0:
            inv = np.linalg.inv(X_X_T)
        else:
            lambda_d = np.matrix(lambda_param * np.identity(X.shape[1]))
            inv = np.linalg.inv(X_X_T - lambda_d)
        inv_X_T = inv *  X_T
        soln = inv_X_T * np.transpose(np.matrix(ratings_vect))
        user_weights[user] = soln
    avg = 0
    if unknown_user <= 0:
        total =  sum([sum(train_label[i]) for i in train_label])
        elements =  sum([len(train_label[i]) for i in train_label])
        avg = total/elements
    def predictor(user, movie_vect):
        if transform:
            movie_vect = do_transform(transform, [movie_vect])[0]
     	if user in user_weights:
            if normalize_param:
                movie_vect = scaler.transform(\
                            np.array(movie_vect).reshape(1,-1)).tolist()
                feature_vects[0][0] = 1
     	    pre =  np.transpose(user_weights[user])*\
                    np.transpose(np.matrix(movie_vect))
            
            return float(pre)+ratings_mean if normalize_param else float(pre)
     	else:   
            print user
            raise Exception
            print "unknown"
     	    if unknown_user >= 0:
     	        return unknown_user
     	    else:
                return avg
    return predictor
#@profile
def speed_up_user_baseline_trainer(options, feature_vects, ratings_vect):
    #Data: items of dictionary of user,movie = vectors
    #Label: items of dictionary of user,movie = ratings
    #Prediction: Gen weights for a given user and predict from movie

    user_weights = {}
    lambda_param = options["lambda"]
    unknown_user = options["unknown_user"]
    transform = options["transform"]
    if transform:
        feature_vects = do_transform(transform, feature_vects)
    feature_vects = [f[1:] for f in feature_vects]
    scaler = preprocessing.StandardScaler()
    scaler.fit(np.array(feature_vects))
    X = scaler.fit_transform(np.array(feature_vects))
    X = np.matrix(X)
    
    ratings_mean = sum(ratings_vect)/len(ratings_vect)
    
    ratings_vect = map(lambda i: i-ratings_mean, ratings_vect)
    X_T = X.transpose()
    X_X_T = X_T * X
    if lambda_param == 0:
        inv = np.linalg.inv(X_X_T)
    else:
        lambda_d = np.matrix(lambda_param * np.identity(X.shape[1]))
        inv = np.linalg.inv(X_X_T - lambda_d)
    inv_X_T = inv *  X_T
    soln = inv_X_T * np.transpose(np.matrix(ratings_vect))

    user_weights = soln 
    
    avg = 0
    def predictor(movie_vect):
        if transform:
            movie_vect = do_transform(transform, [movie_vect])[0]
        movie_vect = scaler.transform(\
                np.array(movie_vect[1:]).reshape(1,-1))
#     	movie_vect = np.append(np.array([[1]]), movie_vect)
        pre =  np.dot(user_weights.T,np.array(movie_vect).T) + ratings_mean
        return float(pre)    
    return predictor
#@profile
def user_baseline_trainer(options, feature_vects, ratings_vect):
    #Data: items of dictionary of user,movie = vectors
    #Label: items of dictionary of user,movie = ratings
    #Prediction: Gen weights for a given user and predict from movie

    user_weights = {}
    lambda_param = options["lambda"]
    if lambda_param == 0:
        normalize_param = False
#        normalize_param = True
    else:
        normalize_param = options["normalize"] 
    unknown_user = options["unknown_user"]
    transform = options["transform"]
#    print transform

    if normalize_param:
        scaler = preprocessing.StandardScaler().fit(np.matrix(feature_vects))
        ratings_mean = sum(ratings_vect)/len(ratings_vect)
    
        feature_vects = scaler.transform(np.array(feature_vects)
                            ).tolist()
        feature_vects = [[1] +  f[1:] for f in feature_vects] 
        ratings_vect = map(lambda i: i-ratings_mean, ratings_vect)
    if transform:
        feature_vects = do_transform(transform, feature_vects)
    X = np.matrix(feature_vects)
    X_T = X.transpose()
    X_X_T = X_T * X
    if lambda_param == 0:
        inv = np.linalg.inv(X_X_T)
    else:
        lambda_d = np.matrix(lambda_param * np.identity(X.shape[1]))
        inv = np.linalg.inv(X_X_T - lambda_d)
    inv_X_T = inv *  X_T
    soln = inv_X_T * np.transpose(np.matrix(ratings_vect))
    user_weights = soln
    
    avg = 0
    def predictor( movie_vect):
        if transform:
            movie_vect = do_transform(transform, [movie_vect])[0]
        if normalize_param:
            movie_vect = scaler.transform(\
                        np.array(movie_vect).reshape(1,-1)).tolist()
            movie_vect[0][0] = 1
     	pre =  np.transpose(user_weights)*\
                np.transpose(np.matrix(movie_vect))
        
        return float(pre)+ratings_mean if normalize_param else float(pre)
    return predictor
#@profile
def problem3():
    all_parts = True
    ratings_train = csv_import("ratings-train.csv")
    ratings_by_user = {}
    ratings_by_movie = {}
    ratings_by_both_u = {}
    ratings_by_both_m = {}
    ratings_by_both = {}
    all_ratings = []
    all_users = []
    all_movies = []
    for user, movie, rating in ratings_train:
        if user in ratings_by_user:
            ratings_by_user[user].append(rating)
        else:
            ratings_by_user[user] = [rating]
        if movie in ratings_by_movie:
            ratings_by_movie[movie].append(rating)
        else:
            ratings_by_movie[movie] = [rating]
        if user in ratings_by_both_u:
            ratings_by_both_u[user][movie] = rating
        else:
            ratings_by_both_u[user] = {movie:rating}
        if movie in ratings_by_both_m:
            ratings_by_both_m[movie][user] = rating
        else:
            ratings_by_both_m[movie] = {user:rating}
        ratings_by_both[user,movie] = rating
        if not user in all_users:
            all_users.append(user)
        if not movie in all_movies:
            all_movies.append(movie)
        all_ratings.append(rating)

    if all_parts:
        predictor_by_user = {}
        predictor_by_movie = {}
        movies_per_user = 0
        users_per_movie = 0
        for user, ratings in ratings_by_user.iteritems():
            predictor_by_user[user] = np.average(ratings)
            movies_per_user += len(ratings)
        for movie, ratings in ratings_by_movie.iteritems():
            predictor_by_movie[movie] = np.average(ratings)
            users_per_movie += len(ratings)
        movies_per_user = movies_per_user / float(len(predictor_by_user)) 
        users_per_movie = users_per_movie / float(len(predictor_by_movie)) 
    
        print "Users in training", len(predictor_by_user)
        print "Movies in training", len(predictor_by_movie)
        print "Avg movies per user:", movies_per_user
        print "Avg users per movie:", users_per_movie
    
    
        users, movies, ratings = zip(*ratings_train)
    
        user_predictor = lambda i : predictor_by_user[i]
        user_pre_err, train_n = square_err(user_predictor,users,ratings) 
        movie_predictor = lambda i : predictor_by_movie[i]
        movie_pre_err, train_n = square_err(movie_predictor,movies,ratings) 
    
        print "train samples: ", train_n
        print "training err user:", user_pre_err
        print "training err movie:", movie_pre_err
    
        ratings_test =  csv_import("ratings-test.csv")
        users, movies, ratings = zip(*ratings_test)
        # Average of all ratings to be used for sets with no 
        # available data
        avg = np.average(all_ratings)
    
        user_predictor = lambda i : predictor_by_user[i] \
                                    if i in predictor_by_user else avg
        user_pre_err, test_n = square_err(user_predictor,users,ratings)
    
        movie_predictor = lambda i : predictor_by_movie[i]\
                                    if i in predictor_by_movie else avg
        movie_pre_err, test_n = square_err(movie_predictor,movies,ratings) 
    
        print "Test samples: ", test_n
        print "Test err user", user_pre_err
        print "Test err movie", movie_pre_err
    
    

        user_predictor = lambda i : avg
        users, movies, ratings = zip(*ratings_train)
        avg_train_err, train_n = square_err(user_predictor,users,ratings) 
    
        users, movies, ratings = zip(*ratings_test)
        avg_te_err, train_n = square_err(user_predictor,users,ratings) 

        print "training err avg:", avg_train_err
        print "test err avg:", avg_te_err

        ratings_test =  csv_import("ratings-test.csv")
        users, movies, ratings = zip(*ratings_test)
        features =  csv_import("movie-features.csv", True)
        #Movies are 1 indexed
        movie_features = [1] 
        for i in features:
            movie_features.append([1] + [float(feat) for feat in i[1:]])
    
        options = []
    #    for lambda_v in [0.1]:
        ls = [01, 0.05, 0.001, 0.005, 0.0001, 0.00005, 0.00001, 0.000005]
        ls = [0.5,1e-6, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9] #all last vals are same
        ls = [0.4, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
#        ls = [2, 1.5, 0.1, 0.01, 0.001, 0.0001]
        ls = [100000, 50000, 40000, 30000, 20000, 10000, 5000, 1000, 600, 300, 100]
    #    ls = [0.2, 0.01]
        ls = [0]
        for lambda_v in ls:
    #        for normalize in [False]:
            for normalize in [True]:
    #                for unknown_user in [0, 2.5, 5, -1]:
                        options.append({"lambda" : lambda_v, 
                                       "normalize" : True,
                                        "transform" : None,
#                                        "transform" : {"type":"subse", 
#                                                "indices": [range(1,19)],
#                                                "order": 2, 
#                                                "interaction" : False},
                                        "unknown_user" : -1})
    #Different transform options tried
    #    options = [{"lambda": 0.001, "normalize": False, "unknown_user": "Except", "transform" : {"type":"subsets", "indices": [range(1,19)], "order": 2, "interaction" : False}}]
    #    options.append({"lambda": 0.001, "normalize": False, "unknown_user": "Except", "transform" : {"type":"subsets", "indices": [range(1,10), range(10,19)], "order": 2, "interaction" : True}})
    #    options.append({"lambda": 0.001, "normalize": False, "unknown_user": "Except", "transform" : {"type":"subsets", "indices": [range(1,10), range(10,19)], "order": 2, "interaction" : False}})
    #    options.append({"lambda": 0.001, "normalize": False, "unknown_user": "Except", "transform" : {"type":"subsets", "indices": [range(1,19)], "order": 3, "interaction" : False}})
    #    options.append({"lambda": 0.001, "normalize": False, "unknown_user": "Except", "transform" : {"type":"o", "indices": [range(1,19)], "order": 2, "interaction" : True}})
    #    options.append({"lambda": 0.001, "normalize": False, "unknown_user": "Except", "transform" : {"type":"o", "indices": [range(1,19)], "order": 2, "interaction" : False}})
        all_features = []
        for movie in all_movies:
                all_features.append(movie_features[movie])
    
        test_data = {}
        test_labels = {}
        preprocess = False
        if preprocess:
            scaler = preprocessing.StandardScaler().fit(np.matrix(all_features))
            mean = sum(all_ratings)/len(all_ratings)
            print "mean: ", mean
        for user, movie, rating in zip(users, movies, ratings):
            if preprocess:
                test_data[user, movie] = scaler.transform(\
                                    np.array(movie_features[movie]
                                            ).reshape(1,-1)).tolist()[0]
                test_labels[user, movie] = rating - mean
            else:
                test_data[user, movie] = movie_features[movie]
                test_labels[user, movie] = rating
    
        all_errs =[]
        i = 0
        for user in ratings_by_both_u:
            ratings_vect =[]
            user_vects = []
            user_test_data =[]
            user_test_labels = []
            for movie in ratings_by_both_u[user]:
                if preprocess:
                   ratings_vect += [ratings_by_both_u[user][movie] - mean]
                   tmp_vect = scaler.transform(\
                                np.array(movie_features[movie]
                                        ).reshape(1,-1)).tolist()[0]
                   tmp_vect[0] = 1
                   user_vects += [tmp_vect] 
                else:
                    ratings_vect += [ratings_by_both_u[user][movie]]
                    user_vects += [movie_features[movie]]
            user_test_data += [test_data[t_user, t_movie] for t_user, t_movie in test_data.keys() if t_user == user]
            user_test_labels += [test_labels[t_user, t_movie] for t_user, t_movie in test_data.keys() if t_user == user]
            print user, ": ", len(user_vects)
            if len(user_vects) > 300:
                print len(user_vects)
                k = 100
                rm_n = 100
                break
            else :
                k = len(user_vects)
                rm_n = 0
            all_errs.append(cross_validation(speed_up_user_baseline_trainer, options,
                                            user_vects,
                                            ratings_vect, k,
                                            user_test_data, 
                                            user_test_labels,
                                            False, rm_n, False)
                                            )
#            if user == 50:
#                break
        print "MEAN ERR", sum(all_errs)/ len(all_errs)

    p_Ks = [4, 6, 10,14]
    p_alphas = [0.2]
    p_lambdas = [0.01,0.00002, 0.00001, 0.000005, 0.00005]
    iterations = range(200)
    avg = np.average(all_ratings)
    for u, m in ratings_by_both:
        ratings_by_both[u,m] -= avg
    for p_K in p_Ks:
        for p_alpha in p_alphas:
            for p_lambda in p_lambdas:
                user_weights = {u:np.random.rand(p_K) for u in all_users}
                movie_learn_fs = {m:np.random.rand(p_K) for m in all_movies}
                err = 0
                new_err = 0
                i = 0
                for (u, m),r in ratings_by_both.items():
                    i+=1
                    new_err += (np.dot(user_weights[u],
                                movie_learn_fs[m])  -  r)**2
                new_err = new_err / float(i)
                first = True
                for i in iterations:
                    print i
                    print "current err:", err
                    gradient_norm = 0
                    deviations = {(u, m) : np.dot(movie_learn_fs[m].T, 
                                               user_weights[u]) - 
                                                ratings_by_both[u,m] 
                                           for u, m in ratings_by_both 
                                           }
                    m_weighted_dev = {}
                    u_weighted_dev = {}
                    for (u,m), dev in deviations.items():
                        if u in m_weighted_dev:
                            m_weighted_dev[u] += dev*movie_learn_fs[m]
                        else:
                            m_weighted_dev[u] = dev*movie_learn_fs[m] 
                        if m in u_weighted_dev:
                            u_weighted_dev[m] += dev*user_weights[u]
                        else:
                            u_weighted_dev[m] = dev*user_weights[u]
                    if (new_err > err and not first):
#                        user_weights = old_user_weights
#                        movie_learn_fs = old_movie_learn_fs
                        break
#                    else:
#                        old_user_weights = user_weights
#                        old_movie_learn_fs = movie_learn_fs

#Attempt at doing line backtracking
#                    while (new_err > err - p_alpha/float(2)*gradient_norm):
#                        print "smaller than new", err - p_alpha/float(2)*gradient_norm
                    if True:
#                        print "repeat alpha: ", p_alpha
                        
                        for user in all_users:
                            u_gradient =  (m_weighted_dev[u] + 
                                                    p_lambda*user_weights[user])
                            gradient_norm = np.dot(u_gradient, u_gradient) 
                            user_weights[user] = user_weights[user] -\
                                        p_alpha * u_gradient
#                                        p_alpha * u_gradient/gradient_norm
                        for movie in all_movies:
                            m_gradient =  (u_weighted_dev[m] + 
                                            p_lambda*movie_learn_fs[movie])
                                                                 
                            movie_learn_fs[movie] = movie_learn_fs[movie] \
                                            - p_alpha* m_gradient
#                                            - p_alpha* m_gradient/gradient_norm
                            gradient_norm = np.dot(m_gradient, m_gradient)
                        err = new_err
                        new_err = 0
                        i = 0
                        for (u, m),r in ratings_by_both.items():
                            i+=1
                            new_err += (np.dot(user_weights[u],
                                        movie_learn_fs[m])  -  r)**2
                        new_err = new_err / float(i)
                        print "\tnew err: ", new_err
                        ratings_test =  csv_import("ratings-test.csv")
                        avg_err = 0
                        first = False
                        
                tests = 0
                for user, movie, rating in ratings_test:
                    if user in user_weights and movie in movie_learn_fs:
                        tests += 1
                        avg_err += (np.dot(user_weights[user],
                                        movie_learn_fs[movie])
                                        - (rating-avg))**2
                avg_err = avg_err / float(tests)
                print err
                print  avg_err, ", ", p_lambda, ",", p_K,",", p_alpha 


problem3() 









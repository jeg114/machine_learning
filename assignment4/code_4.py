"""
assignment4_jeg114.py: Code to solve Problem 4 (SVM for digits classification)
Jesus E. Garcia
Machine Learning 
9 - 03 - 2017
"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing.dummy import Pool as thread_pool
from itertools import product

def get_digits(filename, label_pos, label_neg, split=False):
    raw_data = np.loadtxt(open(filename))
    data = np.zeros((1,raw_data.shape[1]-1))
    pos_data = np.zeros((1,raw_data.shape[1]-1))
    neg_data = np.zeros((1,raw_data.shape[1]-1))
    labels = []
    i=0
    for row in raw_data:
        if row[0] in label_pos:
            data = np.append(data, [row[1:]], axis=0)
            if split:
                pos_data = np.append(pos_data, [row[1:]], axis=0)
            labels.append(1)
        if row[0] in label_neg:
            data = np.append(data, [row[1:]], axis=0)
            if split:
                neg_data = np.append(neg_data, [row[1:]], axis=0)
            labels.append(-1)
    data = np.delete(data, (0), axis=0)
    if split:
        pos_data = np.delete(pos_data, (0), axis=0)
        neg_data = np.delete(neg_data, (0), axis=0)
        return data, np.array(labels), pos_data, neg_data
    else:
        return data, np.array(labels)

def get_error(y1, y2):
    errors = [1 for a,b in zip(y1,y2) if a != b]
    return len(errors)/float(len(y1))

def thread_rbf_cv(args):
    c_val, gamma_val, k, data, labels = args
    rbf_svm = SVC(C=c_val, kernel="rbf", shrinking=False, gamma=gamma_val)
    err = 1-np.mean(cross_val_score(rbf_svm, data, \
                                                        labels, cv=k))
#    print "c: ", c_val , "g: ", gamma_val, "err: ", err
    return ((c_val, gamma_val), err)

def rbf_cross_val(data, labels, min_g, max_g, min_c, max_c, steps, k, n_pca):
    if n_pca != 0:
        pca_calc =  PCA(n_components = n_pca).fit(data)
        data = pca_calc.transform(data)
    step_size = (max_g - min_g)/float(steps)
    gammas = [min_g + i*step_size for i in range(0,steps)] 
    if max_c != min_c:
        step_size = (max_c - min_c)/float(steps)
        cs = [min_c + i*step_size for i in range(0,steps)] 
    else:
        cs = [1]
    MULTITHREAD = True
    if MULTITHREAD:
        thread_args = [(c, g, k, data, labels) for c,g in product(cs,gammas)]
        errs_items = []
        pool = thread_pool()
        errs_items = pool.map(thread_rbf_cv, thread_args)
        pool.close()
        pool.join()
        errs = dict(errs_items)
    else:
        errs = {}
        for c in cs:
            for g in gammas: 
                rbf_svm = SVC(C=c, kernel="rbf", shrinking=False, gamma=g)
                err = 1-np.mean(cross_val_score(rbf_svm, data, \
                                                        labels, cv=k))
                errs[c, g] = 100*err 
                print "c: ", c , "g: ", g, "err: ", err
    best_c, best_g = min(errs, key=errs.get)
    best_err = errs[best_c, best_g]
    if n_pca != 0:
        return [best_c, best_g, best_err, pca_calc]
    else:
        all_errs, all_gammas = zip(*[(val, key[1])\
                                        for key, val in errs.items()])
        return [best_c, best_g, best_err, all_errs, all_gammas]

def problem4():
    
    part_a = True
    part_b = False
#    part_b = True
#    part_c = False
    part_c = True

    train_data, train_labels = get_digits("data/zip.train", [2], [8])
    test_data, test_labels = get_digits("data/zip.test", [2], [8])

    if part_a:
        print "------- Part a -------"
        print "Cross validation of Gamma for rbf kernel SVM to classify 2/8 "
        errs = []
        K=15
        cv_values=100
        best_c, best_g, best_err, errs, gammas = rbf_cross_val(
                                                       data = train_data, 
                                                       labels = train_labels,
                                                       min_g=0.001, 
                                                       max_g=0.03, 
                                                       min_c=1, 
                                                       max_c=1, 
                                                       steps=cv_values, 
                                                       k=K, 
                                                       n_pca=0)
        rbf_svm = SVC(C=best_c, kernel="rbf", shrinking=False, gamma=best_g)
        rbf_svm.fit(train_data, train_labels)
        train_err =  1-rbf_svm.score(train_data, train_labels)
        test_err = 1-rbf_svm.score(test_data, test_labels)
        fig = plt.figure("1")
        ax = fig.add_subplot(111)
        ax.text(0, max(errs)*0.8, 
                        "For optimum gamma: " + "%.3f"%(best_g) + \
                        "\n   CV err: " + "%.3f"%(100*best_err) + \
                        "%\n  Train err: " + "%.3f"%(100*train_err) + \
                        "%\n  Test err: " + "%.3f"%(100*test_err) +\
                        "%" , style='italic',
                bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
        plt.xlabel("Gamma")
        plt.ylabel("Error (%)")
        ax.set_title(str(K)+"-fold CV over "+ str(cv_values) +" gamma values")
        plt.scatter(gammas, errs, color='b', s=3)
        fig.savefig("4a_cv")
    
    if part_b:
        print "------- Part b -------"
        print "Cross validating, training and testing for\
                different PCA dimensins"
        K=20
        k_cv_errs = []
        k_train_errs = []
        k_test_errs = []
        pca_dims = range(1,80)
        for pca_dim in pca_dims:
            best_c, best_g, cv_err, pca_calc = rbf_cross_val(
                                                      data = train_data, 
                                                      labels = train_labels,
                                                      min_g=0.001, 
                                                      max_g=0.015, 
                                                      min_c=0.01, 
                                                      max_c=10, 
                                                      steps=15, 
                                                      k=K, 
                                                      n_pca=pca_dim)

            pca_train_data = pca_calc.transform(train_data)
            rbf_svm = SVC(C=best_c, kernel="rbf", \
                            shrinking=False, gamma=best_g)\
                                        .fit(pca_train_data, train_labels)
            print "pca dim: ", pca_dim
            print "\tcv param: ", best_c, " ", best_g
            print "\tcv train err:", cv_err


            pca_test_data = pca_calc.transform(test_data)
            train_err =  1-rbf_svm.score(pca_train_data, \
                                                    train_labels)
            test_err =  1-rbf_svm.score(pca_test_data, test_labels)
            k_cv_errs.append(cv_err) 
            k_train_errs.append(train_err) 
            k_test_errs.append(test_err) 
            print "\ttrain err: ", train_err
            print "\ttest err: ", test_err 


        fig = plt.figure(num=None, figsize=(6, 10), dpi=80)
        plt.title("Errors as a function of PCA dimensions")

        fig = plt.subplot(3,1,1)
        fig.set_title("CV error")
        plt.plot(pca_dims, k_cv_errs, color='b')
        plt.xlabel("PCA dimensions")
        plt.ylabel("Error (%)")

        fig = plt.subplot(3,1,2)
        fig.set_title("train error")
        plt.plot(pca_dims, k_train_errs, color='r')
        plt.xlabel("PCA dimensions")
        plt.ylabel("Error (%)")

        fig = plt.subplot(3,1,3)
        fig.set_title("test error")
        plt.plot(pca_dims, k_test_errs, color='g')
        plt.xlabel("PCA dimensions")
        plt.ylabel("Error (%)")
        
        plt.tight_layout()
        plt.savefig("4b_pca")
 
    if part_c:
        print "------- Part c -------"
        print "importing data"
        train_data, pca_train_labels,\
                    train_1, train_rest = get_digits("data/zip.train", \
                                                [1], [2,3,4,5,6,7,8,9,0], True)
        test_data, pca_test_labels = get_digits("data/zip.test", \
                                                [1], [2,3,4,5,6,7,8,9,0])
        pca_calc =  PCA(n_components = 2).fit(train_data)
        pca_train_data = pca_calc.transform(train_data)
        pca_1 = pca_calc.transform(train_1)
        pca_rest = pca_calc.transform(train_rest)
        pca_test_data = pca_calc.transform(test_data)
        
        feat_train_data, feat_train_labels, \
                feat_1, feat_rest = get_digits("data/features.train",\
                                                [1], [2,3,4,5,6,7,8,9,0], True)
        feat_test_data, feat_test_labels\
                = get_digits("data/features.test", \
                                                [1], [2,3,4,5,6,7,8,9,0])

        print "Cross validating gamma and C for PCA data"
        pca_best_c, pca_best_g, pca_best_err, _, _ = rbf_cross_val(
                                                  data = pca_train_data, 
                                                  labels = pca_train_labels,
                                                  min_g=0.005, 
                                                  max_g=0.015, 
                                                  min_c=0.01, 
                                                  max_c=10, 
                                                  steps=20, 
                                                  k=10, 
                                                  n_pca=0)
        print "Cross validating gamma and C for synthetic data"
        feat_best_c, feat_best_g, feat_best_err,_,_ = rbf_cross_val(
                                                  data = feat_train_data, 
                                                  labels = feat_train_labels,
                                                  min_g=0.005, 
                                                  max_g=0.015, 
                                                  min_c=0.01, 
                                                  max_c=10, 
                                                  steps=20, 
                                                  k=10, 
                                                  n_pca=0)

        pca_rbf_svm = SVC(C=pca_best_c, kernel="rbf", \
                        shrinking=False, gamma=pca_best_g)\
                                    .fit(pca_train_data, pca_train_labels)
        pca_train_err =  1-pca_rbf_svm.score(pca_train_data, \
                                                pca_train_labels)
        pca_test_err =  1-pca_rbf_svm.score(pca_test_data, pca_test_labels)
        print "pca data:"
        print "\tcv best param: ", pca_best_c, " ", pca_best_g
        print "\tcv err:", pca_best_err
        print "\ttrain err:", pca_train_err
        print "\ttest err:", pca_test_err

        feat_rbf_svm = SVC(C=feat_best_c, kernel="rbf", \
                        shrinking=False, gamma=feat_best_g)\
                                    .fit(feat_train_data, feat_train_labels)
        feat_train_err =  1-feat_rbf_svm.score(feat_train_data, \
                                                feat_train_labels)
        feat_test_err =  1-feat_rbf_svm.score(feat_test_data, feat_test_labels)
        print "feat data:"
        print "\tcv best param: ", feat_best_c, " ", feat_best_g
        print "\tcv err:", feat_best_err
        print "\ttrain err:", feat_train_err
        print "\ttest err:", feat_test_err
	
        print "Find contour of classifier and plot"
    	x_min = feat_train_data[:, 0].min() - 0.1 
        x_max = feat_train_data[:, 0].max() + 0.1
        y_max = feat_train_data[:, 1].max() + 0.1 
        y_min = feat_train_data[:, 1].min() - 0.1
        points_x, points_y = np.meshgrid(np.linspace(x_min, x_max, 1000),\
                    np.linspace(y_min, y_max, 1000))
        Z_feat = feat_rbf_svm.predict(np.c_[points_x.ravel(),\
                                            points_y.ravel()])
        Z_feat = Z_feat.reshape(points_x.shape)


        fig = plt.figure("3", figsize=(5, 8), dpi=120)
        plt.title("PCA and hand features comparison")
        fig = plt.subplot(2,1,1)
        fig.set_title("Hand features")
        plt.scatter(feat_1[:,0], feat_1[:,1], color='b', s=1, marker='o')
        plt.scatter(feat_rest[:,0], feat_rest[:,1], color='r',s=1, marker='o')
        plt.scatter(feat_rbf_svm.support_vectors_[:,0],\
                    feat_rbf_svm.support_vectors_[:,1],\
                    edgecolors='k', facecolors='none',linewidth="0.3", s=4, marker='o')
        plt.contour(points_x, points_y, Z_feat)


	x_min = pca_train_data[:, 0].min() - 0.1 
        x_max = pca_train_data[:, 0].max() + 0.1
        y_max = pca_train_data[:, 1].max() + 0.1 
        y_min = pca_train_data[:, 1].min() - 0.1
        points_x, points_y = np.meshgrid(np.linspace(x_min, x_max, 1000),\
                    np.linspace(y_min, y_max, 1000))
        Z_pca = pca_rbf_svm.predict(np.c_[  points_x.ravel(),\
                                            points_y.ravel()])
        Z_pca = Z_pca.reshape(points_x.shape)
        fig = plt.subplot(2,1,2)
        fig.set_title("PCA")
        plt.scatter(pca_1[:,0], pca_1[:,1], color='b',s=1, marker='o')
        plt.scatter(pca_rest[:,0], pca_rest[:,1], color='r',s=1, marker='o')
        plt.contour(points_x, points_y, Z_pca)
        plt.scatter(pca_rbf_svm.support_vectors_[:,0],\
                    pca_rbf_svm.support_vectors_[:,1],\
                    edgecolors='k', linewidth="0.3", \
                    facecolors='none', s=4, marker='o')
        plt.tight_layout()
        plt.savefig("4c_comparison.eps",format="eps")
         
problem4()

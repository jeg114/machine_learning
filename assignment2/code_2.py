"""
Assignment2.py: Code to solve Problem 5 (Structural risk minimization)
Jesus E. Garcia
Machine Learning 
20 - 02 - 2017
"""

from matplotlib import pyplot as plt
import numpy as np
import math

def f_x(x):
    return x*(x-1)*(x-2)

#Generate a data ser of n samples such that
#the class is 1 if the second dimension is
#greater than a function of the first. Introduce
#errors with probability p_err
def dataset(n, p_err, f, x1_min,  x1_max, x2_min, x2_max):
    data = [[np.random.uniform(x1_min, x1_max), \
            np.random.uniform(x2_min, x2_max)] for i in range(n)]
    error = [True if np.random.uniform(0,1) > 1 - p_err else False \
                                            for i in range(n)]
    labels = [(1 if not err else -1) if x[1] >= f_x(x[0]) \
                                     else ( -1 if not err else 1) \
                                     for err, x in zip(error, data)]
    return data, labels

def split(data, labels):
    x_dot = [d for d, l in zip(data, labels) if l == -1]
    x_cross = [d for d, l in zip(data, labels) if l == 1]
    return x_dot, x_cross

def rm_dim0(data):
    return [i[1:] for i in data]

def add_dim0(data):
    return [[1]+i for i in data]

def transform_polyn(x1, x2, p):
    return [x1**p_i for p_i in range(p+1)] + [x2]

def get_label(x, w):
    return np.sign(sum ([x_d*w_d for (x_d, w_d) in zip(x, w)]))

def get_labels(x_list, w):
    labels = []
    for i in range(len(x_list)):
    	labels.append(get_label(x_list[i], w))
    return labels

#Run perceptron to a limit of iter_lim iterations without
#change in the minimum error
def run_perceptron(x_list, y_list, iter_lim):
    iters = 0
    iters_same = 0
    end = False
    w = [0 for dim in x_list[0]]
    w_best = None
    best_err = 1
    while iters_same < iter_lim and not end:
        iters += 1
        iters_same += 1
        end = True
    	#Search in all x for a missclassified point -> sign(w x) != y
    	for i in range(len(x_list)):
    	    #Update weights if point missclassified
    	    if get_label(x_list[i], w) != y_list[i]:
    	        w = [w_d + x_d*y_list[i] for (x_d,w_d) in zip(x_list[i], w)]
                end = False
        t_err = find_err(x_list, y_list, w)
        if t_err < best_err:
            w_best = w
            best_err = t_err
            iters_same=0
    return w_best, best_err

def find_err(x_list, y_list, w):
    err = 0
    for i in range(len(x_list)):
        if get_label(x_list[i], w) != y_list[i]:
            err += 1
    return float(err)/float(len(x_list))

def complexity(n, d_vc, w_delta):
    return math.sqrt(8*d_vc*math.log(math.e*n*2/d_vc)/n
                    + 8*math.log(4/w_delta)/n)

#Graph for the 16 dichotomies of a rectangle separating class
a_b = [(0.5,1.5,-0.5,0.5),\
       (-0.5,0.5,0.5,1.5),\
       (1.5,2.5,0.5,1.5),\
       (0.5,1.5,1.5,2.5),\
       (0,2,0.5,1.5),\
       (0.5,1.5,0,2),\
       (0,1,0,1),\
       (1,2,0,1),\
       (0,1,1,2),\
       (1,2,1,2),\
       (0,1,0,2),\
       (1,2,0,2),\
       (0,2,1,2),\
       (0,2,0,1),\
       (0,2,0,2),\
       (1.2,1.8,1.2,1.8)]

plt.figure()
plt.suptitle("16 Possible dichotomies for axis aligned rectangle classes")
subplt = 0
for (a1,a2,b1,b2) in a_b:
    subplt += 1
    plt.subplot(4,4,subplt)
    plt.scatter([1, 0, 2, 1],[0,1,1,2])
    plt.plot([a1,a2,a2,a1,a1], [b1,b1,b2,b2,b1], color='r')
plt.savefig("dichotomies")

n = 100
x1_min = 0
x1_max = 2.5 
x2_min = -1
x2_max = 2

#Use 1000 new points to estimate the test error
test_x, test_label = dataset(1000, 0.1, f_x, x1_min, x1_max, x2_min, x2_max) 

subplt = 0

plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
x = np.linspace(x1_min, x1_max, 256) 
y = np.linspace(x2_min, x2_max, 256) 
X, Y = np.meshgrid(x, y)
Z_c = np.where(Y>=f_x(X), 1, 0)

experiment = {}

n_s = [10, 100, 10000]
p_s = range(5)
exp_runs = range(10)

#Experiments running Structural Risk Minimization
#Since the experiments are random repeat exp_runs times
for i in exp_runs:
    print "Run: " + str(i)
    for n in n_s:
        #Generate the train n samples
        train_x, train_label = dataset(n, 0.1, f_x, x1_min, x1_max, x2_min, x2_max)

        #Save the samples for plotting
        x_dot, x_cross = split(train_x, train_label)
        experiment[0,n,0,"x_dot"]=x_dot
        experiment[0,n,0,"x_cross"]=x_cross
        
        best_err_bound = float("inf") 
        best_err_bound_c01 = float("inf") 
        best_err_bound_c001 = float("inf") 
        for p in p_s:
            print "\t" + str(n) + " points with polyn " + str(p)

            #Transform data to higher dimension space
            test_x_trans = [transform_polyn(x[0], x[1], p) for x in test_x]
            train_x_trans = [transform_polyn(x[0], x[1], p) for x in train_x]

            w_out, train_err = run_perceptron(train_x_trans, train_label, 850)

            test_err = find_err(test_x_trans, test_label, w_out)
            experiment[i,n,p,"train_err"] = train_err
            experiment[i,n,p,"test_err"] = test_err
            
            #Update SRM
            if i == 0:
                experiment[i,n,p,"w"]=w_out
            cmplx = complexity(n, p+1, .1/len(p_s))
            bound = train_err + cmplx
            bound_c01 = train_err + 0.1*cmplx
            bound_c001 = train_err + 0.01*cmplx
            if bound < best_err_bound:
                best_err_bound = bound
                experiment[i,n,0,"chosen"] = p 
                experiment[i,n,0,"cmplx"] = cmplx 
            if bound_c01 < best_err_bound_c01:
                best_err_bound_c01 = bound_c01
                experiment[i,n,0,"chosen_c01"] = p 
                experiment[i,n,0,"cmplx_c01"] = cmplx 
            if bound_c001 < best_err_bound_c001:
                best_err_bound_c001 = bound_c001
                experiment[i,n,0,"chosen_c001"] = p 
                experiment[i,n,0,"cmplx_c001"] = cmplx 
       

plt.figure(num=None, figsize=(20, 12), dpi=80)
plt.suptitle("Results of SRM under different parameters")
for n in n_s:
    for p in p_s:
        try:
            subplt += 1
            train_err = 100*sum([val for key, val in experiment.items() \
                                        if key[1] == n\
                                        and key[2] == p\
                                        and key[3] == "train_err"])/len(exp_runs)
            test_err = 100*sum([val for key, val in experiment.items() \
                                        if key[1] == n\
                                        and key[2] == p\
                                        and key[3] == "test_err"])/len(exp_runs)
            chosen = 100*sum([1 for key, val in experiment.items() \
                                        if key[1] == n\
                                        and val == p\
                                        and key[3] == "chosen"])/len(exp_runs)
            chosen_c01 = 100*sum([1 for key, val in experiment.items() \
                                        if key[1] == n\
                                        and val == p\
                                        and key[3] == "chosen_c01"])/len(exp_runs)
            chosen_c001 = 100*sum([1 for key, val in experiment.items() \
                                        if key[1] == n\
                                        and val == p\
                                        and key[3] == "chosen_c001"])/len(exp_runs)
            w_out = experiment[0, n, p,"w"]
            x_dot = experiment[0,n,0,"x_dot"]        
            x_cross = experiment[0,n,0,"x_cross"]
            plt.subplot(3, 5, subplt)
            Z = get_label(transform_polyn(X,Y,p), w_out)
            Z = np.where(Z == -1 , 1, 0)
            plt.pcolormesh(X, Y, Z, cmap = "RdBu")
        except ValueError:
            pass
        if subplt == 1:
            plt.plot([-5,-5],[-4, -4], color = 'r', label="Real f(x)")
            plt.plot([-5,-5],[-4, -4], color = 'y', label="Polynomial approximation")
            plt.legend(loc="upper left")
        plt.contour(X, Y, Z_c, colors = 'r')
        plt.contour(X, Y, Z, colors = 'y')
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.scatter(*zip(*x_dot), marker='o')
        plt.scatter(*zip(*x_cross), marker='x')
        plt.annotate('Train err: ' +  str(train_err) + 
                     '%  Chosen: '+ str(chosen) + 
                     "% " + str(chosen_c01)+ 
                     "% " + str(chosen_c001)+
                     '%\nTest err:' + str(test_err) + 
                     "%          c = 1   0.1   0.01", (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

plt.savefig("Train " + str(p) + " output")

for n in n_s:
    n_train_err = 0
    n_train_err_c01 = 0
    n_train_err_c001 = 0
    n_test_err = 0
    n_test_err_c01 = 0
    n_test_err_c001 = 0
    for i in exp_runs:
        for p in p_s:
            if experiment[i,n,0,"chosen"] == p:
                n_train_err += experiment[i,n,p,"train_err"]
                n_test_err += experiment[i,n,p,"test_err"]
            if experiment[i,n,0,"chosen_c01"] == p:
                n_train_err_c01 += experiment[i,n,p,"train_err"]
                n_test_err_c01 += experiment[i,n,p,"test_err"]
            if experiment[i,n,0,"chosen_c001"] == p:
                n_train_err_c001 += experiment[i,n,p,"train_err"]
                n_test_err_c001 += experiment[i,n,p,"test_err"]
    n_train_err /= len(exp_runs)
    n_train_err_c01 /= len(exp_runs)
    n_train_err_c001  /= len(exp_runs)
    n_test_err  /= len(exp_runs)
    n_test_err_c01  /= len(exp_runs)
    n_test_err_c001  /= len(exp_runs)
    print "For " + str(n) + " points:\n\twith c=1\n\t\tTrain err: " + str(n_train_err) + "\n\t\tTest err:" + str(n_test_err)
    print "\twith c=0.01\n\t\tTrain err: " + str(n_train_err_c01) + "\n\t\tTest err:"+ str(n_test_err_c01)
    print "\twith c=0.01\n\t\tTrain err: " + str(n_train_err_c001) + "\n\t\tTest err:"+ str(n_test_err_c001)

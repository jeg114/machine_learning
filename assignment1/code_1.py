"""
Assignment1.py: Code to solve Problem 3 (Experiments with the Perceptron Learning Algorithm
Jesus E. Garcia
Machine Learning 
28 - 01 - 2017
"""

"""
A code breakup is provided to aid in the readibility of the relevan parts of code:

	-- Tools to simplify code (line 25) :
		Small self explanatory functions
	-- Perceptron algorithm (line 153)
	-- Error Calculating algorithm (line 243)
	-- Functions to solve Problem 3:
		--Problem 3a (line 508)
		--Problem 3b (line 549)
		--Problem 3c (line 768)
"""

import numpy as np
from matplotlib import pyplot as plt
import math

######## Tools to simplify code  ############
""" 
Functions not commented are self-explanatory
"""

def classify(data, condition):
    return [1 if condition(i) else -1 for i in data]

def split(data, condition):
    type1 = [i for i in data if condition(i)]
    type2 = [i for i in data if not condition(i)]
    return (type1, type2)

def rm_dim0(data):
    return [i[1:] for i in data]

#Generate a defined number of samples which can be of either class
#as defined by the input condition function. Includes 0 dimension
def gen_full_data(size, sample, condition):
    points = []
    classif = []
    for i in range(size):
        tmp = sample()
        while condition(tmp) == None:
            tmp = sample()
        points.append([1]+list(tmp))
        classif.append(1 if condition(tmp) else -1)
    return (points, classif)

def gen_data(size, sample, condition):
    data = []
    sample_n = 0
    while sample_n != size:
        tmp = sample()
        if condition(tmp):
            data.append([1] +list(tmp))
            sample_n += 1
    return data

#The following gen_* functions return a function to
#be provided for other functions (First-class functions)
#The returned funcion provides a classification, which can
#be based on a line, a dictionary, a line with a margin...

def gen_cond_known(x_vals, y_vals):
    tuples_x = [tuple(x) for x in x_vals]
    def cond_known(x_to_eval):
        class_data = dict(zip(tuples_x, y_vals))
        return True if class_data[tuple(x_to_eval)] == 1 else False
    return cond_known

def gen_line(m, c, above=True):
    def condition((x, y)):
        if above:
            res = True if y >= m*x + c else False
        else: 
            res = True if y <= m*x + c else False
        return res
    return condition

def gen_margin(m, c, gamma, above=True):
    def condition((x, y)):

        if m*x + c + gamma >= y >= m*x + c - gamma:
            return None
        else:
            if above == True:
                res = True if y >= m*x + c + gamma else False
            else: 
                res = True if y <= m*x + c - gamma else False
            return res
    return condition

def gen_w_classifier(w, above=True):
    def w_classifier(data):
        if sum([w[i]*data[i] for i in range(len(w)) if i>0 ]) > -w[0]: 
            return True if above else False 
        else : 
            return False if above else True
    return w_classifier

def gen_uniform(x_max, y_max, x_min=0, y_min=0):
    def uniform():
        return (np.random.uniform(x_min, x_max), 
                        np.random.uniform(y_min, y_max))
    return uniform

def plane_to_line(plane):
    return (-plane[1]/plane[2], -plane[0]/plane[2])

def data_by_dim(data):
    dim_lists = []
    for d in range(len(data[0])):
        dim=[]
        for i in data:
            dim.append(i[d])
        dim_lists.append(dim)
    return tuple(dim_lists)

def read_file(name):
    with open(name) as f:
        file_lines = f.readlines()
        (digits, data) = zip(* \
                #-1 for 2 digits 1 for 6 digits
                [(-1 if float(line.split()[0])==2.0 else 1 , 
                [1]+[float(val) for val in line.split()[1:]]) \
                        for line in file_lines])
    return (digits, data)


def plot_w(w, color='b', x_max=1):
    m, c = plane_to_line(w)
    plot_line(m, c, color, x_max)

def plot_line(m, c, color='b',x_max=1):
    plt.plot([0, x_max], [c, x_max*m+c], color)
        
#The optimal linar regression weights are the dot product
#of the Moore-Penrose pseudoinvers of X with Y (the vector
#that defines the classes of X as 1 or -1
def opt_regres_w(data, classes):
    x_plus = np.linalg.pinv(data)
    w_opt = np.dot(x_plus, classes)
    return w_opt

def modulus(vect):
    return math.sqrt(sum([d**2 for d in vect]))

######### Perceptron Algorithm  #############

#Single step of the perceptron returns (end:bool, new_w:list)
def step_perceptron(w, x_list, y_list):

    #Search in all x for a missclassified point -> sign(w x) != y
    for i in range(len(x_list)):

        w_dot_x = sum ([x_t*w_t for (x_t, w_t) in zip(x_list[i], w)])

        #Update weights if point missclassified
        if np.sign(w_dot_x) != y_list[i]:
            new_w = [w_d + x_d*y_list[i] for (x_d,w_d) in zip(x_list[i], w)]
            return (False, new_w)

    #If all points explored and none misclassified algorithm ends
    return (True, w)

def run_perceptron(x, y):
    end = False
    w = [0 for i in range(len(x[0]))]
    while not end:
        (end, w) = step_perceptron(w, x, y)
    return w

#This modified version of the initial perceptron is used for 
#data which might not be linearly separable. It therefore saves
#the current best solution and only steps the perceptron up to
#iter_lim. Optionally initial weights can be provided, such as 
#the optimal linear regression. Returns error histogram of both 
#error on best saved weights and on the current weights. Also
#provides historgram of errors in the input test data
def modified_perceptron(x_list, y_list, test_data, test_class, iter_lim,  w=None, return_best=True):
        if w == None:
            w_step = [0 for d in range(len(x_list[0]))]
        else:
            w_step = w 

        #Initialised histograms to save
        err_hist = []
        test_err_hist = []
        old_percep_err = []

        #Initialise first error value with initial weights
	w_best = w_step 
        w_cond =  gen_w_classifier(w_step)
        w_test =  gen_w_classifier(w_best)
        best_error = sum([1 if predict!=actual else 0 \
                            for (predict, actual) in \
                                zip(classify(x_list, w_cond), y_list)])\
			/float(len(x_list))
        test_error = sum([1 if predict!=actual else 0 \
                            for (predict, actual) in \
                                zip(classify(test_data, w_cond), test_class)])\
			/float(len(test_data))
	err_hist.append(best_error)
	test_err_hist.append(test_error)
	old_percep_err.append(best_error)
        
        #Iterate only until limit
        for i in range(iter_lim):

           (end, w_step) = step_perceptron(w_step, x_list, y_list)       

           #Update errors
           w_cond =  gen_w_classifier(w_step)
           w_test =  gen_w_classifier(w_best)
           error = sum([1 if predict != actual else 0 \
                                for (predict, actual) in \
                                    zip(classify(x_list, w_cond), y_list)])\
			/float(len(x_list))
           test_error = sum([1 if predict!=actual else 0 \
                            for (predict, actual) in \
                                zip(classify(test_data, w_test), test_class)])\
			/float(len(test_data))
           if best_error > error:
               best_error = error
               w_best = w_step

	   err_hist.append(best_error)
	   test_err_hist.append(test_error)
	   old_percep_err.append(error)
            
           #If perceptron has finished train data is linearly separable
           if end:
               print "Linearly separable!"
               break
         
        return (w_best, err_hist, old_percep_err, test_err_hist)

############# Error Algorithm  ############
"""
    This is probably the most dense part of the code
    and the style is not great since it was rewritten 
    mutliple times. Hence here is a short description 
    of how it works overall:

    We aim to find the area between two lines in a defined
    square space. The approach will be to find the polygons
    defined by the intersection of the areas wiht the square
    and between themselves. This polygon might include one or
    more polygons. If for example a line intersects the square
    from left to top and the other line from bottom to right,
    we wish to include the bottom left and top right corners:

    -----*-------*  
    |   /        |
    |  /         *  We Would like to include all the * in the 
    | /         /|  polygon to find the area between the lines
    |/         / |
    *         /  |  It gets a bit more complicated when the lines
    |--------*---|  intersect. I won't try to draw that here :)

    The approach to finding then the points to include is the following:
        - Find all the corners left of one line 
        - Find all corners right of the other line
        - Intersect this set two find the common corners
        - Repeat again doing the opposite, first right then left
        - If the lines intersect in the middle, need to separate the points
          into two polygons:
            - Find all points that are connected, eg a intersection on the top
              connects with the top right corner and this with an intersection 
              in the right
            - The remaining points form the other polygon
"""
def error_algorithm(m_1, c_1, m_2, c_2,
                    gamma = 0,
                    below = True, above = True, 
                    x_min=0, x_max=1, y_min=0, y_max=1):

    #For the case of the line with margin, repeat the algorithm twice,
    #providing first the top line and keeping only the top area, and
    #doing the equivalent with the bottom line
    if gamma != 0:
        err = error_algorithm(m_1, c_1 + gamma, m_2, c_2, 0,\
                                False, True, x_min, x_max, y_min, y_max)
        err += error_algorithm(m_1, c_1 - gamma, m_2, c_2, 0,\
                                True, False, x_min, x_max, y_min, y_max)
        return err
    
    #Find all intersection points
    intersect_y=(c_1*m_2 - c_2*m_1)/(m_2-m_1) if m_2 != m_1 else y_min-0.1
    intersect_x=(c_1 - c_2)/(m_2-m_1) if m_2 != m_1 else x_min-0.1
    b_1 = (y_min - c_1)/m_1 if m_1 != 0 else x_min-0.1
    b_2 = (y_min - c_2)/m_2  if m_2 != 0 else x_min-0.1
 

    t_1 = (y_max - c_1)/m_1 if m_1 != 0 else x_min-0.1
    t_2 = (y_max - c_2)/m_2 if m_2 != 0 else x_min-0.1

    r_1 = (m_1*x_max + c_1)
    r_2 = (m_2*x_max + c_2)

    l_1 = (m_1*x_min + c_1)
    l_2 = (m_2*x_min + c_2)
    
    #Intersection points are only included within the limits
    lim_int = {}
    if x_min < t_1 <= x_max:
        lim_int['t', True]=(t_1, y_max)
    if x_min < t_2 <= x_max:
        lim_int['t', False]=(t_2, y_max)
    if x_min < b_1 <= x_max:
        lim_int['b', True]=(b_1, y_min)
    if x_min < b_2 <= x_max:
        lim_int['b', False]=(b_2, y_min)
    if y_min <= r_1 < y_max:
        lim_int['r', True]=(x_max, r_1)
    if y_min <= r_2 < y_max:
        lim_int['r', False]=(x_max, r_2)
    if y_min <= l_1 < y_max:
        lim_int['l', True]=(x_min, l_1)
    if y_min <= l_2 < y_max:
        lim_int['l', False]=(x_min, l_2)

    corners_val = {('tl'):(x_min, y_max),
               ('tr'):(x_max, y_max),
               ('br'):(x_max, y_min),
               ('bl'):(x_min, y_min,)}
    
    #This defines what corners lay left/right 
    #or top/bottom (a/b) of a line given what
    #edges it intersects.
    #Ex: A line which crosses from left to right
    #has the top left and top right corners to its
    #left/top depending on its slope. Everything assumes
    #positive slopes, but if they are negative it will still
    #work because it will be the each set will be negated,
    #and both a and b are tried for each line
    corners_to_add_tmp = {('l', 'r', 'a'):['tl', 'tr'],
                          ('l', 'r', 'b'):['bl', 'br'],
                          ('l', 't', 'a'):['tl'],
                          ('l', 't', 'b'):['tr', 'bl','br'],
                          ('l', 'b', 'a'):['bl'],
                          ('l', 'b', 'b'):['tr', 'tl','br'],
                          ('b', 'r', 'a'):['bl', 'tl', 'tr'],
                          ('b', 'r', 'b'):['br'],
                          ('b', 't', 'a'):['tl', 'bl'],
                          ('b', 't', 'b'):['tr', 'br']}

    missing = {}
    for (o,p, ab), value in corners_to_add_tmp.items():
        missing[p, o, ab] = value 

    corners_to_add = corners_to_add_tmp.copy()
    corners_to_add.update(missing)
    
    #The error is the percentage of area,
    #hence we use the total area to normalize
    area = ((x_max - x_min)*(y_max-y_min))
    
    all_points = []
    corners_added = set()
    axis_int = [key[0] for key, val in lim_int.items()]

    #Look for the intersection of corners to include by 
    #checking the left and right of each line and then
    #the other way around
    for ab in ['a', 'b']:
        
        (type1, type2) = [key[0] for key, val in lim_int.items() if key[1]==True]
        corners_lin1 = corners_to_add[type1, type2, ab]
       

        (type1, type2) = [key[0] for key, val in lim_int.items() if key[1]==False]
        corners_lin2 = corners_to_add[type1, type2, 'a' if ab == 'b' else 'b']

        #Union the current intersection with the already found corners
        corners_added |= set(corners_lin1) & set(corners_lin2)

    all_points += list(corners_added)+ axis_int
    err = 0

    #If the lines intersect in the square
    if (x_min < intersect_x < x_max and y_min < intersect_y < y_max):
        #Define 2 polygons which include the intersection
        intersect_p = [(intersect_x, intersect_y)]
        (p_set1, p_set2) = connected(all_points,lim_int, corners_val)
    else:
        #Define a single polygon
        p_set1 = connected(all_points,lim_int, corners_val, connect_all=True) 
        p_set2 = []
        intersect_p = []
     
    #Filter all the area above or below line 1
    #To be able to calculate the individual areas of
    #the two polygons. This is to implement the error
    #for the margin case
    above_line = gen_line(m_1, c_1, above=True)
    below_line = gen_line(m_1, c_1, above=False)
    keep = lambda p: above_line(p) & above | below_line(p) & below

    filt_p_set1 =  p_set1 + intersect_p if all(keep(p) for p in p_set1) else []
    x1, y1 =  zip(*filt_p_set1) if filt_p_set1 else ([],[])

    filt_p_set2 =  p_set2 + intersect_p if all(keep(p) for p in p_set2) else []
    x2, y2 =  zip(*filt_p_set2) if filt_p_set2 else ([],[])
    
    err += PolyArea(list(x1), list(y1)) if x1 else 0
    err += PolyArea(list(x2), list(y2)) if x2 else 0
    return err / float(area)

def point_from_int_dict(axis, intersects_dict, second=False):
    if second:
        return [intersects_dict[axis, False]]
    else:
        return [intersects_dict[axis, True] if (axis, True) \
                        in intersects_dict  else intersects_dict[axis, False]]

#Split the given points into polygons which have connected verteces and
#return the actual coordinate value of the points. 
#
#It essentially takes care of the different types that have been used
#for defining corners and brings everything to coordinates. It performs
#the splitting by taking into account all possible cases of point arrangements
def connected(list_in, intersects_dict, corners_dict, connect_all=False):
    axis_names=['l', 'r','t', 'b']
    p_set1=[]
    #Split into 2 only if requires (if no intersection of 
    #lines it is not required
    if not connect_all:
        #Start with any edge of the square
        for i in axis_names:
            if i in list_in:
                list_in.remove(i)
                start = i
                break
        #If there is another of sudge edge intersections it
        #forms a triangle with the intersect and we are done
        if start in list_in:
            list_in.remove(start)
            p_set1 += [intersects_dict[start, True],intersects_dict[start, False]]
        else:   
            #Have to look for a corner that connects
            #the edge with another
            p_set1 += point_from_int_dict(start, intersects_dict) 
            axis_names.remove(start)
            for axis in axis_names:
                #Corner names are only defined in one order so try both
                comp = lambda a, s: a+s if a+s in ['tl', 'tr', 'br', 'bl'] else s+a
                corner = comp(axis, start)

                #If found the corner, need to connect to edge
                if corner in list_in:
                    list_in.remove(corner)
                    p_set1.append(corners_dict[corner])
                    if axis in list_in:
                        #Formed a 4 point polygon with 2 edges,
                        #the intersection and a corner
                        list_in.remove(axis)
                        p_set1 += point_from_int_dict(axis, intersects_dict)
                    else:
                        #Must be a 5 point polygon with 2 edges,
                        #the intersection and 2 corners
                        axis_names.remove(start)
                        axis_names.remove(corner)
                        for axis2 in axis_names:
                            corner2 = comp(axis, axis2)
                            if corner in list_in:
                                list_in.remove(corner2)
                                p_set1.append(corners_dict[corner2])


    #All remaining points are in the other polygon
    p_set2 = []
    for i in set(list_in):
        if i in axis_names:
            if (i, True) in intersects_dict and (i, False) in intersects_dict:
                p_set2 += point_from_int_dict(i, intersects_dict)
                p_set2 += point_from_int_dict(i, intersects_dict, True)
            p_set1 += point_from_int_dict(i, intersects_dict)
        else:
            p_set2.append(corners_dict[i])
    return p_set2 if connect_all else (p_set1, p_set2)

#Calculate the area of a polygon using its vertices.
#Source: http://stackoverflow.com/questions/24467972/calculate-
#area-of-polygon-given-x-y-coordinates
def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

############ Auxiliary Graphs for report ##############
def auxiliary_graphs():
    print "Printing auxiliary graphs for report"
    print "------------------"
    plt.figure("Area to find")
    m1 = 0.4
    m2 = 0.5
    c1 = 0.5
    c2 = 0.2
    plt.plot([0, 1], [c1, m1+c1])
    plt.plot([0, 1], [c2, m2+c2])
    plt.savefig("area_to_find")


############ problem 3a ##############
def problem_3a():
    print "Problem 3a: Running perceptron algorithm"
    print "------------------"
    plt.figure('p3a')
    subplt = 1
    m_t = 0.2
    c_t = 0.4
    line = gen_line(m_t, c_t)

    #Running the perceptron for different sizes
    for train_n in [2, 4, 10, 100]:
        
        #Generate data over the uniform unit square
        (x, y) = gen_full_data(train_n, gen_uniform(1,1), line)
        (x_dot, x_cross)  = split(rm_dim0(x), line)

        #If there is not at least one member of each class repeat
        #this is only for a better graph
        while not(x_dot and x_cross):
            (x, y) = gen_full_data(train_n, gen_uniform(1,1), line)
            (x_dot, x_cross)  = split(rm_dim0(x), line)

        w = run_perceptron(x, y)

        #Setup the data for th graphs
        (x1_dot, x2_dot)  = data_by_dim(x_dot) 
        (x1_cross, x2_cross)  = data_by_dim(x_cross)
        
        #Print the result of the 4 perceptron runs
        plt.subplot(2, 2, subplt)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.scatter(x1_dot, x2_dot, marker='o')
        plt.scatter(x1_cross, x2_cross, marker='x')
        plot_w(w)
        plot_line(m_t, c_t, 'r')
        subplt += 1
    plt.tight_layout()
    plt.savefig("p3a") 

#######    problem_3b   ###########
#For problem 3 we will gather the following graphs
#
#   -Error for training set sizes for 4 runs
#   -Iterations taken and the theory bounds
#     for training set sizes for 4 runs
#   -Error average differences for the 3 methods
#   -Error average and confidence intervals for
#     multiple margins
#
#
def problem_3b():
    print "Problem 3b: Starting error graphs"
    print "------------------"
    (m_t, c_t) = plane_to_line([-0.1, -1, 1])
    
    all_train_n = [i*100 for i in range(1,6)]
    
    repeat_n = 100
    
    fig_name = "Gammas"
    fig = plt.figure(fig_name, figsize=(10,15))
    subplt=1

    interrupt=0
    margins = [0, 0.3, 0.1, 0.01, 0.001]
    print "Calculating margins: ", margins
    print "Margin 0 takes up to 4mins"
    print "------------------"
    for margin in margins:
	print margin
        class_line = gen_margin(m_t, c_t, margin)
        
        if margin == 0:
            test_n = 1000
            (x_test, y_test) = gen_full_data(test_n, gen_uniform(1,1), class_line)

        alg_avg = []
        alg_conf_95 = []
        alg_conf_05 = []
        
        sim_avg = []
        sim_abs_avg = []
        pre_avg = []
        pre_abs_avg = []
        
        save_runs = []
        all_Rs = []
        all_ws = []
        all_rhos = []
        all_iters = []

        subplt_runs = 0

        for train_n in all_train_n: 
            all_alg_err = []
            all_sim_err = []
            all_pre_err = []
            all_sim_diff = []
            all_pre_diff = []
            all_sim_abs_diff = []
            all_pre_abs_diff = []
            iters = []
            bounds = []
            Rs = []
            rhos = []
            ws = []
            iters = []
            for repeat in range(repeat_n): 
                
                (x, y) = gen_full_data(train_n, gen_uniform(1,1), class_line)
                (x_dot, x_cross)  = split(rm_dim0(x), class_line)
                (x1_dot, x2_dot)  = data_by_dim(x_dot)
                (x1_cross, x2_cross)  = data_by_dim(x_cross)
                
                end = False
                steps=0
                w=[0,0,0]
                while not end:
                    steps += 1
                    (end, w) = step_perceptron(w, x, y)
                    if steps > 5*train_n and repeat > 2:
                        interrupt+=1
                        end = True
               

                (m_h, c_h) = plane_to_line(w)
                
                alg_err = error_algorithm(m_h, c_h, m_t, c_t, margin)
                all_alg_err.append(alg_err)

                p = (c_h - c_t)/(m_t - m_h)
                pre_err = abs((m_h - m_t)*(2*p**2 -1)/2 + (c_h - c_t)*(2*p-1))
                
                
                y_out = classify(rm_dim0(x_test), gen_margin(m_h, c_h, margin))
                sim_err = sum([1 if i!=j else 0 for (i, j) \
                                in zip(y_test, y_out)])/float(test_n)

                if margin == 0:
                    all_pre_diff.append(pre_err - alg_err)
                    all_sim_diff.append(sim_err - alg_err)
                    all_pre_abs_diff.append(abs(pre_err - alg_err))
                    all_sim_abs_diff.append(abs(sim_err - alg_err))
                    if repeat < 3:
                        R = max([modulus(x_t) for x_t in x])
                        rho = min([y[i]*np.dot(w, x[i]) for i in range(len(x))])
                        mod_w = modulus(w)
                        bound = ((R*mod_w)/rho)**2
                        Rs.append(R)
                        rhos.append(rho)
                        ws.append(mod_w)
                        iters.append(steps)
                
            alg_avg.append(sum(all_alg_err)/float(repeat_n))
            alg_conf_95.append(sorted(all_alg_err)[94])
            alg_conf_05.append(sorted(all_alg_err)[5])
        
            if margin == 0:
                sim_avg.append(sum(all_sim_diff)/float(repeat_n))
                pre_avg.append(sum(all_pre_diff)/float(repeat_n))
                sim_abs_avg.append(sum(all_sim_abs_diff)/float(repeat_n))
                pre_abs_avg.append(sum(all_pre_abs_diff)/float(repeat_n))
                save_runs.append(all_alg_err[0:4])
                all_Rs.append(Rs)
                all_ws.append(ws)
                all_rhos.append(rhos)
                all_iters.append(iters)
        
        if margin == 0:
    	    print "Plotting "

            plt.figure("Differences between error calculating methods")
            plt.subplot(2,1,1)
            l=range(2)
            l[0], = plt.plot(all_train_n, sim_abs_avg, color='b',
                        label='Simulated avg error')
            l[1], = plt.plot(all_train_n, pre_abs_avg, color='g',
                        label='Equation avg error')
            plt.legend(l, ['Simulated avg error','Equation avg error'])
            plt.subplot(2,1,2)
            l[0], = plt.plot(all_train_n, sim_avg, color='b',
                        label='Simulated avg absolute error')
            l[1], = plt.plot(all_train_n, pre_avg, color='g', 
                        label='Equation avg absolute error')
            plt.legend(l, ['Simulated avg absolut error','Equation avg absolute error'])
            plt.tight_layout()
            plt.savefig("errs")
           
            subplt_runs = 1 
            plt.figure("gamma0runs")
            plt.title("Errors (%)  of multiple runs for size varying training")
            for i in range(4):
                plot_name = "run " + str(subplt_runs)
                ax=plt.subplot(2,2,subplt_runs)
                ax.set_title(plot_name)
                ax.scatter(all_train_n, [r[i] for r in save_runs])
                subplt_runs += 1
            plt.tight_layout()
            plt.savefig("gamma0runs")

            plt.figure("gamma0avg+conf")
            plt.title("Error average and 90% confidence interval")
            l=range(4)
            l[0], = plt.plot(all_train_n, alg_avg, color='b',
                        label='Average')
            l[1], = plt.plot(all_train_n, alg_conf_95, color='r',
                        label='90% confidence interval')
            l[2], = plt.plot(all_train_n, alg_conf_05, color='r')
            plt.fill_between(all_train_n, alg_conf_05, alg_conf_95, color='grey', alpha='0.5')
            plt.legend(l, ['Average','90% confidence interval'])
            plt.savefig("gamma0avg+conf")
            
            subplt_runs = 1
	    num_items = len(all_train_n) 
	    ind = np.arange(num_items)
	    margin = 0.01
	    width = (1.-2.*margin)/num_items 
            plt.figure("iterations and bounds")
            plt.suptitle("iterations and bounds")
            ax=plt.subplot(1,1,1)
            ax.set_yscale('log')
            tick_labels = []
            for i in range(len(all_ws[0])):
		x_data = ind+margin+(i*width)
                run_i_bounds = [((all_Rs[j][i]*all_ws[j][i])/ \
                                                                all_rhos[j][i])**2 
                                        for j in range(len(all_train_n))]
                run_i_iters = [all_iters[j][i] \
                                    for j in range(len(all_train_n))]
                plt.bar(x_data, run_i_iters, width, color='b')
                plt.bar(x_data, run_i_bounds, width, color='r',
                            bottom=run_i_iters) 
                plt.bar(x_data, run_i_iters, width, color='b') 
                plt.bar(x_data, run_i_bounds, width, color='r', 
                            bottom=run_i_iters) 
            
            plt.legend(l, ['Actual iterations','Predicted bound'])
            tick_labels = [train_n for train_n in all_train_n]
            ax.set_ylabel('Iterations and bounds (log)')
            ax.set_title('Training set size')
            ax.set_xticks(ind + 0.5)
            ax.set_xticklabels(tick_labels)
            plt.savefig("iterations and bounds")
        else:
    	    print "Plotting "
            plt.figure(fig_name)
            plot_name = "Margin: " + str(margin)
            ax=plt.subplot(len(margins) -1,1,subplt)
            ax.set_title(plot_name)
            ax.plot(all_train_n, alg_avg, color='r')
            ax.plot(all_train_n, alg_conf_95, color='r')
            ax.plot(all_train_n, alg_conf_05, color='r')
            plt.fill_between(all_train_n, alg_conf_05, alg_conf_95, color='grey', alpha='0.5')
            subplt += 1
        
        fig.savefig(fig_name, bbox_inches="tight")
    return

#######    problem_3c   ###########
def problem_3c():
    #Tool to filter data: grep -E "^[ ]+(2|8).*"

    iters = 25 
    
    (digits, data) = read_file("train_2_8_features.txt")
    (test_class, test_data) = read_file("test_2_8_features.txt")
    print "Problem 3b: Starting digit classification"
    print "Training data has: ", len(data), " samples"
    print "------------------"

    print "Running modified perceptron on 2D feature data"
    (w_out, err, old_err, test_err) = modified_perceptron(data, digits, 
							test_data, test_class, 
									iters) 
    print "Train error:", err[-1]
    print "Test error:", test_err[-1]
    print "Iterations:", len(test_err)-1
    print "------------------"
    dim_data = rm_dim0(data)
    (x_dot, x_cross)  = split(dim_data, gen_cond_known(dim_data,digits))
    (x1_dot, x2_dot)  = data_by_dim(x_dot)
    (x1_cross, x2_cross)  = data_by_dim(x_cross)
    
    dim_test = rm_dim0(test_data)
    (x_test_dot, x_test_cross)  = split(dim_test, \
					gen_cond_known(dim_test,test_class))
    (x1_test_dot, x2_test_dot)  = data_by_dim(x_test_dot)
    (x1_test_cross, x2_test_cross)  = data_by_dim(x_test_cross)
    
    plt.figure("Perceptron on features")
    l = range(4)
    plt.scatter(x1_dot, x2_dot, color='b', marker='o')
    plt.scatter(x1_cross, x2_cross, color='r', marker='x')
    plt.scatter(x1_test_dot, x2_test_dot, color='g', marker='o')
    plt.scatter(x1_test_cross, x2_test_cross,color='k', marker='x')
    plot_w(w_out, x_max=0.7) 
    plt.savefig("features_perce")

    plt.figure("Change on error on features")
    iterations = range(len(err))
    l[0], = plt.plot(iterations, err, label ='Output error')
    l[1], = plt.plot(iterations, old_err, label ='Unmodified perceptron error')
    l[2], = plt.plot(iterations, test_err, label ='Error on test data')
    plt.legend(l, ['Output error', 
                    'Unmodified perceptron error', 'Error on test data'])
    plt.savefig("error_change")

    print "Running modified perceptron on 2D feature data with initialised weights"
    w_opt = opt_regres_w(data, digits)
    (w_out, err, old_err, test_err) = modified_perceptron(data, digits, 
							test_data, test_class, 
							iters, w_opt)
    print "Train error:", err[-1]
    print "Test error:", test_err[-1]
    print "Iterations:", len(test_err)-1
    print "------------------"
    dim_data = rm_dim0(data)
    (x_dot, x_cross)  = split(dim_data, gen_cond_known(dim_data,digits))
    (x1_dot, x2_dot)  = data_by_dim(x_dot)
    (x1_cross, x2_cross)  = data_by_dim(x_cross)
    
    plt.figure("Initialied perceptron")
    plt.scatter(x1_dot, x2_dot)
    plt.scatter(x1_cross, x2_cross)
    plot_w(w_out, x_max=0.7)
    plot_w(w_opt, x_max=0.7, color='r--')
    plt.savefig("regres_perce")
   
    l=range(4)

    plt.figure("Change on error on features with opt")
    iterations = range(iters+1)
    l[0], = plt.plot(iterations, err, label ='Output error')
    l[1], = plt.plot(iterations, old_err, label ='Unmodified perceptron error')
    l[2], = plt.plot(iterations, test_err, label ='Error on test data')
    plt.legend(l, ['Output error', 
                    'Unmodified perceptron error', 'Error on test data'])
    plt.savefig("init_error_change")

    iters=10000
    (digits, data) = read_file("train_2_8_raw.txt")
    (test_class, test_data) = read_file("test_2_8_raw.txt")

    print "Running modified perceptron on raw data"
    (w_out, err, old_err, test_err) = modified_perceptron(data, digits, 
							test_data, test_class, 
									iters) 
    print "Train error:", err[-1]
    print "Test error:", test_err[-1]
    print "Iterations:", len(test_err)-1
    print "------------------"

    plt.figure("Change on error on raw")
    iterations = range(len(err))
    l[0], = plt.plot(iterations, err, label ='Output error')
    l[1], = plt.plot(iterations, old_err, label ='Unmodified perceptron error')
    l[2], = plt.plot(iterations, test_err, label ='Error on test data')
    plt.legend(l, ['Output error', 
                    'Unmodified perceptron error', 'Error on test data'])
    plt.savefig("raw_error_change")
 
    print "Running modified perceptron initialised with the optimal regression"
    w_opt = opt_regres_w(data, digits)
    (w_out, err, old_err, test_err) = modified_perceptron(data, digits, 
							test_data, test_class, 
							iters, w_opt)
    print "Train error:", err[-1]
    print "Test error:", test_err[-1]
    print "Iterations:", len(test_err)-1
    print "------------------"
    plt.figure("Change on error on raw with opt")
    iterations = range(len(err))
    l[0], = plt.plot(iterations, err, label ='Output error')
    l[1], = plt.plot(iterations, old_err, label ='Unmodified perceptron error')
    l[2], = plt.plot(iterations, test_err, label ='Error on test data')
    plt.legend(l, ['Output error', 
                    'Unmodified perceptron error', 'Error on test data'])
    plt.savefig("init_raw_error_change")

problem_3a()
problem_3b()
problem_3c()

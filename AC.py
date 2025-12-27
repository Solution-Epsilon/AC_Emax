import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import statistics


#Simulations for the E%-WTA model.

# Functions and classes:
# -random_weight(): responsible for assigning values to synaptic weights when a synapse exists;
# -overlap_matrix(assembly_list): returns an overlap matrix for a set of neural assemblies (assembly_list) within the same memory area;
# -G_density(): Returns the density of a group of neurons (graph), that is, of a neural assembly or a memory area.
# -artificial_neuron: generates an artificial neuron. Synaptic weight adjustment and synaptic integration are carried out here;
# -artificial_area: generates an artificial area, which can be either a memory area or a stimulus area using the “+” operator. 
#     What determines whether an area is a memory or a stimulus area in the code is how the “+” operator is used:
#       Memory area:
#           a) M_1 + M_1 → recurrent synapses (within the same area);
#           b) M_1 + M_2 → synapses from neurons in M_1 project to neurons in M_2. The opposite is performed with M_2 + M_1.
#       Stimulus area:
#           a) S_1 + M_1 → neurons that generate stimuli project to neurons in M_1;
#           b) S_1 + S_1 → although this operation is allowed, by definition of the model and its areas it does not correspond to a memory area, 
#                          since it is an operation between stimulus areas. The same applies to recurrent connections (S_1 + S_1).
# -FIRE(stimuli,B,T): Fires a set of neurons (stimuli) T times that project to a given memory area (B).
# -FIRE_F(stimuli,B): Fires a set of neurons (stimuli) that project to a given memory area (B) until the formation conditions are met.
# -FIRE_R(stimuli,B, assembly,T): Retrieves the formed assembly (assembly) in a memory area (B) by firing the set of neurons (stimuli) that formed it T times.
# Note: To generate a stimulus, simply use the method built into the artificial_area class, sample_random_set(k), where k is the stimulus size.

def random_weight():
    if np.random.binomial(1,.2) == 1:
        return -0.2
    return 1

def overlap_matrix(assembly_list):
    df = pd.DataFrame()
    for i in range(0,len(assembly_list)):
        list_sobreposition = []
        for j in assembly_list:
            list_sobreposition.append(len([k for k in assembly_list[i] if k in j]))
        df[str(i)] = list_sobreposition
    #print(df)
    return df

class artificial_neuron:
    
    def __init__(self,index):

        self.f = False
        self.f_1 = False
        self.index = index
        self.synaptic_input = 0
        self.pre_synaptic_list = [] #dict {neuron : weight}
        
        ####################
        self.flag = False

    # INFO:
    def __str__(self):
        self.__show_att__()
        return ""

    def __repr__(self):
        return "Neuron {0}".format(self.index)
    
    def __show_att__(self):
        for i,j in self.__dict__.items():
            print("{0}:".format(i),j)
 
    # OPERATIONS:
    def __eq__(self, other_neuron):
        return self.index == other_neuron.index
    
    def __gt__(self, other_neuron):
        return self.synaptic_input > other_neuron.synaptic_input
    
    def __add__(pre_neuron, pos_neuron):
        pos_neuron.pre_synaptic_list.append({"neuron": pre_neuron, "w" : random_weight() })

    # INTEGRATION AND LEARNING:
    def synaptic_integration(self):
        self.synaptic_input = 0
        for i in self.pre_synaptic_list:
            self.synaptic_input += i["neuron"].f_1 * i["w"]
        #self.synaptic_input = np.array([self.pre_synaptic_list[i]['neuron'].f_1 * self.pre_synaptic_list[i]['w'] for i in range(len(self.pre_synaptic_list))]).sum()
    
    def update_weights(self, beta):
        for i in self.pre_synaptic_list:
            if i['neuron'].f_1 == True:
                i['w'] *= (1 + beta)

class artificial_area:

    def __init__(self,n,p,k,beta,d,tau,e_max, index):
        
        self.n = n
        self.p = p
        self.k = k
        self.beta = beta
        self.index = index
        self.e_max = e_max
        self.e_max_constant = (1 - d/tau)
        self.neuron_list = [artificial_neuron(i + index*n) for i in range(0,n)]
        #===========#
        self.neuron_winners = None
        self.neuron_winners_1 = None

        self.recovery_list = []
        self.recovery_fire = []
    # INFO:
    def __str__(self):
        self.__show_att__()
        return ""

    def __repr__(self):
        return "Area {0}".format(self.index)
    
    def __show_att__(self):
        for i,j in self.__dict__.items():
            print("{0}:".format(i),j)

    # OPERATIONS:
    def __add__(pre_area,pos_area):
        
        #pre_area = pos_area  --> recurrence
        #pre_area != pos_area --> afference
        
        for i in pre_area.neuron_list:
            for j in pos_area.neuron_list:
                if (np.random.binomial(1,pre_area.p) == 1 and i != j):
                    i + j

    # METHODS:
    def sample_random_set(self, size):
        return rnd.sample(self.neuron_list,size)

    def winners_list(self):
        for i in self.neuron_list:
            i.synaptic_integration()
            
        if self.e_max == False:
            self.neuron_list.sort(reverse=True)
            return [self.neuron_list[i] for i in range(0,self.k)]

        elif self.e_max == True:
            inputs_area = [i.synaptic_input for i in self.neuron_list]

            return [i for i in self.neuron_list if i.synaptic_input >= self.e_max_constant * max(inputs_area)]

def G_density(graph):
    vertex = len(graph)
    if vertex > 1:
        edge = 0
        for i in graph:
            for j in i.pre_synaptic_list:
                if j["neuron"] in graph:
                    edge += 1
        density = (edge)/(vertex*(vertex - 1))
        return density
    return 0

def FIRE(stimuli,B,T):
    for i in stimuli:
        i.f_1 = True

    for t in range(1,T+1):
            
        if t > 1:
            if B.neuron_winners != None:
                for i in B.neuron_winners:
                    i.f_1 = True
                B.neuron_winners_1 = B.neuron_winners

        new = 0
        B.neuron_winners = B.winners_list()
        for i in B.neuron_winners:
            i.update_weights(B.beta)
            if i.flag == False:     # D.
                i.flag = True       # D.
                new += 1            # D.

        if t == 1:
            new_1 = len(B.neuron_winners)
        else:
            new_1 = len([i for i in B.neuron_winners if i not in B.neuron_winners_1])

        print(t,new,new_1, len(B.neuron_winners))      # D.

        if t > 1:
            for i in B.neuron_winners_1:
                i.f_1 = False


    for i in stimuli:
        i.f_1 = False

    for i in B.neuron_list:
        i.flag = False ; i.f_1 = False

    assembly = B.neuron_winners
    B.neuron_winners = None ; B.neuron_winners_1 = None

    return assembly 

def FIRE_F(stimuli,B):
    t = 1
    for i in stimuli:
        i.f_1 = True

    while True:

        if t > 1:
            if B.neuron_winners != None:
                for i in B.neuron_winners:
                    i.f_1 = True
                B.neuron_winners_1 = B.neuron_winners

        new = 0
        B.neuron_winners = B.winners_list()
        for i in B.neuron_winners:
            i.update_weights(B.beta)
            if i.flag == False:
                i.flag = True
                new += 1

        if t == 1:
            new_1 = len(B.neuron_winners)
        else:
            new_1 = len([i for i in B.neuron_winners if i not in B.neuron_winners_1])
        #print(t, new, new_1, len(B.neuron_winners))

        if t > 1:
            for i in B.neuron_winners_1:
                i.f_1 = False
        
            if B.e_max == True:
                if (new_1 == 0) and (len(B.neuron_winners) == len(B.neuron_winners_1)):
                    break
            else:
                if (new == 0):
                    break
            
        t +=1

    for i in stimuli:
        i.f_1 = False

    for i in B.neuron_list:
        i.flag = False ; i.f_1 = False

    assembly = B.neuron_winners
    B.neuron_winners = None ; B.neuron_winners_1 = None
    if G_density(assembly) > B.p and len(assembly) > 5:
        return assembly,t
    return None,None

def FIRE_R(stimuli, B, assembly, T):
    for i in stimuli:
        i.f_1 = True

    B.beta = 0

    for t in range(1,T+1):
            
        if t > 1:
            if B.neuron_winners != None:
                for i in B.neuron_winners:
                    i.f_1 = True
                B.neuron_winners_1 = B.neuron_winners

        B.neuron_winners = B.winners_list()
        for i in B.neuron_winners:
            i.update_weights(B.beta)

        print(t, len([i for i in B.neuron_winners if i in assembly]), len(B.neuron_winners))
        B.recovery_list.append(len([i for i in B.neuron_winners if i in assembly]))
        B.recovery_fire.append(len(B.neuron_winners))
        if t > 1:
            for i in B.neuron_winners_1:
                i.f_1 = False

    for i in stimuli:
        i.f_1 = False

    for i in B.neuron_list:
        i.flag = False ; i.f_1 = False
    return len([i for i in B.neuron_winners if i in assembly])


e_max = True # (False = k-winners-take-all/True = E%-winners-take-all)
if e_max == True:
    n = 1000 ; p = .5 ; k = 200 ; beta = 0.01 ; T = 15 ; d = 3 ; tau = 30 #e_max parameters (E%-wta)
else:
    n = 1000 ; p = .1 ; k = 37 ; beta = 0.01 ; T = 15 ; d = 3 ; tau = 30 #k_winners_take_all parameters (k-wta)

#Note: The parameters “d” and “tau” are required in the k-winners-take-all only for the use of the classes and do not affect the behavior of the original model.

# 1) ================
# - Failure rate: (figure 2-d)
# - Simulation to verify the failure rate in assembly formation. w_inh = -0.2, -0.4, -0.6, -0.8, -1.0 (for all other simulations, w_inh should be set to −0.2)
# - To change the value of w_inh, modify it in the random_weight function (the return statement inside the if loop)
'''
B_1 = [.1,.05,.01,.005,.001]    # <---- List of synaptic plasticities used.
error_list = []
for beta in B_1:
    error = []
    for i in range(0,100):
        print("Trial: {0}".format(i))
        fail = 0
        for j in range(0,20):
            S = artificial_area(n,p,k,beta,d,tau,e_max, 0)
            A = artificial_area(n,p,k,beta,d,tau,e_max, 1)

            S + A
            A + A

            set_test = S.sample_random_set(k)
            assembly,time_f = FIRE_F(set_test,A)
            if assembly == None:
                fail += 1
        error.append(fail/20)
    error_list.append(error)
error_dict = {'.1': error_list[0], '.05': error_list[1], '.01': error_list[2], '.005': error_list[3], '.001': error_list[4],}
df = pd.DataFrame(data=error_dict)
print(df)
df.to_csv("error_formation_02")  # <---- Saves the data to a .csv file. Note that the filename has the value "02" at the end, indicating that the value of w_inh = -0.2
                                 #       Each column of the file is associated with the value of synaptic plasticity and the first line of the file contains the column names (plasticity values).
'''

# 2) ================
# - Characteristics of the neural assemblies formed 
# - Simulations for Table 1 and for Figure 3-a.
# - For the size simulations with and without feedforward inhibition, it suffices to set w_inh = 1 (which reduces to the original model case).
# - To use different synaptic plasticity values, it is necessary to change the values of the parameter ‘beta’ presented earlier.
'''
size_list = []
density_list = []
time_list = []
df_formation = pd.DataFrame()
print("BETA: ",beta)
for i in range(0,500):

    print("Iteration {0}".format(i))
    S = artificial_area(n,p,k,beta,d,tau,e_max, 0)
    A = artificial_area(n,p,k,beta,d,tau,e_max, 1)

    S + A
    A + A

    set_test = S.sample_random_set(k)
    assembly,time_f = FIRE_F(set_test,A)

    size_list.append(len(assembly))
    density_list.append(G_density(assembly))
    time_list.append(time_f)


df_formation["size"] = size_list        # Size of assemblies formed.
df_formation["time"] = time_list        # Number of total iteration
df_formation["density"] = density_list  # Synaptic density
df_formation.to_csv("formation_0_05_k_winners")   # Generates the .csv file with the characteristics, where "0_05" represents the synaptic plasticity value 
                                                  # and "k_winners" represents the neuron selection method.
'''

# 3) ================
# - Ability to retrieve a formed assembly by firing the stimulus that generated it 15 times (T = 15, Figure 3-b):
'''
beta_list = [0.1,0.05,0.01,0.005,0.001]
beta_str = ["0_1","0_05","0_01","0_005","0_001"]
for j in range(0,len(beta_list)):
    print("Beta recovery: {0}".format(beta_list[j]))
    list_recovery = []
    i = 0
    while len(list_recovery)< 200:
        print("===ITERACAO {0}===".format(i))

        S = artificial_area(n,p,k,beta_list[j],d,tau,e_max, 0)
        A = artificial_area(n,p,k,beta_list[j],d,tau,e_max, 1)

        S + A
        A + A

        set_test = S.sample_random_set(k)
        assembly,time_f = FIRE_F(set_test,A)
        if assembly != None:
            FIRE_R(set_test,A,assembly,T)
            list_recovery.append(A.recovery_list[-1]/len(assembly))
            i += 1
    df_recovery = pd.DataFrame(list_recovery)
    df_recovery.to_csv("recovery_e_max_" + str(beta_str[j]) + "_" + str(T))
'''

# 4) ================
# - Multiple neural assemblies in the same memory area. (Figures 3-c and 3-d) 
# - Formação de várias assembleias em uma área de memória (A) para beta = 0.01.
'''
n_assemblies_in_area = 0
max_assemblies_in_area = 10
for j in range(0,100):
    while n_assemblies_in_area < max_assemblies_in_area:
        S = artificial_area(n,p,k,beta,d,tau,e_max, 0)
        A = artificial_area(n,p,k,beta,d,tau,e_max, 1)

        S + A
        A + A
        assemblies = []
        stimuli = []
        for i in range(0,max_assemblies_in_area):
            print(i)
            set_test = S.sample_random_set(k)
            assembly,time_f = FIRE_F(set_test,A)
            if assembly != None:
                stimuli.append(set_test)
                assemblies.append(assembly)
                n_assemblies_in_area += 1
                #print("Density = {0}".format(G_density(assembly)))
            #print()
        
        if len(assemblies) == max_assemblies_in_area:
            df_A = overlap_matrix(assemblies)             
            df_S = overlap_matrix(stimuli)
            df_A.to_csv("matrix_assembly_emax_{0}".format(j))   # Overlap matrix for neural assemblies and stimuli (below).
            df_S.to_csv("matrix stimuli_emax_{0}".format(j))    # "emax" represents the selection method used here.

'''

# 5) ================
# - Evaluating the effect of probability and stimulus size (figures 4-a and 4-b):
# - The formation success rate is obtained by dividing the number of assemblies formed by the total number of simulations (in our case, 200 simulations).
# a) Stimulus size:
'''
n = 1000 ; p = .5 ; k = 200 ; beta = 0.01 ; T = 15 ; d = 3 ; tau = 30
e_max = True


size_k_s = [50,100,150,250,500]
for size in size_k_s:
    size_list = []
    for i in range(0,200):
        print("Iteration: {0}".format(i))

        S = artificial_area(n,p,size,beta,d,tau,e_max, 0)
        A = artificial_area(n,p,size,beta,d,tau,e_max, 1)

        S + A
        A + A

        set_test = S.sample_random_set(size)
        assembly,time_f = FIRE_F(set_test,A)
        if assembly != None:
            size_list.append(len(assembly))
    df_size_list = pd.DataFrame(size_list)
    df_size_list.to_csv("size_stimuli_" + str(size))
'''
# b) Probability:
'''
n = 1000 ; p = .5 ; k = 200 ; beta = 0.01 ; T = 15 ; d = 3 ; tau = 30
e_max = True


prob_list = [.1,.2,.3,.4,.5]
prob_str = ["0_1","0_2","0_3","0_4","0_5"]
for j in range(0,len(prob_list)):
    p_list = []
    for i in range(0,200):
        print("Iteration: {0}".format(i))

        S = artificial_area(n,prob_list[j],k,beta,d,tau,e_max, 0)
        A = artificial_area(n,prob_list[j],k,beta,d,tau,e_max, 1)

        S + A
        A + A

        set_test = S.sample_random_set(k)
        assembly,time_f = FIRE_F(set_test,A)
        if assembly != None:
            p_list.append(len(assembly))
    df_size_list = pd.DataFrame(p_list)
    df_size_list.to_csv("prob_" + prob_str[j])
'''

# 6) ================
# -Catastrophic forgetting (Figure 4-c).
'''
max_set = 10
perc_list = []

while len(perc_list) < 50:
    j = 0
    S = artificial_area(n,p,k,beta,d,tau,e_max, 0)
    A = artificial_area(n,p,k,beta,d,tau,e_max, 1)

    S + A
    A + A
    assemblies = []
    stimuli = []
    for i in range(0,max_set):
        print(i)
        set_test = S.sample_random_set(k)
        assembly,time_f = FIRE_F(set_test,A)
        print(assembly)
        if assembly != None:
            stimuli.append(set_test)
            assemblies.append(assembly)
            print("Density = {0}".format(G_density(assembly)))
            if i == 0:
                rec_assembly = assembly
                rec_set = set_test
            j = i 
        else:
            break
    print()
    if j + 1 == max_set:
        print(len(rec_assembly))
        sized = FIRE_R(rec_set,A,rec_assembly,T)
        perc_list.append(sized/len(rec_assembly))

df_a = pd.DataFrame(perc_list)
df_a.to_csv("rec_"+ str(max_set))
'''

# 7) ================
# - Convergence of the first two conditions (Equations 8 and 9)
# - The FIRE_TEST function is defined only to visualize convergence
# - Each column represents the following:
#   1) Iteration number;
#   2) Number of new neurons (N_t), as in the original article;
#   3) New neurons relative to the previous iteration (X_t);
#   4) Number of neurons that fire (f) in the memory area (A).

'''
def FIRE_TEST(stimuli,B,T):
    for i in stimuli:
        i.f_1 = True

    for t in range(1,T+1):
            
        if t > 1:
            if B.neuron_winners != None:
                for i in B.neuron_winners:
                    i.f_1 = True
                B.neuron_winners_1 = B.neuron_winners

        new = 0
        B.neuron_winners = B.winners_list()
        for i in B.neuron_winners:
            i.update_weights(B.beta)
            if i.flag == False:     # D.
                i.flag = True       # D.
                new += 1            # D.

        if t == 1:
            new_1 = len(B.neuron_winners)
        else:
            new_1 = len([i for i in B.neuron_winners if i not in B.neuron_winners_1])

        print(t,new,new_1, len(B.neuron_winners))      # D.

        if t > 1:
            for i in B.neuron_winners_1:
                i.f_1 = False


    for i in stimuli:
        i.f_1 = False

    for i in B.neuron_list:
        i.flag = False ; i.f_1 = False

    assembly = B.neuron_winners
    B.neuron_winners = None ; B.neuron_winners_1 = None

    return assembly

S = artificial_area(n,p,k,beta,d,tau,e_max, 0)
A = artificial_area(n,p,k,beta,d,tau,e_max, 1)

S + A
A + A

set_test = S.sample_random_set(k)
FIRE_TEST(set_test,A,T)
'''

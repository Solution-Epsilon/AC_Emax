import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
import os, shutil
from statistics import median
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import Divider
import mpl_toolkits.axes_grid1.axes_size as Size
from matplotlib.cbook import get_sample_data


def filter(df):
    return df[(df["size"] > 5) & (df["density"] > .5)]

def outliers(df):
    print("n_data: {0}".format(len(df["size"])))
    Q3 = df.quantile(.75) ; Q1 = df.quantile(.25)
    IQR = Q3 - Q1
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

#1 - Figure 1-f
'''
bin_dist = [2,2,4,5,6,8,9,9,10]
e_max = [item for item in bin_dist if item >= .9*max(bin_dist)]
print(len(bin_dist) - len(e_max))
numb = len(bin_dist) - len(e_max)
print(e_max)
bin_dist.sort()
plt.bar(np.arange(0,numb),bin_dist[:numb],color = ['lightskyblue'])
plt.bar(np.arange(numb,9),bin_dist[numb:],color = ['lightcoral'])
plt.axhline(y=.9*max(bin_dist), color='dimgrey', linestyle='--')
plt.ylabel("Sum of synaptic inputs (h)", fontsize = 12)
plt.xlabel("Neuron index (j)", fontsize = 12)
plt.text(.36,9.2, r"$h_{min}(t) = \left(1 - \epsilon \right)h_{max}(t)$", fontsize=12)
plt.xticks(np.arange(0,10,2))
plt.show()
'''

'''
bin_dist = [2,2,4,5,6,8,9,9,10]

bin_dist.sort()
plt.bar(np.arange(0,len(bin_dist)),bin_dist,color = ['lightskyblue'])
plt.ylabel("Sum of synaptic inputs (h)", fontsize = 12)
plt.xlabel("Neuron index (j)", fontsize = 12)
plt.text(.36,9.2, "k-winners-take-all", fontsize = 12)
plt.xticks(np.arange(0,10,2))
plt.show()
'''
#2 - Figure 2-b
'''
def plot_function(ax, beta, beta_float,ticks):
    df = pd.read_csv("k_winners\k_winners_{0}".format(beta))
    iter = np.arange(1,101)
    ax.bar(iter, df["new"], color = "lightskyblue")
    ax.set_title(r'$\bf\beta = {0}$'.format(beta_float))
    ax.set_xlim(0,ticks)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$|N_t|$')
    x_plot = list(df["new"]).index(0)
    print(x_plot)
    ax.axvline(x = x_plot+1, color = "dimgrey", ls = "--", lw = 1)
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\left|X_t\right|$')
    ax2.plot(iter,df["new_1"], color = "lightcoral", lw = .5)

fig, axes = plt.subplots(1,2)

beta = ["0_01","0_001"]
beta_float = [0.01,0.001]
ticks = [45,70]
for i in range(0,2):
    plot_function(axes[i],beta[i],beta_float[i],ticks[i])
plt.show()
'''

#3 - Figure 2-c
'''
beta = ["01","001"]
beta_n =[.01,.001]
j = 79
fig, ax  = plt.subplots(2,1,sharex=True)

def plot_axis(ax,data,beta_n,y):
    delta_fire = []
    for i in range(0,200):
        if i == 0:
            delta_fire.append(data.iloc[i])
        else:
            delta_fire.append(data.iloc[i] - data.iloc[i-1])

    ax.plot(np.arange(0,200),delta_fire, linewidth = 1, c = "lightcoral")
    ax.set_xlim(-1,200)
    #ax.set_ylim(-50,50)
    ax.set_ylabel(r'$\sum \Delta f$', fontsize = 16)
    ax.text(165,y,r'$\beta = {0}$'.format(beta_n))
    #ax.set_ylabel(r'$\sum\Delta f_{t}$')
y = [18,20]

for i in range(0,len(beta)):
    df = pd.read_csv("convergence\convergence_0_"+ beta[i] + "_" + str(j + 1))
    df_fire = df["fire"]
    plot_axis(ax[i],df_fire,beta_n[i],y[i])
print(df)
ax[1].set_xlabel(r'$t$', fontsize = 16)
plt.show()
'''

'''
beta = ["01","001"]
beta_n =[.01,.001]
j = 79
fig, ax  = plt.subplots(2,1,sharex=True)

def plot_axis(ax,data,beta_n,y):
    delta_fire = []
    for i in range(0,200):
        if i == 0:
            delta_fire.append(data.iloc[i])
        else:
            delta_fire.append(data.iloc[i])

    ax.plot(np.arange(0,200),delta_fire, linewidth = 1, c = "lightskyblue")
    ax.set_xlim(-1,200)
    #ax.set_ylim(-50,50)
    ax.set_ylabel(r'$\left|X_{t}\right|$', fontsize = 16)
    ax.text(165,y,r'$\beta = {0}$'.format(beta_n))
    #ax.set_ylabel(r'$\sum\Delta f_{t}$')
y = [18,20]

for i in range(0,len(beta)):
    df = pd.read_csv("convergence\convergence_0_"+ beta[i] + "_" + str(j + 1))
    df_fire = df["new_1"]
    plot_axis(ax[i],df_fire,beta_n[i],y[i])
print(df)
ax[1].set_xlabel(r'$t$', fontsize = 16)
plt.show()
'''
#4 - Figure 2-d
'''
inhibition = ["02","04","06","08","10"]
df_mean = pd.DataFrame()
df_std = pd.DataFrame()
for i in inhibition:
    df= pd.read_csv("error_formation_{0}".format(i), index_col=0)
    df_mean[i] = df.mean()
    df_std[i] = df.std()

inhibition = ["- 0.2","- 0.4","- 0.6","- 0.8","- 1.0"]
beta = ['0.1','0.05','0.01','0.005','0.001']

df_mean.columns = inhibition
df_std.columns = inhibition
df_mean.index = beta
df_std.index = beta
print(df_mean)
print(df_std)



ax = df_mean.plot.bar(yerr=df_std,rot = 0,edgecolor = 'black', capsize=4,color=["#A8F0F0","#A8F0BB","#EAF0A8","#AAA8F0","#F0A8A8"])
plt.legend(title = r'$\omega_{inh}$', frameon = False)
plt.ylabel(r'Failure rate (mean)', fontsize = 15)
plt.xlabel(r'Synaptic plasticity $(\beta)$', fontsize = 15)
ax.invert_xaxis()
plt.show()
'''

#5 - Figure 3-a
'''
def outliers(df):
    print("n_data: {0}".format(len(df["size"])))
    Q3 = df.quantile(.75) ; Q1 = df.quantile(.25)
    IQR = Q3 - Q1
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
f, axes = plt.subplots(1,2)
size = 7
conc_mdf = []
conc_02 = []
for i in range(0,5):
    df_without = filter(pd.read_csv("{0}ormation_without\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = 0))
    df_with = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = 1))
    df_02 = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = beta_float[i]))
    conc_02.append(df_02)
    print("BETA: ", beta[i])
    print("========WITHOUT========")
    print(df_without.describe())
    print("========WITH========")
    print(df_with.describe())
    print()

    cdf = pd.concat([df_without,df_with])
    mdf = pd.melt(cdf, id_vars=["size"], value_vars=["Trial"])
    mdf["variable"] = beta_float[i]
    conc_mdf.append(mdf)
cdf = pd.concat(conc_mdf)
cdf_02 = pd.concat(conc_02)

print(cdf_02)
cdf['value'] = np.where(cdf['value']==1, "With inhibition", 'Without inhibition')
sns.boxplot(x = "variable", hue = "value", y="size", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.7, palette=['lightskyblue', 'lightcoral'], linewidth=.5, ax=axes[1])
sns.boxplot(x = "Trial", y="size", data=cdf_02, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightcoral'], linewidth=.5, ax=axes[0])
axes[1].set_xlabel(r'Synaptic plasticity $(\beta)$')
axes[0].set_xlabel(r'Synaptic plasticity $(\beta)$')
axes[1].set_ylabel(r'Size $(\left|A\right|)$')
axes[0].set_ylabel(r'Size $(\left|A\right|)$')
axes[0].spines[['right', 'top']].set_visible(False)
axes[1].spines[['right', 'top']].set_visible(False)
legend = axes[1].legend(frameon = False)
plt.show()
'''

#6 - Figure 3-XX
'''
beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
f, axes = plt.subplots(1,1)
conc_cdf = []
for i in range(0,5):
    df_emax = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_emax["variable"] = beta_float[i]
    conc_cdf.append(df_emax)
cdf = pd.concat(conc_cdf)
sns.boxplot(x = "variable", y="density", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.7, palette=['lightskyblue'], linewidth=.5)
axes.legend(frameon=False)
axes.set_xlabel(r'Synaptic plasticity $(\beta)$')
axes.set_ylabel(r"Synaptic density $(D_{A})$")
axes.spines[['right', 'top']].set_visible(False)
plt.show()
'''

#7 - Figure 3-c
'''
norm = mcolors.Normalize(0, 60)

fig,(axes_1,axes_2) = plt.subplots(2,2)
df_A = pd.read_csv("overlap_matrix\matrix_assembly", index_col=0)
df_S = pd.read_csv("overlap_matrix\matrix_stimuli", index_col=0)
print(df_A)
print(df_S)

axes_1[0] = sns.heatmap(data=df_S, fmt='d', cmap='YlOrBr', cbar=True, square=True, ax=axes_1[0],cbar_kws={'label': 'Number of neurons'})
axes_1[0].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)

axes_1[1] = sns.heatmap(data=df_A, fmt='d', cmap='crest', cbar=True, square=True, ax=axes_1[1], norm=norm,cbar_kws={'label': 'Number of neurons'})
axes_1[1].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)


df_Ak = pd.read_csv("overlap_matrix\matrix_assembly_k", index_col=0)
df_Sk = pd.read_csv("overlap_matrix\matrix_stimuli_k", index_col=0)
print(df_Ak)
print(df_Sk)

axes_2[0] = sns.heatmap(data=df_Sk, fmt='d', cmap='YlOrBr', cbar=True, square=True, ax=axes_2[0],cbar_kws={'label': 'Number of neurons'})
axes_2[0].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)

axes_2[1] = sns.heatmap(data=df_Ak, fmt='d', cmap='crest', cbar=True, square=True, ax=axes_2[1], norm=norm, cbar_kws={'label': 'Number of neurons'})
axes_2[1].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)
plt.show()
'''

#8 - Figure 3-d
'''
data_emax_overlap = []
data_k_win_overlap = []
for i in range(0,100):
    df_matrix_E = pd.read_csv("overlap_matrix\matrix_assembly_emax_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_E.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_E))] = False
    new_matrix = df_matrix_E.to_numpy()[mask]
    data_emax_overlap += list(new_matrix)

    df_matrix_K = pd.read_csv("overlap_matrix\matrix_assembly_k_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_K.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_K))] = False
    new_matrix = df_matrix_K.to_numpy()[mask]
    data_k_win_overlap += list(new_matrix)

print(median(data_emax_overlap))
print(median(data_k_win_overlap))

fig,axes = plt.subplots(4,1,sharex=True)
sns.histplot(data=data_emax_overlap, color="lightskyblue" ,ax=axes[1], bins = range(10), label = r'E%-WTA model')
axes[1].legend(frameon=False)
axes[1].set_ylabel(r"#$|A_{i}\cap A_{j}|$", fontsize = 16)
axes[1].spines[['right', 'top']].set_visible(False)
sns.boxplot(data=data_emax_overlap, ax=axes[0], orient="h", color="lightskyblue",width=.2,flierprops={'marker': 'x', 'markersize': 5})
axes[0].set(yticks=[])
sns.despine(ax=axes[1])
sns.despine(ax=axes[0], left=True)

sns.histplot(data=data_k_win_overlap, color="lightcoral",ax=axes[3], bins = range(10), label = r'AC model')
axes[3].legend(frameon=False)
axes[3].set_xlabel(r'$|A_{i}\cap A_{j}|$', fontsize = 16)
axes[3].set_ylabel(r"#$|A_{i}\cap A_{j}|$", fontsize = 16)
axes[3].spines[['right', 'top']].set_visible(False)
axes[3].set_xticks(np.arange(0,20,2))
sns.boxplot(data=data_k_win_overlap, ax=axes[2], orient="h", color="lightcoral",width=.2,flierprops={'marker': 'x', 'markersize': 5})
axes[2].set(yticks=[])
sns.despine(ax=axes[3])
sns.despine(ax=axes[2], left=True)
plt.show()
'''

#9 - Fig 4/a,b,c
'''
fig, axes = plt.subplots(1,3)
df_list = []
rec_list = [2,4,6,8,10]
for i in range(0,len(rec_list)):
    df = pd.DataFrame(pd.read_csv("rec_"+ str(rec_list[i]), index_col=0)).assign(Trial = rec_list[i])
    df_list.append(df)
cdf = pd.concat(df_list)
print(cdf)
sns.boxplot(x = "Trial", y= "0", data=cdf, ax = axes[2], color="lightskyblue",flierprops={'marker': 'x', 'markersize': 5})
axes[2].set_ylabel(r'Recovered portion ($|A|_{rec}$/$|A|$)')
axes[2].set_xlabel(r"Assemblies per area")
axes[2].spines[['right', 'top']].set_visible(False)

axes[0].plot([.1,.2,.3,.4,.5],[58/200,107/200,138/200,170/200,179/200],color="lightskyblue")   # See the simulations file (AC.py) to understand these inserted values (Simulation 5).
axes[0].spines[['right', 'top']].set_visible(False)
axes[0].set_xlabel(r"Synaptic connection probability($p_{s}$)")
axes[0].set_ylabel(r"Sucess rate ")
axes[1].plot([50,100,150,250,500],[66/200,146/200,161/200,189/200,198/200],color="lightskyblue") # See the simulations file (AC.py) to understand these inserted values (Simulation 5).
axes[1].spines[['right', 'top']].set_visible(False)
axes[1].set_ylabel(r"Sucess rate")
axes[1].set_xlabel(r"Stimulus size ($k_{s}$)")
plt.show()
'''

#==== STATISTICAL ANALYSIS (Nonparametrical-tests) ====#

#Kruskal + Dunn-Test (T) Table 1
# Note: Code below: Indicates how many of the 500 simulations formed assemblies and were used in the analysis of the characteristics.
'''
import scikit_posthocs as sp

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
f, axes = plt.subplots(1,1)
conc_mdf = []

for i in range(0,5):
    df_emax = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_kwin = pd.read_csv("{0}ormation_without_emax\{1}ormation_0_{2}_k_winners_".format("f","f",beta[i]), index_col=0)
    print("==== BETA:{0} ====".format(beta_float[i]))
    print("##Adapted_Model")  
    print(df_emax.info()) # Indicates how many of the 500 simulations formed assemblies and were used in the analysis of the characteristics.
    #print("##Original Model")
    #print(df_kwin.describe())
    print()
    df_emax["plasticity"] = beta_float[i]
    conc_mdf.append(df_emax)

cdf = pd.concat(conc_mdf)
#print(cdf)
result = sp.posthoc_dunn(cdf, 'time', 'plasticity', 'bonferroni')
print(result)
#    df_kwin = pd.read_csv("{0}ormation_without_emax\{1}ormation_0_{2}_k_winners_".format("f","f",beta[i]), index_col=0).assign(Trial = 1)
'''

#Mann-Whitney (T) Table 1
'''
import scipy.stats as stats

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
for i in range(0,5):
    df_emax = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_kwin = pd.read_csv("{0}ormation_without_emax\{1}ormation_0_{2}_k_winners_".format("f","f",beta[i]), index_col=0)
    
    stat, p = stats.mannwhitneyu(df_emax['time'],df_kwin['time'], alternative = 'two-sided')
    print("Beta {0}: ".format(beta_float[i]), p)
    print("##Adapted_Model")
    print(df_emax.describe())
    print("##Original_Model")
    print(df_kwin.describe())
    print("=======================")
    print()
'''

#Mann-Whitney (Size, density) Table 1
'''
import scipy.stats as stats
import scikit_posthocs as sp

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
cdf_concat = []
for i in range(0,5):
    df_with = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_without = filter(pd.read_csv("{0}ormation_without\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0))
    df_with['plasticity'] = beta_float[i]
    
    stat, p = stats.mannwhitneyu(df_with['size'],df_without['size'], alternative = 'two-sided')
    print("Beta {0}: ".format(beta_float[i]), p)
    stat, p_with = shapiro(df_with["density"])
    stat, p_without = shapiro(df_without["density"])
    print("Shapiro with: ", p_with)
    print("Shapiro without: ", p_without)
    #print(df_with.describe())
    print(df_without.info())
    #print(df_with)
    #print(df_without)
    print("=======================")
    print()
    cdf_concat.append(df_with)

cdf = pd.concat(cdf_concat)
result_size = sp.posthoc_dunn(cdf, 'size', 'plasticity', 'bonferroni')
result_density = sp.posthoc_dunn(cdf, 'density', 'plasticity', 'bonferroni')
print("Kruskal Size")
print(result_size)
print("Kruskal Density")
print(result_density)
#result = sp.posthoc_dunn(cdf, 'time', 'plasticity', 'bonferroni')
'''

#Mann-Whitney (Overlap)
'''
import scipy.stats as stats
import scikit_posthocs as sp


data_emax_overlap = []
data_k_win_overlap = []
for i in range(0,100):
    df_matrix_E = pd.read_csv("overlap_matrix\matrix_assembly_emax_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_E.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_E))] = False
    new_matrix = df_matrix_E.to_numpy()[mask]
    data_emax_overlap += list(new_matrix)

    df_matrix_K = pd.read_csv("overlap_matrix\matrix_assembly_k_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_K.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_K))] = False
    new_matrix = df_matrix_K.to_numpy()[mask]
    data_k_win_overlap += list(new_matrix)

df_E = pd.DataFrame(data_emax_overlap)
df_K = pd.DataFrame(data_k_win_overlap)

print(df_E.describe())

print(df_K.describe())

stat, p = stats.mannwhitneyu(df_E,df_K, alternative = 'two-sided')
print("p-value:", p)
'''

#Mann-Whitney & Kruskal-Wallis: Recovery + Plot (8 - Figure 3-b / Table 1)
'''
import scipy.stats as stats
import scikit_posthocs as sp

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]

size_stimuli_emax = [50,100,150,200]
size_stimuli_kwin = [9,18,28,37]
f, axes = plt.subplots(1,1)

conc_mdf = []
conc_emax = []

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
conc_mdf = []
conc_emax = []
for i in range(0,5):
    df_e_max = pd.read_csv("recovery\{1}ecovery_e_max_0_{0}_15".format(beta[i],"r"), index_col=0).assign(Trial = 0)
    df_k_win = pd.read_csv("recovery\{1}ecovery_k_win_0_{0}_15".format(beta[i],"r"), index_col=0).assign(Trial = 1)
    print("BETA: {0}".format(beta_float[i]))
    stat, p = stats.mannwhitneyu(df_e_max['0'],df_k_win['0'], alternative = 'two-sided')
    print("Mann-Whitney test: ", p)
    print(df_e_max.describe())
    print(df_k_win.describe())
    print()
    cdf_t = pd.concat([df_e_max,df_k_win])
    mdf = pd.melt(cdf_t, id_vars=["0"], value_vars=["Trial"])
    mdf["variable"] = beta_float[i]
    df_e_max["variable"] = beta_float[i]
    conc_mdf.append(mdf)
    conc_emax.append(df_e_max)
cdf_1 = pd.concat(conc_mdf)
cdf_emax_1 = pd.concat(conc_emax)

result = sp.posthoc_dunn(cdf_emax_1, '0', 'variable', 'bonferroni')
print("Kruskal-Wallis")
print(result)


#cdf['value'] = np.where(cdf['value']==1, 'k-winners-take-all', 'E%-winners-take-all')
cdf_1['value'] = np.where(cdf_1['value']==1, r'AC model', r'E%-WTA model')
#sns.boxplot(x = "variable", hue = "value", y="0", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightskyblue', 'lightcoral'], linewidth=.5, ax=axes[1])
sns.boxplot(x = "variable", hue = "value", y="0", data=cdf_1, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightskyblue','lightcoral'], linewidth=.5, ax=axes)
#axes[1].set_xlabel(r'Tamanho do estímulo $(\%)$')
axes.set_xlabel(r'Synaptic plasticity $(\beta)$')
axes.spines[['right', 'top']].set_visible(False)
#axes[1].set_ylabel(r'Porção recuperada $(\%)$')
axes.set_ylabel(r'Recovered portion ($|A|_{rec}$/$|A|$)')
#legend = axes[1].legend(frameon = False)
legend = axes.legend(frameon = False)
plt.show()
'''

#Figure 1 (Paper)
'''
siz = 0.4
fig = plt.figure(figsize=(12,10))
rect = [(.07, 0, siz, siz) ,#1
        (-0.03, .4, .55, .55), #2
        (0.57, 0.12, .15, .25),#3
        (0.44, 0.41, .57, .57),#4
        (0.77, 0.12, .15, .25)]
ax = [fig.add_axes(rect[i]) for i in range(5)]
plt.figtext(0.05, 0.93, "a)", fontsize=15,weight = 'bold')
plt.figtext(0.05, 0.65, "b)", fontsize=15,weight = 'bold')
plt.figtext(0.537, 0.94, "c)", fontsize=15,weight = 'bold')
plt.figtext(0.537, 0.66, "d)", fontsize=15,weight = 'bold')
plt.figtext(0.05, 0.38, "e)", fontsize=15,weight = 'bold')
plt.figtext(0.537, 0.38, "f)", fontsize=15,weight = 'bold')

im_1 = plt.imread(get_sample_data('C:\\Users\\IRIS\\Desktop\\Mestrado\\Artigo\\A_B.png'))
ax[1].imshow(im_1)
ax[1].axis('off')
im_2 = plt.imread(get_sample_data('C:\\Users\\IRIS\\Desktop\\Mestrado\\Artigo\\C_D.png'))
plt.figtext(0.62, 0.95, r"$t = 1$", fontsize=10)
plt.figtext(0.81, 0.95, r"$t = 2$", fontsize=10)
plt.figtext(0.62, 0.666, r"$t = 0$", fontsize=10)
plt.figtext(0.805, 0.666, r"$t = 10$", fontsize=10)
plt.figtext(0.685, 0.95, r"$f_5(1) = 0$", fontsize=10)
plt.figtext(0.685, 0.715, r"$f_{40}(1) = 1$", fontsize=10)
ax[3].imshow(im_2)
ax[3].axis('off')
im_3 = plt.imread(get_sample_data('C:\\Users\\IRIS\\Desktop\\Mestrado\\Artigo\\E.png'))
ax[0].imshow(im_3)
ax[0].axis('off')


bin_dist = [2,2,4,5,6,8,9,9,10]
e_max = [item for item in bin_dist if item >= .9*max(bin_dist)]
print(len(bin_dist) - len(e_max))
numb = len(bin_dist) - len(e_max)

bin_dist.sort()
ax[2].bar(np.arange(0,numb),bin_dist[:numb],color = ['lightskyblue'])
ax[2].bar(np.arange(numb,9),bin_dist[numb:],color = ['lightcoral'])
ax[2].axhline(y=.9*max(bin_dist), color='dimgrey', linestyle='--')
ax[2].set_ylabel("Sum of synaptic inputs (h)")
ax[2].set_xlabel("Neuron index (j)")
ax[2].text(.36,9.2, r"$h_{min}(t) = \left(1 - \epsilon \right)h_{max}(t)$", fontsize=8)
ax[2].set_xticks(np.arange(0,10,2))


bin_dist = [2,2,4,5,6,8,9,9,10]

bin_dist.sort()
ax[4].bar(np.arange(0,len(bin_dist)),bin_dist,color = ['lightskyblue'])
ax[4].set_ylabel("Sum of synaptic inputs (h)")
ax[4].set_xlabel("Neuron index (j)")
ax[4].text(.36,9.2, "k-winners-take-all", fontsize = 8)
ax[4].set_xticks(np.arange(0,10,2))
plt.savefig('Figure_1')
plt.show()
'''

#Figure 2 (Paper)
'''
siz = 0.4
fig = plt.figure(figsize=(15,10))
rect = [(-0.03, .45, .45, .5),#1
        (0.475, 0.53, .21, .4),
        (0.755, 0.53, .21, .4),
        (0.5, 0.09, .45, .35),
        (.05, 0.37, .37, .08),
        (.05, 0.28, .37, .08),
        (.05, 0.145, .37, .08),
        (.05, 0.055, .37, .08)] 
ax = [fig.add_axes(rect[i]) for i in range(8)]

im_1 = plt.imread(get_sample_data('C:\\Users\\IRIS\\Desktop\\Mestrado\\Artigo\\A_2.png'))
ax[0].imshow(im_1)
ax[0].axis('off')


def plot_function(ax, beta, beta_float,ticks):
    df = pd.read_csv("k_winners\k_winners_{0}".format(beta))
    iter = np.arange(1,101)
    ax.bar(iter, df["new"], color = "lightskyblue")
    ax.set_title(r'$\bf\beta = {0}$'.format(beta_float))
    ax.set_xlim(0,ticks)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$|N_t|$')
    x_plot = list(df["new"]).index(0)
    print(x_plot)
    ax.axvline(x = x_plot+1, color = "dimgrey", ls = "--", lw = 1)
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\left|X_t\right|$')
    ax2.plot(iter,df["new_1"], color = "lightcoral", lw = .5)

axes = [ax[1],ax[2]]
beta = ["0_01","0_001"]
beta_float = [0.01,0.001]
ticks = [45,70]
for i in range(0,2):
    plot_function(axes[i],beta[i],beta_float[i],ticks[i])
####
beta = ["01","001"]
beta_n =[.01,.001]
j = 79

def plot_axis(ax,data,beta_n,y):
    delta_fire = []
    for i in range(0,200):
        if i == 0:
            delta_fire.append(data.iloc[i])
        else:
            delta_fire.append(data.iloc[i])

    ax.plot(np.arange(0,200),delta_fire, linewidth = 1, c = "lightskyblue")
    ax.set_xlim(-1,200)
    #ax.set_ylim(-50,50)
    ax.set_ylabel(r'$\left|X_{t}\right|$')
    ax.text(165,y,r'$\beta = {0}$'.format(beta_n))
    #ax.set_ylabel(r'$\sum\Delta f_{t}$')
y = [18,20]
axes = [ax[4],ax[5]]
for i in range(0,len(beta)):
    df = pd.read_csv("convergence\convergence_0_"+ beta[i] + "_" + str(j + 1))
    df_fire = df["new_1"]
    plot_axis(axes[i],df_fire,beta_n[i],y[i])
print(df)
ax[5].set_xlabel(r'$t$')
ax[4].set_xticklabels([],color = 'w')
####
beta = ["01","001"]
beta_n =[.01,.001]
j = 79

def plot_axis(ax,data,beta_n,y):
    delta_fire = []
    for i in range(0,200):
        if i == 0:
            delta_fire.append(data.iloc[i])
        else:
            delta_fire.append(data.iloc[i] - data.iloc[i-1])

    ax.plot(np.arange(0,200),delta_fire, linewidth = 1, c = "lightcoral")
    ax.set_xlim(-1,200)
    #ax.set_ylim(-50,50)
    ax.set_ylabel(r'$\sum \Delta f$')
    ax.text(165,y,r'$\beta = {0}$'.format(beta_n))
    #ax.set_ylabel(r'$\sum\Delta f_{t}$')
y = [14,10]
axes = [ax[6],ax[7]]
for i in range(0,len(beta)):
    df = pd.read_csv("convergence\convergence_0_"+ beta[i] + "_" + str(j + 1))
    df_fire = df["fire"]
    plot_axis(axes[i],df_fire,beta_n[i],y[i])

ax[7].set_xlabel(r'$t$')
ax[6].set_xticklabels([],color = 'w')

#####
inhibition = ["02","04","06","08","10"]
df_mean = pd.DataFrame()
df_std = pd.DataFrame()
for i in inhibition:
    df= pd.read_csv("error_formation_{0}".format(i), index_col=0)
    df_mean[i] = df.mean()
    df_std[i] = df.std()

inhibition = ["- 0.2","- 0.4","- 0.6","- 0.8","- 1.0"]
beta = ['0.1','0.05','0.01','0.005','0.001']

df_mean.columns = inhibition
df_std.columns = inhibition
df_mean.index = beta
df_std.index = beta


df_mean.plot.bar(yerr=df_std,rot = 0,edgecolor = 'black', capsize=4,color=["#A8F0F0","#A8F0BB","#EAF0A8","#AAA8F0","#F0A8A8"], ax = ax[3])
ax[3].legend(title = r'$\omega_{inh}$', frameon = False, prop={'size': 7}, loc =2 )
ax[3].set_ylabel(r'Failure rate (mean)')
ax[3].set_xlabel(r'Synaptic plasticity $(\beta)$')
ax[3].invert_xaxis()


plt.figtext(0.005, 0.95, "a)", fontsize=15,weight = 'bold')
plt.figtext(0.45, 0.95, "b)", fontsize=15,weight = 'bold')
plt.figtext(0.005, 0.45, "c)", fontsize=15,weight = 'bold')
plt.figtext(0.45, 0.45, "d)", fontsize=15,weight = 'bold')

plt.show() 
'''

#Figure 3 (Paper)
'''
siz = 0.4
siz_1 = 0.2
fig = plt.figure(figsize=(15,10))
rect = [(.05, .25, siz_1, siz_1), # x
        (.05, .57, .2, siz), # x
        (0.55, 0.43, .4, .08),#
        (0.55, 0.57, siz, siz),# x
        (.3, .57, .2, siz), # x
        (.25, .25, siz_1, siz_1), # x
        (.05, .01, siz_1, siz_1), # x
        (.25, .01, siz_1, siz_1), # x
        (0.55, 0.3, .4, .1), #
        (0.55, 0.20, .4, .08), #
        (0.55, 0.07, .4, .1) ] # 
ax = [fig.add_axes(rect[i]) for i in range(11)]



def outliers(df):
    print("n_data: {0}".format(len(df["size"])))
    Q3 = df.quantile(.75) ; Q1 = df.quantile(.25)
    IQR = Q3 - Q1
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]

size = 7
conc_mdf = []
conc_02 = []
for i in range(0,5):
    df_without = filter(pd.read_csv("{0}ormation_without\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = 0))
    df_with = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = 1))
    df_02 = filter(pd.read_csv("{0}ormation_02\{1}ormation_0_{2}".format("f","f",beta[i]), index_col=0).assign(Trial = beta_float[i]))
    conc_02.append(df_02)
    print("BETA: ", beta[i])
    print("========WITHOUT========")
    print(df_without.describe())
    print("========WITH========")
    print(df_with.describe())
    print()

    cdf = pd.concat([df_without,df_with])
    mdf = pd.melt(cdf, id_vars=["size"], value_vars=["Trial"])
    mdf["variable"] = beta_float[i]
    conc_mdf.append(mdf)
cdf = pd.concat(conc_mdf)
cdf_02 = pd.concat(conc_02)

print(cdf_02)
cdf['value'] = np.where(cdf['value']==1, "With inhibition", 'Without inhibition')
sns.boxplot(x = "variable", hue = "value", y="size", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.7, palette=['lightskyblue', 'lightcoral'], linewidth=.5, ax=ax[4])
sns.boxplot(x = "Trial", y="size", data=cdf_02, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightcoral'], linewidth=.5, ax=ax[1])
ax[4].set_xlabel(r'Synaptic plasticity $(\beta)$')
ax[1].set_xlabel(r'Synaptic plasticity $(\beta)$')
ax[4].set_ylabel(r'Size $(\left|A\right|)$')
ax[1].set_ylabel(r'Size $(\left|A\right|)$')
ax[1].spines[['right', 'top']].set_visible(False)
ax[4].spines[['right', 'top']].set_visible(False)
legend = ax[4].legend(frameon = False)

import scipy.stats as stats
import scikit_posthocs as sp

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]

size_stimuli_emax = [50,100,150,200]
size_stimuli_kwin = [9,18,28,37]

conc_mdf = []
conc_emax = []

beta = ["1","05","01","005","001"]
beta_float = [.1,.05,.01,.005,.001]
conc_mdf = []
conc_emax = []
for i in range(0,5):
    df_e_max = pd.read_csv("recovery\{1}ecovery_e_max_0_{0}_15".format(beta[i],"r"), index_col=0).assign(Trial = 0)
    df_k_win = pd.read_csv("recovery\{1}ecovery_k_win_0_{0}_15".format(beta[i],"r"), index_col=0).assign(Trial = 1)
    print("BETA: {0}".format(beta_float[i]))
    stat, p = stats.mannwhitneyu(df_e_max['0'],df_k_win['0'], alternative = 'two-sided')
    print("Mann-Whitney test: ", p)
    print(df_e_max.describe())
    print(df_k_win.describe())
    print()
    cdf_t = pd.concat([df_e_max,df_k_win])
    mdf = pd.melt(cdf_t, id_vars=["0"], value_vars=["Trial"])
    mdf["variable"] = beta_float[i]
    df_e_max["variable"] = beta_float[i]
    conc_mdf.append(mdf)
    conc_emax.append(df_e_max)
cdf_1 = pd.concat(conc_mdf)
cdf_emax_1 = pd.concat(conc_emax)

result = sp.posthoc_dunn(cdf_emax_1, '0', 'variable', 'bonferroni')
print("Kruskal-Wallis")
print(result)


#cdf['value'] = np.where(cdf['value']==1, 'k-winners-take-all', 'E%-winners-take-all')
cdf_1['value'] = np.where(cdf_1['value']==1, r'AC model', r'E%-WTA model')
#sns.boxplot(x = "variable", hue = "value", y="0", data=cdf, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightskyblue', 'lightcoral'], linewidth=.5, ax=axes[1])
sns.boxplot(x = "variable", hue = "value", y="0", data=cdf_1, flierprops={'marker': 'x', 'markersize': 5}, width=.5, palette=['lightskyblue','lightcoral'], linewidth=.5, ax=ax[3])
#axes[1].set_xlabel(r'Tamanho do estímulo $(\%)$')
ax[3].set_xlabel(r'Synaptic plasticity $(\beta)$')
ax[3].spines[['right', 'top']].set_visible(False)
#axes[1].set_ylabel(r'Porção recuperada $(\%)$')
ax[3].set_ylabel(r'Recovered portion ($|A|_{rec}$/$|A|$)')
#legend = axes[1].legend(frameon = False)
legend = ax[3].legend(frameon = False)

norm = mcolors.Normalize(0, 60)


df_A = pd.read_csv("overlap_matrix\matrix_assembly", index_col=0)
df_S = pd.read_csv("overlap_matrix\matrix_stimuli", index_col=0)
print(df_A)
print(df_S)

ax[0] = sns.heatmap(data=df_S, fmt='d', cmap='managua', cbar=True, square=True, ax=ax[0],cbar_kws={'label': 'Number of neurons'})
ax[0].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)

ax[5] = sns.heatmap(data=df_A, fmt='d', cmap='crest', cbar=True, square=True, ax=ax[5], norm=norm,cbar_kws={'label': 'Number of neurons'})
ax[5].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)


df_Ak = pd.read_csv("overlap_matrix\matrix_assembly_k", index_col=0)
df_Sk = pd.read_csv("overlap_matrix\matrix_stimuli_k", index_col=0)
print(df_Ak)
print(df_Sk)

ax[6] = sns.heatmap(data=df_Sk, fmt='d', cmap='managua', cbar=True, square=True, ax=ax[6],cbar_kws={'label': 'Number of neurons'})
ax[6].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)

ax[7] = sns.heatmap(data=df_Ak, fmt='d', cmap='crest', cbar=True, square=True, ax=ax[7], norm=norm, cbar_kws={'label': 'Number of neurons'})
ax[7].tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True, length=0)


data_emax_overlap = []
data_k_win_overlap = []
for i in range(0,100):
    df_matrix_E = pd.read_csv("overlap_matrix\matrix_assembly_emax_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_E.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_E))] = False
    new_matrix = df_matrix_E.to_numpy()[mask]
    data_emax_overlap += list(new_matrix)

    df_matrix_K = pd.read_csv("overlap_matrix\matrix_assembly_k_{0}".format(i), index_col=0)
    mask = np.ones(df_matrix_K.shape, dtype=bool)
    mask[np.triu_indices(len(df_matrix_K))] = False
    new_matrix = df_matrix_K.to_numpy()[mask]
    data_k_win_overlap += list(new_matrix)


#fig,axes = plt.subplots(4,1,sharex=True)
sns.histplot(data=data_emax_overlap, color="lightskyblue" ,ax=ax[8], bins = range(10), label = r'E%-WTA model')
ax[8].legend(frameon=False)
ax[8].set_ylabel(r"#$|A_{i}\cap A_{j}|$")
ax[8].spines[['right', 'top']].set_visible(False)
sns.boxplot(data=data_emax_overlap, ax=ax[2], orient="h", color="lightskyblue",width=.2,flierprops={'marker': 'x', 'markersize': 5})
ax[2].set_xticklabels([],color = 'w')
sns.despine(ax=ax[8])
sns.despine(ax=ax[2], left=True)

sns.histplot(data=data_k_win_overlap, color="lightcoral",ax=ax[10], bins = range(10), label = r'AC model')
ax[10].legend(frameon=False)
ax[10].set_xlabel(r'$|A_{i}\cap A_{j}|$')
ax[10].set_ylabel(r"#$|A_{i}\cap A_{j}|$")
ax[10].spines[['right', 'top']].set_visible(False)
ax[10].set_xticks(np.arange(0,20,2))
sns.boxplot(data=data_k_win_overlap, ax=ax[9], orient="h", color="lightcoral",width=.2,flierprops={'marker': 'x', 'markersize': 5})
ax[9].set_xticklabels([],color = 'w')
ax[8].set_xticklabels([],color = 'w')
sns.despine(ax=ax[10])
sns.despine(ax=ax[9], left=True)
ax[8].sharex(ax[10]) ; ax[9].sharex(ax[10]) ; ax[2].sharex(ax[10])
ax[2].get_yaxis().set_visible(False)
ax[9].get_yaxis().set_visible(False)

plt.figtext(0.005, 0.97, "a)", fontsize=15,weight = 'bold')
plt.figtext(0.5, 0.97, "b)", fontsize=15,weight = 'bold')
plt.figtext(0.07, 0.47, "c)", fontsize=15,weight = 'bold')
plt.figtext(0.49, 0.47, "d)", fontsize=15,weight = 'bold')
plt.figtext(0.08, 0.29, "E%-WTA model", fontsize=10,rotation=90)
plt.figtext(0.08, 0.07, "AC model", fontsize=10,rotation=90)
plt.figtext(0.14, 0.48, "Stimuli", fontsize=10)
plt.figtext(0.33, 0.48, "Assemblies", fontsize=10)
plt.show()
'''
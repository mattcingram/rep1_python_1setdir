#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setup


# ## Load packages

# In[1]:


import os
import pandas as pd
import matplotlib
import requests   # manage web requests
import numpy as np   # core methods package for math and to manage various data objects; many other uses
import matplotlib.pyplot as plt  # plotting library
#from sklearn import linear_model   # data science library; not loading in jupyter
import statsmodels.formula.api as sm  # statistics library
import seaborn as sns   # data visualization library
import inspect # contains getsource() function to inspect source code
import platform # to check identifying info of python, e.g., version


# ## Check Python Version

# In[2]:


platform.python_version()


# ## Set Directory Structure
# 

# ### Working Directory
# Not really needed on jupyter container because home directory is set. All filepaths are relative to home directory.

# In[3]:


print(os.getcwd()) # check current working dir
path = '/home/jovyan/'
os.chdir(path)
print(os.getcwd()) # ensure cwd changed to desired dir


# ### Make Subdirectories

# In[4]:


os.makedirs('./code', exist_ok=True)
os.makedirs('./data', exist_ok=True)
os.makedirs('./figures', exist_ok=True)
os.makedirs('./tables', exist_ok=True)


# ### Check Directory Contents

# In[7]:


os.listdir()


# # Load Data

# In[5]:


df = pd.read_csv('./data/original/metoo_data.csv')


# In[5]:


df


# In[34]:


# can also index within objects
df.iloc[0:5,0:9]
# notice first index value in python is 0


# In[6]:


df.dtypes


# # Data Processing

# In[6]:


# recode experimental condition
df.loc[df['condition']==1, 'condition2'] = 'Jokes'
df.loc[df['condition']==2, 'condition2'] = 'Assault'
df.loc[df['condition']==3, 'condition2'] = 'Control'


# In[37]:


df


# In[38]:


df.dtypes


# In[7]:


# make categorical
df['condition2'] = pd.Categorical(df['condition2'])
df.dtypes


# In[8]:


# check order
df['condition2']


# In[9]:


# relevel so Control is baseline category
# reorder
df['condition2'].cat.reorder_categories(['Control', 'Jokes', 'Assault'], inplace=True)
df['condition2ord'] = df['condition2']
df['condition2ord'].cat.reorder_categories(['Control', 'Jokes', 'Assault'], ordered=True, inplace=True)
# check order again
df['condition2']


# In[10]:


# and check order of ordinal version
df['condition2ord']


# In[11]:


# create new variable: pid3 ####

pd.crosstab(df['pid7'], df.pid7)
# notice two different ways of indexing


# In[12]:


df.pid3 = 'NaN'


# In[13]:


df.loc[df['pid7']=='Lean Democrat', 'pid3'] = 'Democrat'
df.loc[df['pid7']=='Strong Democrat', 'pid3'] = 'Democrat'
df.loc[df['pid7']=='Not very strong Democrat', 'pid3'] = 'Democrat'

df.loc[df['pid7']=='Lean Republican', 'pid3'] = 'Republican'
df.loc[df['pid7']=='Strong Republican', 'pid3'] = 'Republican'
df.loc[df['pid7']=='Not very strong Republican', 'pid3'] = 'Republican'

df.loc[df['pid7']=='Independent', 'pid3'] = 'Independent'
df.loc[df['pid7']=='Not sure', 'pid3'] = 'Independent'

# make categorical
df['pid3'] = pd.Categorical(df['pid3'])


# In[11]:


df


# In[14]:


# recode: punishment 

# punishment 1
df.loc[df.punishment_1=="Agree strongly", 'needmoreevidence'] = 5
df.loc[df.punishment_1=="Agree somewhat", 'needmoreevidence'] = 4
df.loc[df.punishment_1=="Neither disagree nor agree", 'needmoreevidence'] = 3
df.loc[df.punishment_1=="Disagree somewhat", 'needmoreevidence'] = 2
df.loc[df.punishment_1=="Disagree strongly", 'needmoreevidence'] = 1

pd.crosstab(df.needmoreevidence, df.needmoreevidence)


# In[ ]:





# In[ ]:





# In[ ]:


# punishment 2
dat$apology[dat$punishment_2=="Agree strongly"] <- 5
dat$apology[dat$punishment_2=="Agree somewhat"] <- 4
dat$apology[dat$punishment_2=="Neither disagree nor agree"] <- 3
dat$apology[dat$punishment_2=="Disagree somewhat"] <- 2
dat$apology[dat$punishment_2=="Disagree strongly"] <- 1

# punishment 3
dat$longtimeago[dat$punishment_3=="Agree strongly"] <- 5
dat$longtimeago[dat$punishment_3=="Agree somewhat"] <- 4
dat$longtimeago[dat$punishment_3=="Neither disagree nor agree"] <- 3
dat$longtimeago[dat$punishment_3=="Disagree somewhat"] <- 2
dat$longtimeago[dat$punishment_3=="Disagree strongly"] <- 1
table(dat$longtimeago)


# punishment 4
dat$resign[dat$punishment_4=="Agree strongly"] <- 5
dat$resign[dat$punishment_4=="Agree somewhat"] <- 4
dat$resign[dat$punishment_4=="Neither disagree nor agree"] <- 3
dat$resign[dat$punishment_4=="Disagree somewhat"] <- 2
dat$resign[dat$punishment_4=="Disagree strongly"] <- 1


# punishment 5
dat$elitecues[dat$punishment_5=="Agree strongly"] <- 5
dat$elitecues[dat$punishment_5=="Agree somewhat"] <- 4
dat$elitecues[dat$punishment_5=="Neither disagree nor agree"] <- 3
dat$elitecues[dat$punishment_5=="Disagree somewhat"] <- 2
dat$elitecues[dat$punishment_5=="Disagree strongly"] <- 1

# recode punishment: reverse codes
# need more evidence
dat$needmoreevidence_reverse[dat$punishment_1=="Agree strongly"] <- 1
dat$needmoreevidence_reverse[dat$punishment_1=="Agree somewhat"] <- 2
dat$needmoreevidence_reverse[dat$punishment_1=="Neither disagree nor agree"] <- 3
dat$needmoreevidence_reverse[dat$punishment_1=="Disagree somewhat"] <- 4
dat$needmoreevidence_reverse[dat$punishment_1=="Disagree strongly"] <- 5
table(dat$needmoreevidence_reverse, dat$needmoreevidence)

# long time ago
dat$longtimeago_reverse[dat$longtimeago==5] <- 1
dat$longtimeago_reverse[dat$longtimeago==4] <- 2
dat$longtimeago_reverse[dat$longtimeago==3] <- 3
dat$longtimeago_reverse[dat$longtimeago==2] <- 4
dat$longtimeago_reverse[dat$longtimeago==1] <- 5 
table(dat$longtimeago_reverse, dat$longtimeago)

# new variable: mean punitiveness score ####
dat$meanpunishment <- ((dat$apology+dat$resign+dat$needmoreevidence_reverse+dat$longtimeago_reverse)/4)


# In[ ]:





# In[ ]:





# In[15]:


## new variable: same party as legislator####
pd.crosstab(df.senator_party, df.senator_party)


# In[16]:


df.loc[((df['pid3'] == 'Democrat') & (df['senator_party'] == 'Democrat')) | 
       ((df.pid3=="Republican") and (df.senator_party=="Republican")), 'sameparty'] = 'Same party' 

df.loc[((df['pid3'] == 'Democrat') & (df['senator_party'] == 'Republican')) | 
       ((df.pid3=="Republican") and (df.senator_party=="Democrat")), 'sameparty'] = 'Opposite party' 

df.loc[df['pid3'] == 'Independent', 'sameparty'] = 'Independents/Not sures' 


# In[17]:


#make categorical
df.sameparty = pd.Categorical(df['sameparty'])


# In[20]:


df.dtypes


# In[18]:


# recode: pre sexism ####
# sexism_1,2,4 reverse coded

# create dictionaries for sexism
sexism_dict1 = { 'Agree strongly':5,
              'Agree somewhat':4,
               'Neither disagree nor agree':3,
               'Disagree somewhat':2,
               'Disagree strongly':1
              }
sexism_dict2 = { 'Agree strongly':1,
              'Agree somewhat':2,
               'Neither disagree nor agree':3,
               'Disagree somewhat':4,
               'Disagree strongly':5
              }

# use sexism_dict1 to recode 1, 2, and 4; use sexism_dict2 to recode 3
# see original R code from authors
df['pre_sexism_1new'] = df.pre_sexism_1.map(sexism_dict1)
df['pre_sexism_2new'] = df.pre_sexism_2.map(sexism_dict1)
df['pre_sexism_3new'] = df.pre_sexism_3.map(sexism_dict2) # using second dictionary here
df['pre_sexism_4new'] = df.pre_sexism_4.map(sexism_dict1)


# In[24]:


df


# In[26]:


df.dtypes


# In[19]:


# new variable: pre_sexism ####
df['pre_sexism'] = ((df['pre_sexism_1new'] + df['pre_sexism_2new'] + df['pre_sexism_3new'] + df['pre_sexism_4new'])/4)


# In[20]:


df


# In[29]:


df.dtypes


# In[ ]:





# In[21]:


# recode: post sexism ####
# sexism_1,2,4 reverse coded

# create same dictionaries as for pre_sexism

# use sexism_dict1 to recode 1, 2, and 4; use sexism_dict2 to recode 3
# see original R code from authors
df['post_sexism_1new'] = df.post_sexism_1.map(sexism_dict1)
df['post_sexism_2new'] = df.post_sexism_2.map(sexism_dict1)
df['post_sexism_3new'] = df.post_sexism_3.map(sexism_dict2) # using second dictionary here
df['post_sexism_4new'] = df.post_sexism_4.map(sexism_dict1)

# new variable: post_sexism ####
df['post_sexism'] = ((df['post_sexism_1new'] + df['post_sexism_2new'] + df['post_sexism_3new'] + df['post_sexism_4new'])/4)


# In[22]:


df


# In[25]:



### new variable: raw change from pretest to posttest ####
# favorability
df['change_favorability'] = (df.post_favorability+1) - (df.pre_favorability+1)

# vote
df['change_vote'] = (df.post_vote) - (df.pre_vote)

# sexism
df['change_sexism'] = (df.post_sexism) - (df.pre_sexism)

### new variable: percent change from pretest to posttest ##### favorability
df['perchange_favorability'] = (((df.post_favorability+1) - (df.pre_favorability+1))/(df.pre_favorability+1))*100

# vote
df['perchange_vote'] = (((df.post_vote+1) - (df.pre_vote+1))/(df.pre_vote+1))*100

# sexism
df['perchange_sexism'] = (((df.post_sexism+1) - (df.pre_sexism+1))/(df.pre_sexism+1))*100


# In[26]:


df


# In[27]:


df.dtypes


# In[29]:


# subset: without independents/notsures 
partydat = df[df["sameparty"] != 'Independents/Not sures']

# subset: people that share party with senator, people that do not share party with senator
samepartydat = df[df["sameparty"] != 'Same party']
opppartydat = df[df["sameparty"] != 'Opposite party']


# # Model 1

# In[31]:


# using statsmodels
m1 = sm.ols(formula="perchange_favorability ~ condition2", 
                data=samepartydat).fit()
m2 = sm.ols(formula="perchange_favorability ~ condition2", 
                data=opppartydat).fit()
m1.summary()


# In[32]:


m2.summary()


# In[35]:


coef_df = pd.DataFrame()
for i, mod in enumerate([m1, m2]):
    errorbars = mod.params - mod.conf_int()[0]
    coef_df = coef_df.append(pd.DataFrame({'coef': mod.params.values[1:],
                                           'err': errorbars.values[1:],
                                           'varname': errorbars.index.values[1:],
                                           'model': 'model %d'%(i+1)
                                          })
                            )
coef_df


# In[71]:


# Figure 1
marker_list = 'so'
width=.25
## 2 covariates in total
base_x = np.arange(2) - 0.2
base_x

fig, ax = plt.subplots(figsize=(8, 5))
for i, mod in enumerate(coef_df.model.unique()):
    mod_df = coef_df[coef_df.model == mod]
    mod_df = mod_df.set_index('varname').reindex(coef_df['varname'].unique())
    ## offset x posistions
    X = base_x + width*(i+1)
    ax.bar(X, mod_df['coef'],  
           color='none',yerr=mod_df['err'])
    ## remove axis labels
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.scatter(x=X, 
               marker=marker_list[i], s=120, 
               y=mod_df['coef'], color='black')
    ax.axhline(y=0, linestyle='--', color='black', linewidth=4)
    ax.xaxis.set_ticks_position('none')
    _ = ax.set_xticklabels(['', 'Assault', '', 'Jokes'], 
                           rotation=0, fontsize=16)


    
## finally, build customized legend
legend_elements = [plt.Line2D([0], [0], marker=m,  # added plt. before Line2D because Line2D not defined
                          label='Model %d'%(i+1),
                          color = 'k',
                          markersize=10)
                   for i, m in enumerate(marker_list)
                  ]
_ = ax.legend(handles=legend_elements, loc=2, 
              prop={'size': 15}, labelspacing=1.2)


# In[ ]:


## CONDITIONAL ON SEXISM ####

# favorability
model.sexism <- lm(perchange_favorability ~ condition2 + pre_sexism + condition2*pre_sexism, data=dat)
summary(model.sexism)

#### FIGURE 5 ####
sexism_fav_interplot <- interplot(m=model.sexism, var1="condition2", var2="pre_sexism", rfill="lightgrey", ralpha=.35, facet_labs = c("Assault", "Jokes")) + theme_bw() + 
  geom_line(size=.7) + 
  geom_hline(yintercept=0, linetype="dashed", colour="darkgrey") + 
  ylim(-60, 3) +
  labs(x="Sexism (1-5)", y="Treatment Effect", title="Conditional Effects of Sexism on Change in Favorability") +
  theme(axis.text=element_text(size=16),axis.title=element_text(size=18), plot.subtitle = element_text(size = 16), plot.title = element_text(size = 20), strip.text = element_text(size = 16))

sexism_fav_interplot


tiff(file="sexism_fav_plot.tiff", width=9, height=5.5, units="in", res=800)
sexism_fav_interplot
dev.off()


# In[35]:


m5 = sm.ols(formula="perchange_favorability ~ condition2 + pre_sexism + condition2*pre_sexism", 
                data=samepartydat).fit()
m5.summary()


# In[47]:


sns.lmplot(x="pre_sexism", y="perchange_favorability", col="condition2", data=df)


# In[54]:


m1.params


# In[56]:


m1.conf_int()


# In[62]:


errorbars = m1.params - m1.conf_int()[0]
errorbars


# In[63]:


# ignore intercept 
coef_df = pd.DataFrame({'coef': m1.params.values[1:],
                        'err': errorbars.values[1:],
                        'varname': errorbars.index.values[1:]
                       })
coef_df


# In[73]:


#plot
fig, ax = plt.subplots(figsize=(8, 5))
coef_df.plot(x='varname', y='coef', kind='bar', 
             ax=ax, color='none', 
             yerr='err', legend=False)
ax.set_ylabel('')
ax.set_xlabel('')
ax.scatter(x=np.arange(coef_df.shape[0]), 
           marker='s', s=120, 
           y=coef_df['coef'], color='black')
ax.axhline(y=0, linestyle='--', color='black', linewidth=4)
ax.xaxis.set_ticks_position('none')
_ = ax.set_xticklabels(['Assault', 'Jokes'#, 'Edu.', 'Catholic', 'Infant Mort.'
                       ], 
                       rotation=0, fontsize=16)


# In[ ]:





# In[49]:


# coefplot following this guide: https://zhiyzuo.github.io/Python-Plot-Regression-Coefficient/
#Collect coefficients
formula_1 = 'fertility ~ %s'%(" + ".join(df.columns.values[1:-1]))
print(formula_1)
mod_1 = smf.ols(formula_1, data=df).fit()
mod_1.params
fertility ~ agri + exam + edu + catholic

Intercept    91.055424
agri         -0.220646
exam         -0.260582
edu          -0.961612
catholic      0.124418
dtype: float64
formula_2 = 'fertility ~ %s'%(" + ".join(df.columns.values[1:-2].tolist() + ['infant_mort']))
print(formula_2)
mod_2 = smf.ols(formula_2, data=df).fit()
mod_2.params
fertility ~ agri + exam + edu + infant_mort

Intercept      68.773136
agri           -0.129292
exam           -0.687994
edu            -0.619649
infant_mort     1.307097
dtype: float64
coef_df = pd.DataFrame()
for i, mod in enumerate([mod_1, mod_2]):
    err_series = mod.params - mod.conf_int()[0]
    coef_df = coef_df.append(pd.DataFrame({'coef': mod.params.values[1:],
                                           'err': err_series.values[1:],
                                           'varname': err_series.index.values[1:],
                                           'model': 'model %d'%(i+1)
                                          })
                            )
coef_df
coef	err	model	varname
0	-0.220646	0.148531	model 1	agri
1	-0.260582	0.553176	model 1	exam
2	-0.961612	0.392609	model 1	edu
3	0.124418	0.075207	model 1	catholic
0	-0.129292	0.151049	model 2	agri
1	-0.687994	0.456646	model 2	exam
2	-0.619649	0.355803	model 2	edu
3	1.307097	0.820514	model 2	infant_mort
Plot!
## marker to use
marker_list = 'so'
width=0.25
## 5 covariates in total
base_x = pd.np.arange(5) - 0.2
base_x
array([-0.2,  0.8,  1.8,  2.8,  3.8])
fig, ax = plt.subplots(figsize=(8, 5))
for i, mod in enumerate(coef_df.model.unique()):
    mod_df = coef_df[coef_df.model == mod]
    mod_df = mod_df.set_index('varname').reindex(coef_df['varname'].unique())
    ## offset x posistions
    X = base_x + width*i
    ax.bar(X, mod_df['coef'],  
           color='none',yerr=mod_df['err'])
    ## remove axis labels
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.scatter(x=X, 
               marker=marker_list[i], s=120, 
               y=mod_df['coef'], color='black')
    ax.axhline(y=0, linestyle='--', color='black', linewidth=4)
    ax.xaxis.set_ticks_position('none')
    _ = ax.set_xticklabels(['', 'Agriculture', 'Exam', 'Edu.', 'Catholic', 'Infant Mort.'], 
                           rotation=0, fontsize=16)

    fs = 16
    ax.annotate('Control', xy=(0.3, -0.2), xytext=(0.3, -0.35), 
                xycoords='axes fraction', 
                textcoords='axes fraction', 
                fontsize=fs, ha='center', va='bottom',
                bbox=dict(boxstyle='square', fc='white', ec='black'),
                arrowprops=dict(arrowstyle='-[, widthB=6.5, lengthB=1.2', lw=2.0, color='black'))

    ax.annotate('Study', xy=(0.8, -0.2), xytext=(0.8, -0.35), 
                xycoords='axes fraction', 
                textcoords='axes fraction', 
                fontsize=fs, ha='center', va='bottom',
                bbox=dict(boxstyle='square', fc='white', ec='black'),
                arrowprops=dict(arrowstyle='-[, widthB=3.5, lengthB=1.2', lw=2.0, color='black'))
    
## finally, build customized legend
legend_elements = [Line2D([0], [0], marker=m,
                          label='Model %d'%i,
                          color = 'k',
                          markersize=10)
                   for i, m in enumerate(marker_list)
                  ]
_ = ax.legend(handles=legend_elements, loc=2, 
              prop={'size': 15}, labelspacing=1.2)


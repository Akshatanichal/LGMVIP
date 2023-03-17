#!/usr/bin/env python
# coding: utf-8
# # LetsGrowMore: Data Science
# ## Task02 (Intermesiate Level Task): Exploratory Data Analysis on Terrorism dataset
# ### Name of Intern:Akshata Naganath Nichal
#  importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
data=pd.read_csv("C:/Users/91986/Downloads/terrorism.csv",encoding="latin1")
data.head()
data.info()
data.shape
data.columns
for i in data.columns:
    print(i,end=",")
data=data[["iyear","imonth","iday","country_txt","region_txt","provstate","city","latitude","longitude","location","summary","attacktype1_txt","targtype1_txt","gname","motive","weapsubtype1_txt","nkill","nwound","addnotes"]]
data.head()
data.rename(columns={"iyear":"Year","imonth":"Month","iday":"Day","country_txt":"Country","region_txt":"Region","provstate":"province","city":"City","latitude":"Latitude","location":"Location","summary":"Summary","attacktype1_txt":"Attack Type","targtype1_txt":"Target Type","gname":"Group Name","motive":"Motive","weapsubtype1_txt":"Weapon Type","nkill":"Killed","nwound":"Wounded","addnotes":"Add Notes"},inplace=True)
data.head(10)
data.info()
data.isnull().sum()
# In[11]:
data["Killed"]=data["Killed"].fillna(0)
data["Wounded"]=data["Wounded"].fillna(0)
data["Casuality"]=data["Killed"]+data["Wounded"]
# In[12]:
data.describe()
# # Data Visualization
attacks=data["Year"].value_counts(dropna=False).sort_index().to_frame().reset_index().rename(columns={"index":"Year","Year":"Attacks"}).set_index("Year")
attacks.heaad(10)
attacks.plot(kind="bar",color="cornflowerblue",figsize=(15,6),fontsize=13)
plt.title("Timeline of Attaks",fontsize=15)
plt.xlabel("Years",fontsize=15)
plt.ylabel("Number of Attacks",fontsize=15)
plt.show()
# #### i)Most Number of attacks in 2014
# #### ii)Least Number of attacks in 1971
yc=data[["Year","Casuality"]].groupby("Year").sum()
yc.head()
yc.plot(kind="bar",color="cornflowerblue",figsize=(15,6),fontsize=13)
plt.title("Yearwise Casualities",fontsize=15)
plt.xlabel("Years",fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=15)
plt.show()
# #### Killed in each year
yk=data[["Year","Killed"]].groupby("Year").sum()
yk.head()
reg=pd.crosstab(data.Year,data.Region)
reg.head()
# #### Region Wise Attacks
# Distribution of Terrorist attacks over region from 1970-2017
reg.plot(kind="area",stacked=False,alpha=0.5,figsize=(20,10))
plt.title("Region wise attcks",fontsize=20)
plt.xlabel("Years",fontsize=20)
plt.ylabel("Number of Attacks",fontsize=20)
plt.show()
# Total Terrorist Attacks in each Region from 1970-2017
regt=reg.transpose()
regt["Total"]=regt.sum(axis=1)
ra=regt["Total"].sort_values(ascending=False)
ra
ra.plot(kind='bar',figsize=(15,6))
plt.title("Total Number of Attacks in each Region from 1970-2017")
plt.xlabel("Region")
plt.ylabel("Number of Attacks")
plt.show()
# Killed in each Region
rk=data[["Region","Killed"]].groupby("Region").sum().sort_values(by="Killed",ascending=False)
rk
rw=data[["Region","Wounded"]].groupby("Region").sum().sort_values(by="Wounded",ascending=False)
rw
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)
#killed
rk.plot(kind="bar",color="Cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each region")
ax0.set_xlabel("Region")
ax0.set_ylabel("Number of people Killed")
#Wounded
rw.plot(kind="bar",color="Cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Killed in each region")
ax1.set_xlabel("Region")
ax1.set_ylabel("Number of people Wounded")
plt.show()
# #### Country wise Attacks
# 1.Number of Attacks in each country
ct=data["Country"].value_counts().head(10)
ct
ct.plot(kind="bar",color="cornflowerblue",figsize=(15,6),fontsize=13)
plt.title("Country wise Attacks",fontsize=15)
plt.xlabel("Countries",fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=15)
plt.show()
# Total Casualities(Killed+Wounded)in each Country
cnc=data[["Country","Casuality"]].groupby("Country").sum().sort_values(by="Casuality",ascending=False)
cnc.head(10)
cnc[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Country wie Casualties",fontsize=13)
plt.xlabel("Countries",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()
# Killed in each Country
cnk=data[["Country","Killed"]].groupby("Country").sum().sort_values(by="Killed",ascending=False)
cnk.head(10)
#wounded in each Country
cnw=data[["Country","Wounded"]].groupby("Country").sum().sort_values(by="Wounded",ascending=False)
cnw.head(10)
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)
#Killed
cnk[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each Country")
ax0.set_xlabel("Countries")
ax0.set_ylabel("Number of People Killed")
#Wounded
cnw[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded in each Country")
ax1.set_xlabel("Countries")
ax1.set_ylabel("Number of People Wounded")
# ## City wise Attacks - Top 10
# Number of Attacks in each city
city=data["City"].value_counts()[1:11]
city
city.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("City wise Attacks",fontsize=13)
plt.xlabel("Cities",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()
# Total Casualties (Killed + Wounded) in each City
cc=data[["City","Casuality"]].groupby("City").sum().sort_values(by="Casuality",ascending=False).drop("Unknown")
cc.head(10)
cc[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("City wise Casualties",fontsize=13)
plt.xlabel("Cities",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()
# Killed in each City
ck=data[["City","Killed"]].groupby("City").sum().sort_values(by="Killed",ascending=False).drop("Unknown")
ck.head(10)
# Wounded in each City
cw=data[["City","Wounded"]].groupby("City").sum().sort_values(by="Wounded",ascending=False).drop("Unknown")
cw.head(10)
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)
#Killed
ck[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each City")
ax0.set_xlabel("Cities")
ax0.set_ylabel("Number of People Killed")
#Wounded
cw[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded in each City")
ax1.set_xlabel("Cities")
ax1.set_ylabel("Number of People Wounded")
# ## Terrorist Group wise Attacks - Top 10
# Number of Attacks by each Group
grp=data["Group Name"].value_counts()[1:10]
grp
grp.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Group wise Attacks",fontsize=13)
plt.xlabel("Terrorist Groups",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()
# Total Casualties(Killed + Wounded) by each Group
gc=data[["Group Name","Casuality"]].groupby("Group Name").sum().sort_values(by="Casuality",ascending=False).drop("Unknown")
gc.head(10)
gc.head(10).plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Casualties by each Group",fontsize=13)
plt.xlabel("Terrorist Groups",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()
# Killed by each Group
gk=data[["Group Name","Killed"]].groupby("Group Name").sum().sort_values(by="Killed",ascending=False).drop("Unknown")
gk.head(10)
# Wounded by each Group
gw=data[["Group Name","Wounded"]].groupby("Group Name").sum().sort_values(by="Wounded",ascending=False).drop("Unknown")
gw.head(10)
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)
#Killed
gk[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed by each Group")
ax0.set_xlabel("Terrorist Groups")
ax0.set_ylabel("Number of people Killed")
#Wounded
gw[:10].plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded by each Group")
ax1.set_xlabel("Terrorist Groups")
ax1.set_ylabel("Number of people Wounded")
plt.show()
# ## Attack Type wise Attacks
# Number of Attacks by each Attack Type
at=data["Attack Type"].value_counts()
at
at.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Types of Attacks",fontsize=13)
plt.xlabel("Attack Types",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()
# Total Casualties (Killed + Wounded) by each Attack Type
ac=data[["Attack Type","Casuality"]].groupby("Attack Type").sum().sort_values(by="Casuality",ascending=False)
ac
ac.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Casualties in each Attack",fontsize=13)
plt.xlabel("Attack Types",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()
# Killed by each Attack Type
ak=data[["Attack Type","Killed"]].groupby("Attack Type").sum().sort_values(by="Killed",ascending=False)
ak
# Wounded by each Attack Type
aw=data[["Attack Type","Wounded"]].groupby("Attack Type").sum().sort_values(by="Wounded",ascending=False)
aw
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)
#Killed
ak.plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax0)
ax0.set_title("People Killed in each Attack Type")
ax0.set_xlabel("Attack Types")
ax0.set_ylabel("Number of people Killed")
#Wounded
aw.plot(kind="bar",color="cornflowerblue",figsize=(15,6),ax=ax1)
ax1.set_title("People Wounded in each Attack Type")
ax1.set_xlabel("Attack Types")
ax1.set_ylabel("Number of people Wounded")
plt.show()
# ## Target Type wise Attacks
# Number of Attacks over each Target Type
ta=data["Target Type"].value_counts()
ta
ta.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Types of Targets",fontsize=13)
plt.xlabel("Target Types",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()
tc=data[["Target Type","Casuality"]].groupby("Target Type").sum().sort_values(by="Casuality",ascending=False)
tc
tc.plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Casualties in each Target Attack",fontsize=13)
plt.xlabel("Target Types",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Casualties",fontsize=13)
plt.show()
tk=data[["Target Type","Killed"]].groupby("Target Type").sum().sort_values(by="Killed",ascending=False)
tk
tw=data[["Target Type","Wounded"]].groupby("Target Type").sum().sort_values(by="Wounded",ascending=False)
tw
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)
#Killed
tk.plot(kind="bar",color="cornflowerblue",figsize=(17,6),ax=ax0)
ax0.set_title("People Killed in each Target Attack")
ax0.set_xlabel("Target Types")
ax0.set_ylabel("Number of people Killed")
#Wounded
tw.plot(kind="bar",color="cornflowerblue",figsize=(17,6),ax=ax1)
ax1.set_title("People Wounded in each Target Attack")
ax1.set_xlabel("Target Types")
ax1.set_ylabel("Number of people Wounded")
plt.show()
# ## Group + Country wise - Top10
# Sorting by number of Attacks
gca=data[["Group Name","Country"]].value_counts().drop("Unknown")
gca.head(10)
gca.head(10).plot(kind="bar",color="cornflowerblue",figsize=(15,6))
plt.title("Countries with most attacks by a particular Group",fontsize=13)
plt.xlabel("(Terrorist Group,Country)",fontsize=13)
plt.xticks(fontsize=12)
plt.ylabel("Number of Attacks",fontsize=13)
plt.show()
#  ## Humanity Affected (World-wide) by Terrorist Attacks from 1970 to 2017
# Total Casualties (Killed + Wounded) due to Terrorist Attacks
casuality=data.loc[:,"Casuality"].sum()
print("Total number of Casualties due to Terrorist Attacks from 1970 to 2017 across the world :\n",casuality)
# Killed due to Terrorist Attacks
kill=data.loc[:,"Killed"].sum()
print("Total number of people killed due to Terrorist Attacks from 1970 to 2017 across the world :\n",kill)
# Wounded due to Terrorist Attacks
wound=data.loc[:,"Wounded"].sum()
print("Total number of people killed due to Terrorist Attacks from 1970 to 2017 across the world :\n",wound)

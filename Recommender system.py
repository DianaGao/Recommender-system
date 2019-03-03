#/usr/bin/env python 3.6
# Mapper Reducer for mutual friends

import pandas as pd
import csv
import numpy as np
import math


# friends similarity
follower = pd.read_csv('path.csv')

#The reverse df sort by the first column category alphabetically
followee = follower[follower.columns[::-1]]
followee = followee.sort_values(by='followee_id')

# At last, the user_id can be defined by the person who uses this algorithm
# But now, it's subject to the user ids in the test set
user_idlist_f = np.unique(follower[['follower_id']].values)

# change the dfs into arrays
follower = follower.values
followee = followee.values


# interests similarity
user_i = pd.read_csv('path.csv')

#The reverse df sort by the first column category alphabetically
int_u = user_i[user_i.columns[::-1]]
int_u = int_u.sort_values(by='category')

# get the target_useridlist
user_idlist_i = np.unique(user_i[['user_id']].values)
user_idlist_i = user_idlist_i.astype('float').astype('U')

# change the initial interest dataframe into numpy array
user_i = user_i.values
int_u = int_u.values


def transform(arrdata):
    arrdata = np.concatenate((arrdata,np.array([[0,0]])), axis=0)
    counter = 0
    arrline = np.empty((0))
    arrfollower = list()
    for line in arrdata:
        if line[0] == arrdata[counter -1][0]:
            arrline = np.append(arrline,line[1])
        else:
            if counter != 0:
                arrfollower.append(arrline)
            arrline = np.empty((0))
            arrline = np.append(arrline,line[0])
            arrline = np.append(arrline,line[1])
        counter += 1
    return arrfollower


def friends(i, array):
    for row in array:
        if np.array(row[0]) == i:
            return row[1:]


# To map the target user and his second degree friends
# Return one line if they follow one common friends
def friends_mapper(user_ids, arrfollower, arrfollowee):
    counter = 0
    mapper = np.array(['0','0'])
    for i in user_ids:
        if friends(i, arrfollower) is None:
            continue
        for j in friends(i, arrfollower):
            if friends(j , arrfollowee) is None:
                continue
            for k in friends(j , arrfollowee):
                #if k == i or k in friends(i, arrfollower):
                if k == i:
                    continue
                pair = str("%s_%s" % (i, k))
                item = np.array([pair, int(1)])
                mapper = np.vstack((mapper,item))
    mapper = np.delete(mapper,0, 0)
    df = pd.DataFrame(mapper)
    df = df.sort_values(by=0)
    df = df.values
    return df


# This part for interests mapper

def frequency(array):
    freq_i = np.array(['interests', 0])
    prev_i = None
    count = 0
    for row in array:
        interest = row[0]
        if interest == prev_i:
            count += 1
        elif prev_i is None:
            count = 1
            prev_i = row[0]
        else:
            item = np.array([prev_i, count])
            freq_i = np.vstack((freq_i, item))
            count = 1
            prev_i = row[0]
    freq_i = np.delete(freq_i, 0, 0)
    return freq_i

# to identify the top 5% hot topics
def hot(freq_i):
    # convert the array to pandas df to better rank the hot topics
    dffreq_i = pd.DataFrame(freq_i)
    dffreq_i[1] = dffreq_i[1].astype(float).fillna(0.0)
    dffreq_i = dffreq_i.sort_values(by=[1], ascending= False)
    dffreq_i = dffreq_i.reset_index(drop=True)
    hot = dffreq_i.loc[0:round(dffreq_i.shape[0]*0.05)]
    hottopic = hot[0].values # get the hot topics in numpy array
    return hottopic


def interests_mapper(user_ids, arrfollower, arrfollowee, hottopic):
    hot = 0 # 0 - not hot topic   1 - hot topic
    mapper = np.array(['0','0'])
    hotmapper = np.array(['0','0'])
    for i in user_ids:
        if friends(i, arrfollower) is None:
            continue
        for j in friends(i, arrfollower): 
            hot = 0
            if j in hottopic:
                hot = 1
            for k in friends(j , arrfollowee):
                k = k.astype('float').astype('U')
                if k == i:
                    continue
                pair = str("%s_%s" % (i, k))
                item = np.array([pair, int(1)])
                hotitem = np.array([pair, hot])
                mapper = np.vstack((mapper,item))
                hotmapper = np.vstack((hotmapper,hotitem))
    mapper = np.delete(mapper,0, 0)
    hotmapper = np.delete(hotmapper,0, 0)
    df = pd.DataFrame(mapper)
    dfhot = pd.DataFrame(hotmapper)
    # sort by the first column to put similar pair together to pump into reducerr function
    df = df.sort_values(by=0)
    dfhot = dfhot.sort_values(by=0)
    df = df.values
    dfhot = dfhot.values
    return df, dfhot # return two numpy array sorted by the 1st column
  

# Reducer part
# Word Count Function, Reducer of Mapper.file
def reducer(mapper):
    reducer = np.array(['0','0'])
    prev_key = None
    key = None
    current_count = 0
    for line in mapper:
        key = line[0]
        count = line[1]
        count = int(count)
        if key == prev_key:
            current_count += count
        else:
            if prev_key:
                item = np.array([prev_key, current_count])
                reducer = np.vstack((reducer, item))
            current_count = count
            prev_key = key
    # store the last row in the mapper file 
    if key == prev_key:
        item = np.array([prev_key, current_count])
        reducer = np.vstack((reducer, item))
    reducer = np.delete(reducer,0, 0)
    return reducer



# Create a better array for recommendation : target, sd_user, comfriends
def clean(reducer):
    dfreducer = pd.DataFrame(reducer)
    dfreducer['target'] = dfreducer[0].str.split('_').str[0]
    dfreducer['sd_user'] = dfreducer[0].str.split('_').str[1]
    dfreducer['comfriends'] = dfreducer[1]
    dfreducer = dfreducer.drop([0,1], axis=1)
    dfreducer = dfreducer.values 
    # the array has string element
    cleanarr = dfreducer.astype(float)
    return cleanarr


# Normalize the common friends by dividing the number of total friends both users follow
def friends_normal(cleanarr, arrfollower):
    counter = 0
    for row in cleanarr:
        #sd_user means second degree friends - the candidates
        target_f = len(friends(row[0], arrfollower))
        sd_uder_f = len(friends(row[1], arrfollower))
        normalscore = np.round(row[2] / int(target_f + sd_uder_f), 7)
        row = np.append(row,normalscore)
        if counter == 0:
            arrnormal = row
        else:
            arrnormal = np.vstack((arrnormal,row))
        counter += 1
    farrnormal = arrnormal
    return farrnormal

# combine the interest array with hot topic array
def combine(array1, array2):
    hotscore = array2[:,2:3] 
    combine_iscore = np.hstack((array1, hotscore))
    return combine_iscore

# Interests Score Normalization
# return the normalized array: target, sd_user, cominterests, normalscore
def interests_normal(cleanarr, arrfollower):
    counter = 0
    for row in cleanarr:
        # calculate the number of topics that target user and recommend candidate likes
        target_i = len(friends(str(row[0]), arrfollower))
        sd_user_i = len(friends(str(row[1]), arrfollower))
        if row[3] != 0:
        # use Cosine similarity to normalize the interest score & add punishment factor on the hot topics
            normalscore = np.round((float(row[2] * (1/np.log(1+row[3])))  / math.sqrt(target_i * sd_user_i)) , 7)
        else:
            normalscore = np.round(row[2] / math.sqrt(target_i * sd_user_i) , 7)
        row = np.append(row,normalscore)
        # use Jaccard coefficient to calculate the normalize score
        # normalscore = np.round((float(row[2] * (1/np.log(1+row[3]))) / (target_i + sd_user_i)) , 7)
        if counter != 0:
            arrnormal = np.vstack((arrnormal,row))
        else:
            arrnormal = row
        counter += 1
    iarrnormal = arrnormal
    return iarrnormal


def match(farrnormal, iarrnormal):
    farrnormal = farrnormal[:,[0,1,3]]
    iarrnormal = iarrnormal[:,[0,1,4]]
    # change the user id into string
    df_f = pd.DataFrame(farrnormal, columns = ['P1','P2','S1'])
    df_f['P1'] = df_f.P1.map(int).map(str)
    df_f['P2'] = df_f.P2.map(int).map(str)
    df_i = pd.DataFrame(iarrnormal, columns = ['P1','P2','S2'])
    df_i['P1'] = df_i.P1.map(int).map(str)
    df_i['P2'] = df_i.P2.map(int).map(str)
    # pair the target_user and sd_user
    df_f['key'] = df_f.P1 + '_' + df_f.P2
    df_i['key'] = df_i.P1 + '_' + df_i.P2
    # merge the friends score array and interests score array
    full = df_f.merge(df_i, left_on ='key', right_on ='key', how = 'left')[['P1_x','P2_x','S1','S2']]
    full = full.fillna(0)
    full['S3'] = full['S1'] + full['S2'] 
    full.rename(columns = {'P1_x':'Target_User','P2_x':'Sd_User'}, inplace=True)
    #select the top-5 users with highest final score
    full.sort_values('S3',ascending=False, inplace = True)
    full = full.groupby('Target_User').head(5).sort_values('Target_User')
    return full


def validate(full):
    arrfull = full.values[:,:2].astype('float')
    arrfull = transform(arrfull)
    total = 0
    for row in arrfull:
        # use the Precision Evaluation Method
        tp = np.intersect1d(friends(row[0], arrfull),friends(row[0],arrfollower))
        precision = np.round(len(tp)/5,5)
        total = total + precision
    average = total / len(arrfull)
    print(average)



if __name__ =="__main__":
    # friends similarity
    user_ids_f = user_idlist_f
    #mapper
    arrfollower = transform(follower)
    arrfollowee = transform(followee)
    df_f = friends_mapper(user_ids_f, arrfollower, arrfollowee)
    #reducer
    reducer_f = reducer(df_f)
    cleanarr_f = clean(reducer_f)
    #normalize
    np.set_printoptions(suppress = True)
    farrnormal = friends_normal(cleanarr_f, arrfollower)

    # interest similarity
    user_ids_i = user_idlist_i
    # mapper
    arruser_i = transform(user_i)
    arrint_u = transform(int_u)
    arrfreq = frequency(int_u)
    hottopic = hot(arrfreq)
    df_i, dfhot = interests_mapper(user_ids_i, arruser_i, arrint_u, hottopic)
    # reducer 
    reducer_i = reducer(df_i) # should be reducerr
    cleanarr_i = clean(reducer_i)
    reducerhot = reducer(dfhot) # should be reducerr here
    cleanarrhot = clean(reducerhot)
    # combine
    combine_iscore = combine(cleanarr_i, cleanarrhot)
    np.set_printoptions(suppress = True)
    iarrnormal = interests_normal(combine_iscore, arruser_i)

    #final matching two scores
    full = match(farrnormal, iarrnormal)
    validate(full)





import numpy as np
import tarfile
import json
import pandas as pd
import matplotlib.pyplot as mpl
from matplotlib import rcParams
# import matplotlib.markers as markers
from scipy import stats
import scikit_posthocs as ph
import datetime
import ast
import math
import requests
import seaborn as sns
from decimal import Decimal

# from urlparse import urlparse
from threading import Thread
import httplib2, sys, urllib
import queue


rcParams['axes.titlepad'] = 10
rcParams['figure.subplot.hspace'] = 0.95

startTime = datetime.datetime.now()


# top100 = open('top-100.json')
# top100 = json.load(top100)
# top100 = list(top100.values())
# print("unique top 100 songs: ",len(top100))

i = 0
lim = 40000
b = True

d= {}

uniqueGenres = set()

# data = [None]*lim


jsonTypeMap = {
  "lowlevel":
  {"average_loudness":"num","barkbands":"d2","dissonance":"d1", "dynamic_complexity":"num", "melbands": "d2","mfcc":"d2","pitch_salience":"d1","spectral_centroid":"d1","spectral_energy":"d1","spectral_energyband_high":"d1","spectral_energyband_low":"d1","spectral_energyband_middle_high":"d1","spectral_energyband_middle_low":"d1","spectral_entropy":"d1","spectral_flux":"d1","spectral_kurtosis":"d1","spectral_rms":"d1","spectral_rolloff":"d1","spectral_skewness":"d1","spectral_spread":"d1","spectral_strongpeak":"d1"},
  "metadata":{"tags":'d2', "audio_properties":"d1"},
  "rhythm":{"beats_count":"num", "bpm":"num"},
  "tonal":
  {"chords_changes_rate":"num", "chords_histogram":"list", "chords_key":"str", "chords_scale":"str", "key_key":"str", "key_scale":"str", "key_strength":"num"}
}

def cleanGenres(s):
    if (s is None):
        return None
    r = None
    if (',' in s):
        r = s.split(',')
    elif (';' in s):
        r = s.split(";")
    elif ('/' in s):
        r = s.split('/')
    elif ('\\' in s):
        r = s.split('\\')
    else:
        r = [s]
    if r != None:
        for i in range(len(r)):
            r[i] = r[i].lstrip().rstrip().lower()
    return r



def safe_first(l):
    if (type(l) != list):
        return l
    else:
        if (len(l)<1):
            return None
        else:
            return safe_first(l[0])

def column(matrix, i):
    return [row[i] for row in matrix]

# For comparing title and artist names
def compare(s1,s2):
    if (s1.lower()==s2.lower()):
        return True

    s1 = s1.lower().replace(" ","")
    s2 = s2.lower().replace(" ","")

    if(s1==s2):
        return True
    return False

# generates csv
def generateTop100Csv (outputURL='top100-high-level.csv',lim=None,startAt=0):

    top100 = open('top-100.json')
    top100 = json.load(top100)
    top100 = list(top100.values())
    print("unique top 100 songs: ",len(top100))

    hasPutHeader = False;
    count = 0;
    tar = tarfile.open("high-level.tar.bz2",mode="r:bz2")

    for member in tar:
        count += 1
        if(count-1<startAt):
            continue;
        f=tar.extractfile(member)

        s =  f.read().decode('utf-8')
        content = json.loads(s)
        mbid = safe_first(content.get("metadata",d).get('tags',d).get('musicbrainz_recordingid'))

        title = safe_first(content.get('metadata',d).get('tags',d).get('title'))
        artist = safe_first(content.get('metadata',d).get('tags',d).get('artist'))

        if(artist!=None and title != None):
            for i in top100:
                if compare(i['title'],title):
                    if compare(artist, i['artist']):

                        # output[mbid] = content
                        top100.remove(i)
                        with open(outputURL, 'a') as outfile:
                            highlevel = content.get('highlevel')

                            if(hasPutHeader==False):
                                headerStr = "mbid, title(no commas), artist(no commas),"
                                for i in highlevel:
                                    for j in highlevel[i].get('all'):
                                        headerStr = headerStr +i+":"+j+","
                                headerStr += "\n"
                                outfile.write(headerStr)
                                hasPutHeader = True


                            s = mbid+", \""+title.replace(",","")+"\", \""+artist.replace(",","")+"\","
                            for i in highlevel:
                                s=s+dictToCSV(highlevel[i].get('all'))

                            s = s +"\n"
                            outfile.write(s);
                            outfile.close()
                            print("saved: ",title," - ",artist, " on iteration ",count, " ~",count/1800000,"%")

        if lim != None:
            lim -= 1
            if (lim<=0):
                print("breaking on lim")
                break


def dictToCSV (d):
    s = ""
    for i in d:
        s = s+str(d[i])+","
    return s


def removeEndRowCommas(inputCsvPath, outputCsvPath):
    with open(inputCsvPath, 'r') as inputFile:
        with open(outputCsvPath, "a") as outputFile:
            for i in inputFile:
                outputFile.write(i[:-2]+"\n")
            outputFile.close()
        inputFile.close();


def generateMetadataCsv(inputCsvPath="", inputJsonPath="top-100.json", outputCsvPath="",startAt=0):
    csv = pd.read_csv(inputCsvPath, sep=",",header=0)

    t = open('top-100.json')
    t = json.load(t)
    top100 = {}
    for i in t:
        top100[i.lower()] = t[i]
    # top100 = list(top100.values())

    outfile = open(outputCsvPath,"a", encoding="utf-8")

    tar = tarfile.open("high-level.tar.bz2",mode="r:bz2")
    tags_fields = ["date","genre","originaldate","releasecountry","musicbrainz album release country","tracknumber","label"]
    audio_properties_fields = ["sample_rate","replay_gain","length","downmix","codec","bit_rate","equal_loudness","lossless"]

    hasPutHeader = False;
    mbids = csv.as_matrix()[:,0]
    peakErrs = []
    tryErrs = []
    count = 0
    for member in tar:
        try:
            count += 1
            if(count-1<startAt):
                continue;

            f=tar.extractfile(member)
            mbid = member.name[-43:-7]
            index = np.argwhere(mbids==mbid).flatten()
            if (index.shape[0] > 0):
                index = index[0]
                mbids = np.delete(mbids,index)
                print(mbid, "  |  iteration: ",count,"  -  ", count*100/1800000, "% finished")
                content = json.loads(tar.extractfile(member).read().decode('utf-8'))
                metadata = content.get('metadata')
                title = safe_first(content.get('metadata',{}).get('tags',{}).get('title'))
                artist = safe_first(content.get('metadata',{}).get('tags',{}).get('artist'))
                audio_properties = metadata.get('audio_properties')
                tags = metadata.get('tags')

                if(hasPutHeader==False):
                    headerStr = "mbid, title(no commas), artist(no commas), peakPos,"
                    for i in audio_properties_fields:
                        headerStr += "audio_properties:"+i+","
                    for i in tags_fields:
                        headerStr += "tags:" + i + ","

                    headerStr = headerStr[:-1]+"\n"
                    outfile.write(headerStr)
                    hasPutHeader = True

                if(title==None):
                    title =""
                if(artist==None):
                    artist=""


                s = mbid+", \""+title.replace(",","")+"\", \""+artist.replace(",","")+"\","
                peakPos = top100.get(title.lower() +" - "+artist.lower(),{}).get('peakPos',"")
                if(peakPos==""):
                    print("********************could not find peakPos for:  ",title," - ",artist)
                    peakErrs.append(mbid)
                s += str(peakPos)+","

                for i in audio_properties_fields:
                    s += str(audio_properties.get(i,""))+","
                for i in tags_fields:
                    s += str(tags.get(i,"")).replace(",","|")+","
                    # print(str(tags[i]).replace(",","|"))
                    # s+= str(tags[i]).replace(",","|") +","
                s = s[:-1]+"\n"
                # print(s)
                outfile.write(s)
                # try:
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            print("on count: ",count)
            outfile.close()
            return;
        except:
            print("*********@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@error on: ",count);
            tryErrs.append(count)

    print("peak errors on ", len(errs)," songs")
    print(errs)
    print('try errors on ',len(tryErrs)," songs")
    print(tryErrs)
    outfile.close()

def calculateBasicStats (csvPath, feature, replaceEmpty=False):
    csv = pd.read_csv(csvPath, sep=",",header=0)
    if replaceEmpty:
        csv.replace("",np.nan,inplace=True)
        csv.dropna(subset=[feature],inplace=True)
    data = csv.loc[:,feature]
    print("\nSummary stats: ",feature)
    print("Total mean     : ",data.mean())
    print("Total variance : ",data.var())
    return

def plotBasicStats (csvPath, features, replaceEmpty=False):
    csv = pd.read_csv(csvPath, sep=",",header=0)
    # if replaceEmpty:
        # csv.replace("",np.nan,inplace=True)
        # csv.dropna(subset=[feature],inplace=True)
    data = csv.loc[:,features]
    data.boxplot()
    mpl.show()
    return




def cleanDate(s):
    s = str(s)
    if (s == None or s == ""):
        return np.datetime64("NaT")
    try:
        s = ast.literal_eval(s)
        if(type(s)==list):
            s = s[0]
    except:
        r = np.datetime64("NaT")
    try:
        s = s.replace(" ","")
        r = pd.to_datetime(s,utc=True).date();
        if r.year >2020:
            r = r - pd.OffsetDate(years=100)
        # print('returned pd date: ',r)
    except:
        return np.datetime64("NaT")
    return r

def isNaT(x):
    return (np.isnat(x.to_datetime64()))

def getTimeSeries(features,csvUrl='top100-high-level-complete.csv', fillMissingOriginalDate=True):
    csv = pd.read_csv(csvUrl, sep=",",header=0)
    features = features+['tags:originaldate','tags:date']
    data = csv.loc[:,features]

    data['tags:originaldate'] = data['tags:originaldate'].apply(cleanDate)
    data['tags:date'] = data['tags:date'].apply(cleanDate)

    if (fillMissingOriginalDate):
        data['tags:originaldate'] = data['tags:date'].where(data['tags:originaldate'].map(isNaT),other=data['tags:originaldate'])

    data = data[data['tags:originaldate'].map(lambda a: not isNaT(a))]
    return data.loc[:,features]



def dateLinRegress(features, csv,dateRange=('1900','2020'),plot=False):
    data = getTimeSeries(features,csv)
    dateRange = (pd.to_datetime(str(dateRange[0])), pd.to_datetime(str(dateRange[1])))
    # x = data['tags:originaldate'].map(lambda a: a.year).as_matrix()
    data = data[data['tags:originaldate']<=dateRange[1]]
    data = data[dateRange[0] <=data ['tags:originaldate']]
    f = None
    if plot:
        f, axes = mpl.subplots(nrows = math.ceil(len(features)/2), ncols = 2,figsize=(10,30))
    for i in range(len(features)):
        print(features[i]," :")
        t = data[pd.notnull(data[features[i]])]
        x = t['tags:originaldate'].map(lambda a: a.year).values
        y = t[features[i]].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        if plot:
            axes[i//2][i%2].scatter(x,y,s=0.01)
            axes[i//2][i%2].set_title(features[i])
        print("slope: ",slope, "  p: ",p_value)
        print('_____________\n')
    if plot:
        mpl.show()




def timeRangeKruskal(features,csv="top100-high-level-complete.csv",timeWindowSize=10, plot=False,dateRange=(1950,2010)):
    data = getTimeSeries(features,csv)
    # [ timerange( [ features(mean,var) ] ) ]
    alpha = 0.001
    data['tags:originaldate'] = data['tags:originaldate'].map(lambda a: math.floor(a.year/timeWindowSize)*timeWindowSize)
    data = data[data['tags:originaldate']<=dateRange[1]]
    data = data[dateRange[0] <=data ['tags:originaldate']]
    print("\n")
    count = 0
    for i in features:
        # mod = sm.OLS(i+" ~ tags:originaldate", data=data)
        print("| ",i,": ")
        clean_data = data[pd.notnull(data[i])]
        unique_years = pd.unique(clean_data['tags:originaldate'])
        unique_years.sort()

        l = pd.DataFrame()
        l2 = []
        for j in unique_years:
            t= clean_data[clean_data['tags:originaldate']==j]
            t= t[i]
            l[j] = t
            l2.append(t)
        # print(l2[:5])
        kruskal = stats.kruskal(*l2)
        post_hoc = ph.posthoc_dunn(clean_data,group_col="tags:originaldate",val_col=i)
        dunn_significant = set()
        for y1 in range(post_hoc.shape[0]):
            for y2 in range(y1):
                if post_hoc.iloc[y1,y2] < alpha and y1 != y2:
                    dunn_significant.add((post_hoc.index.values[y1],post_hoc.columns.values[y2]))

        print("| ",kruskal)
        print("|  significant pairs: ",dunn_significant)
        print("| ",post_hoc)
        print("| ",(post_hoc<alpha).values.flatten().sum())
        print("|___________________\n")
        if plot:
            # f, axes = mpl.subplots(nrows=1, ncols = 2, constrained_layout=True)
            # for j in unique_years:
                # clean_data.boxplot(i,by="tags:originaldate", ax=axes[1])
            a=clean_data.boxplot(i,by="tags:originaldate")
            a.set_title(i,fontsize=20)


            # axes[1].set_title("")
            # axes[1].set_xlabel("")
            # for j in unique_years:
                # sns.distplot(clean_data[clean_data['tags:originaldate']==j].loc[:,i],label=str(j),bins=40,norm_hist=True,ax=axes[0])
            # axes[0].legend(loc="upper right")
            # axes[0].set_xlabel("")
            # mpl.suptitle(i)
            mpl.gcf().subplots_adjust(bottom=0.25)

            dunnStr = "P<" +str(alpha)+ " (Dunn post-hoc): "
            c = len(dunnStr);
            for i in str(dunn_significant):
                c +=1
                dunnStr+=i
                if (c%55==0):
                    dunnStr += "\n"
            mpl.xlabel(dunnStr, fontsize=14)
        count+=1
    mpl.show()



def timeRangeAnova(features,csv="top100-high-level-complete.csv",timeWindowSize=10, plot=False,dateRange=(1950,2010)):
    data = getTimeSeries(features,csv)
    # [ timerange( [ features(mean,var) ] ) ]
    alpha = 0.05
    data['tags:originaldate'] = data['tags:originaldate'].map(lambda a: math.floor(a.year/timeWindowSize)*timeWindowSize)
    data = data[data['tags:originaldate']<=dateRange[1]]
    data = data[dateRange[0] <=data ['tags:originaldate']]
    print("\n")
    count = 0
    for i in features:
        # mod = sm.OLS(i+" ~ tags:originaldate", data=data)
        print("| ",i,": ")
        clean_data = data[pd.notnull(data[i])]
        unique_years = pd.unique(clean_data['tags:originaldate'])
        unique_years.sort()

        l = pd.DataFrame()
        l2 = []
        for j in unique_years:
            t= clean_data[clean_data['tags:originaldate']==j]
            t= t[i]
            l[j] = t
            l2.append(t)
        # print(l2[:5])
        kruskal = stats.f_oneway(*l2)
        # post_hoc = ph.posthoc_dunn(clean_data,group_col="tags:originaldate",val_col=i)

        post_hoc = ph.posthoc_tukey_hsd(clean_data[i], clean_data['tags:originaldate'], alpha=alpha)

        significant = set()
        for y1 in range(post_hoc.shape[0]):
            for y2 in range(y1):
                if post_hoc.iloc[y1,y2] == 1:
                    significant.add((post_hoc.index.values[y1],post_hoc.columns.values[y2]))

        print("| ",kruskal)
        print("|  significant pairs: ",significant)
        print("| ",post_hoc)
        print("| ",(post_hoc<alpha).values.flatten().sum())
        print("|___________________\n")
        if plot:
            # f, axes = mpl.subplots(nrows=1, ncols = 2, constrained_layout=True)
            # for j in unique_years:
                # clean_data.boxplot(i,by="tags:originaldate", ax=axes[1])
            a=clean_data.boxplot(i,by="tags:originaldate")
            a.set_title(i,fontsize=20)


            # axes[1].set_title("")
            # axes[1].set_xlabel("")
            # for j in unique_years:
                # sns.distplot(clean_data[clean_data['tags:originaldate']==j].loc[:,i],label=str(j),bins=40,norm_hist=True,ax=axes[0])
            # axes[0].legend(loc="upper right")
            # axes[0].set_xlabel("")
            # mpl.suptitle(i)
            mpl.gcf().subplots_adjust(bottom=0.25)

            sigStr = "P<" +str(alpha)+ " (Tukey HSD post-hoc): "
            c = len(sigStr);
            for i in str(significant):
                c +=1
                sigStr+=i
                if (c%55==0):
                    sigStr += "\n"
            mpl.xlabel(sigStr, fontsize=14)
        count+=1
    mpl.show()







def peakPosKruskal(features,csv="top100-high-level-complete.csv",popSize=10, plot=False,dateRange=(1950,2010)):
    features.append(" peakPos")
    features= list(set(features))
    data = getTimeSeries(features,csv)
    data = data[pd.notnull(data[' peakPos'])]
    alpha = 0.001

    data[' peakPos'] = data[' peakPos'].map(lambda a: math.floor(a/popSize)*popSize)

    print("\n")
    count = 0
    for i in features:
        # mod = sm.OLS(i+" ~ tags:originaldate", data=data)
        print("| ",i,": ")
        clean_data = data[pd.notnull(data[i])]
        unique_pos = pd.unique(clean_data[' peakPos'])
        unique_pos.sort()

        l = pd.DataFrame()
        l2 = []
        for j in unique_pos:
            t= clean_data[clean_data[' peakPos']==j]
            t= t[i]
            l[j] = t
            l2.append(t)

        kruskal = stats.kruskal(*l2)
        post_hoc = ph.posthoc_dunn(clean_data,group_col=" peakPos",val_col=i)
        dunn_significant = set()
        for y1 in range(post_hoc.shape[0]):
            for y2 in range(y1):
                if post_hoc.iloc[y1,y2] < alpha and y1 != y2:
                    dunn_significant.add((post_hoc.index.values[y1],post_hoc.columns.values[y2]))

        print("| ",kruskal)
        print("|  significant pairs: ",dunn_significant)
        print("| ",post_hoc<alpha)
        print("|___________________\n")
        if plot:
            f, axes = mpl.subplots(nrows=1, ncols = 1, constrained_layout=True)
            for j in unique_pos:
                clean_data.boxplot(i,by=" peakPos", ax=axes)
            # clean_data.boxplot(i,by=" peakPos")

            axes.set_title("")
            axes.set_xlabel("")
            # for j in unique_pos:
                # sns.distplot(clean_data[clean_data[' peakPos']==j].loc[:,i],label=str(j),bins=40,norm_hist=True,ax=axes[0])
            # axes[0].legend(loc="upper right")
            # axes[0].set_xlabel("")
            mpl.suptitle(i)
            mpl.gcf().subplots_adjust(bottom=0.25)

            dunnStr = "P<" +str(alpha)+ " (Dunn post-hoc): "
            c = len(dunnStr);
            for i in str(dunn_significant):
                c +=1
                dunnStr+=i
                if (c%55==0):
                    dunnStr += "\n"
            mpl.xlabel(dunnStr, fontsize=14)
        count+=1
    mpl.show()


def peakPosAnova(features,csv,popSize=10, plot=False,dateRange=(1950,2010)):
    data = getTimeSeries([' peakPos']+features,csv)
    print(pd.notnull(data[' peakPos']))
    data = data[pd.notnull(data[' peakPos'])]
    print(data[:5])

    # # features.append(" peakPos")
    # # features= list(set(features))
    # print(csv)
    # data = getTimeSeries(features,csv)
    # features = features+[' peakPos']
    # data = getTimeSeries(features ,csv)
    # alpha = 0.001
    #
    # print(data[:5])
    # print("&&&&&&&&")
    # data[' peakPos'] = data[' peakPos'].map(lambda a: math.floor(a/popSize)*popSize)
    # print(data[:5])
    # print("&&&&&&&&")
    # print("\n")
    # count = 0
    # for i in features:
    #     # mod = sm.OLS(i+" ~ tags:originaldate", data=data)
    #     print("| ",i,": ")
    #     clean_data = data[pd.notnull(data[i])]
    #     print(clean_data[:5])
    #     unique_pos = pd.unique(clean_data[' peakPos'])
    #     unique_pos.sort()
    #     print("fuck")
    #     print(unique_pos) # &&
    #     # print(unique_pos)
    #     l = pd.DataFrame()
    #     l2 = []
    #     for j in unique_pos:
    #         t= clean_data[clean_data[' peakPos']==j]
    #         t= t[i]
    #         l[j] = t
    #         l2.append(t)
    #     anova = stats.f_oneway(*l2)
    #     post_hoc = ph.posthoc_tukey_hsd(clean_data[i], clean_data[' peakPos'], alpha=alpha)
    #     # post_hoc = ph.posthoc_tukey_hsd(clean_data,group_col=" peakPos",val_col=i)
    #     significant = set()
    #     for y1 in range(post_hoc.shape[0]):
    #         for y2 in range(y1):
    #             if post_hoc.iloc[y1,y2] < alpha and y1 != y2:
    #                 significant.add((post_hoc.index.values[y1],post_hoc.columns.values[y2]))
    #
    #     print("| ",anova)
    #     print("|  significant pairs: ",significant)
    #     print("| ",post_hoc<alpha)
    #     print("|___________________\n")
    #     if plot:
    #         f, axes = mpl.subplots(nrows=1, ncols = 1, constrained_layout=True)
    #         for j in unique_pos:
    #             clean_data.boxplot(i,by=" peakPos", ax=axes)
    #         # clean_data.boxplot(i,by=" peakPos")
    #
    #         axes.set_title("")
    #         axes.set_xlabel("")
    #         # for j in unique_pos:
    #             # sns.distplot(clean_data[clean_data[' peakPos']==j].loc[:,i],label=str(j),bins=40,norm_hist=True,ax=axes[0])
    #         # axes[0].legend(loc="upper right")
    #         # axes[0].set_xlabel("")
    #         mpl.suptitle(i)
    #         mpl.gcf().subplots_adjust(bottom=0.25)
    #
    #         sigStr = "P<" +str(alpha)+ " (Tukey post-hoc): "
    #         c = len(sigStr);
    #         for i in str(significant):
    #             c +=1
    #             sigStr+=i
    #             if (c%55==0):
    #                 sigStr += "\n"
    #         mpl.xlabel(sigStr, fontsize=14)
    #     count+=1
    # mpl.show()


jsonTypeMap = {
  "lowlevel":
  {"average_loudness":"num","dissonance":"d1", "dynamic_complexity":"num","pitch_salience":"d1","spectral_centroid":"d1","spectral_energy":"d1","spectral_energyband_high":"d1","spectral_energyband_low":"d1","spectral_energyband_middle_high":"d1","spectral_energyband_middle_low":"d1","spectral_entropy":"d1","spectral_flux":"d1","spectral_kurtosis":"d1","spectral_rms":"d1","spectral_rolloff":"d1","spectral_skewness":"d1","spectral_spread":"d1","spectral_strongpeak":"d1"},
  "rhythm":{"beats_count":"num", "bpm":"num"},

  "tonal": {"chords_changes_rate":"num", "chords_histogram":"list", "chords_key":"str", "chords_scale":"str", "key_key":"str", "key_scale":"str", "key_strength":"num"},
}




def parseByFormat(jsonVal, format, t):
    s = ""
    h = ""
    if format == "num":
        s = str(jsonVal)+","
        h = t+","
    elif format == "d1":
        for k in jsonVal:
            s = s+str(jsonVal[k])+","
            h = h + t+":" + k + ","
    elif format == "d2":
        for k in jsonVal:
            for i in range(len(jsonVal[k])):
                s = s + str(jsonVal[k][i])+","
                h = h+t+":"+ k+":"+str(i) +","
    elif format == "d3":
        for k in jsonVal:
            if (k!="file_name"):
                for i in range(len(jsonVal[k])):
                    s = s + str(jsonVal[k][i]).replace(","," ")+","
                    h = h+t+":"+ k+":"+str(i) +","
    elif format == "list":
        for i in range(len(jsonVal)):
            s = s+str(jsonVal[i]) +","
            h = h + t + ":" + str(i) + ","
    elif format == "str":
        s = str(jsonVal)+","
        h = h + t+","
    else:
        print("***********warning parsing by unknown type: ",format)
        s = None
    return (s,h)


def appendJSONToLowlevelCSV(csvURL, jsonVal):
    # print(jsonVal.keys())
    # wroteHeader = False
    with open(csvURL,"a", encoding="utf-8") as outfile:
        header = ["mbid",""]
        for song in jsonVal:
            s = song + ","
            # h = ""
            for type in jsonTypeMap: # lowlevel, metadata, etc...
                for feature in jsonTypeMap[type]: # spectra_centroid, avg.loudness,etc...
                    temp  = parseByFormat(jsonVal[song]['0'][type][feature],jsonTypeMap[type][feature],type+":"+feature)
                    s = s + temp[0]
                    # h = h + temp[1]
            tags = jsonVal[song]['0'].get('metadata',{}).get('tags')

            for metadata_property in ["date","genre","originaldate",'title','tracknumber']:
                val = tags.get(metadata_property,[""])
                s = s + str(val[0]).replace(",","")+","
                # h = h + "metadata:tags:"+metadata_property+","
            # h=h +"\n"
            # if wroteHeader == False:
                # print("okay...")
                # print(h)
                # wroteHeader = True
            s = s +"\n"
            outfile.write(s)
    return None




def pull():
    concurrent = 50
    csv = pd.read_csv('top100-high-level-complete.csv')
    mbids = list(csv['mbid'])
    def doWork():
        while True:
            mbid = q.get()
            r = requests.get('https://acousticbrainz.org/api/v1/low-level',params={"recording_ids":[mbid]})
            jsonVal = r.json()
            # print(jsonVal.keys())
            appendJSONToLowlevelCSV("top100-low-level-complete.csv", jsonVal)
            q.task_done()
    cc = 0
    q = queue.Queue(concurrent * 2)
    for i in range(concurrent):
        t = Thread(target=doWork)
        t.daemon = True
        t.start()
    try:
        for mbid in mbids:
            q.put(mbid)
            cc+=1
            print("iter: ",cc,"  -  ",cc/15000,"% done")
            q.join()
    except KeyboardInterrupt:
        sys.exit(1)


def yearCoverage(csvURL,timeWindowSize=10):
    data = getTimeSeries(["metadata:tags:title"],csvURL)
    data['tags:originaldate'] = data['tags:originaldate'].map(lambda a: math.floor(a.year/timeWindowSize)*timeWindowSize)
    min =data['tags:originaldate'].min()
    max = data['tags:originaldate'].max()
    a = data.hist(bins=max-min)
    a[0][0].set_title("")
    mpl.suptitle("Coverage of Billboard Top 100 in AcoustirBrainz High-level Dataset")

    mpl.show()

def peakPosCoverage(csvURL,windowSize = 1):
    data = getTimeSeries([" peakPos"],csvURL)
    data = data[pd.notnull(data[' peakPos'])]
    data[' peakPos'] = data[' peakPos'].map(lambda a: math.floor(a/windowSize)*windowSize)

    top10 = data[data[' peakPos']<=10]
    print("10:  ",top10.shape)

    data = data[' peakPos']
    data.hist(bins=100)
    mpl.show()

def ugh():
    csv1 = pd.read_csv("top100-high-level-commas-removed.csv", sep=",",header=0).as_matrix()[:,0]
    csv2 = pd.read_csv("top100-high-level-metadata.csv", sep=",",header=0).as_matrix()[:,0]
    print(csv2.shape)
    print(csv1.shape)
    i=0;
    cond = True
    while (cond):
        if (csv1[i]!=csv2[i]):
            print(i)
            print(csv1[i],"  ",csv2[i])
            cond=False
            i +=1
            print(csv1==csv2)


def ampDb (amp):
    return 20*Math.log(amp)


def dateLinRegress2(features, csv,dateRange=('1900','2020'),plot=False):
    data = getTimeSeries(features,csv)
    dateRange = (pd.to_datetime(str(dateRange[0])), pd.to_datetime(str(dateRange[1])))
    # x = data['tags:originaldate'].map(lambda a: a.year).as_matrix
    data = data[data['tags:originaldate']<=dateRange[1]]
    data = data[dateRange[0] <=data ['tags:originaldate']]
    data['tags:originaldate'] = data['tags:originaldate'].map(lambda a: a.year)
    data = data.groupby('tags:originaldate',as_index=False).mean();
    print(data[:5])


# lowlevelFeatures = ["lowlevel:spectral_energyband_low:mean","lowlevel:spectral_energy:mean","empty nothing"]

    # data['lowlevel:spectral_energy_low:mean'] = data['lowlevel:spectral_energy_low:mean'].map(ampDb)
    # data['lowlevel:spectral_energy:mean'] = data['lowlevel:spectral_energy:mean'].map(ampDb)


    f = None
    if plot:
        f, axes = mpl.subplots(nrows = math.ceil(len(features)/2), ncols = 2,figsize=(10,30))
    for i in range(len(features)):
        print(features[i]," :")
        # t = data[pd.notnull(data[features[i]])]
        # axes[i//2][i%2] = data[features[i]].scatter()
        x = data['tags:originaldate'].values
        y = data[features[i]].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        if plot:
            axes[i//2][i%2].scatter(x,y,s=1)
            axes[i//2][i%2].set_title(features[i])
            # axes[i//2][i%2].set_ylim(0,1)
            # axes[i//2][i%2].set_ylabel("squared amplitude (full-scale)")

            axes[i//2][i%2].text(0.05, 0.95, ("m: %0.2E, p: %0.2E" % (Decimal(slope), Decimal(p_value))), transform=axes[i//2][i%2].transAxes, fontsize=12, verticalalignment='top', bbox= dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        print("slope: ",slope, "  p: ",p_value)
        print('_____________\n')
    if plot:
        mpl.show()


lowlevelFeatures = ["lowlevel:average_loudness","lowlevel:dissonance:mean","lowlevel:dynamic_complexity","lowlevel:pitch_salience:mean","lowlevel:pitch_salience:median","lowlevel:spectral_centroid:mean","lowlevel:spectral_centroid:median","lowlevel:spectral_centroid:var","lowlevel:spectral_energy:mean","lowlevel:spectral_energy:median","lowlevel:spectral_energy:var","lowlevel:spectral_energyband_high:mean","lowlevel:spectral_energyband_high:median","lowlevel:spectral_energyband_high:var","lowlevel:spectral_energyband_low:mean","lowlevel:spectral_energyband_low:median","lowlevel:spectral_energyband_low:var","lowlevel:spectral_energyband_middle_high:mean","lowlevel:spectral_energyband_middle_high:median","lowlevel:spectral_energyband_middle_high:var","lowlevel:spectral_energyband_middle_low:mean","lowlevel:spectral_energyband_middle_low:median","lowlevel:spectral_energyband_middle_low:var","lowlevel:spectral_entropy:mean","lowlevel:spectral_entropy:median","lowlevel:spectral_entropy:var","lowlevel:spectral_kurtosis:mean","lowlevel:spectral_kurtosis:median","lowlevel:spectral_kurtosis:var","lowlevel:spectral_spread:mean","lowlevel:spectral_spread:median","lowlevel:spectral_spread:var","rhythm:beats_count","rhythm:bpm"]

l = []
for i in lowlevelFeatures:
    if not "var" in i and not "median" in i and not 'kurtosis' in i and not 'complexity' in i and not 'salience' in i:
        l.append(i)
lowlevelFeatures = l

# lowlevelFeatures = ["lowlevel:average_loudness","lowlevel:dissonance:mean","lowlevel:spectral_centroid:mean","lowlevel:spectral_centroid:var","lowlevel:spectral_energy:mean","lowlevel:spectral_energy:var","lowlevel:spectral_entropy:mean","lowlevel:spectral_entropy:var","lowlevel:spectral_kurtosis:mean","lowlevel:spectral_kurtosis:var","lowlevel:spectral_spread:mean","lowlevel:spectral_spread:var","rhythm:beats_count","rhythm:bpm"]

highlevelFeatures = ["danceability:danceable", " gender:female", " mood_happy:happy", " mood_sad:sad", " mood_party:party", " mood_aggressive:aggressive"," mood_relaxed:relaxed"," timbre:bright"," tonal_atonal:tonal"," voice_instrumental:instrumental"," mood_electronic:electronic"," mood_acoustic:acoustic"]

# highlevelFeatures = ["danceability:danceable", " gender:female", " mood_happy:happy", " mood_sad:sad", " mood_party:party", " mood_aggressive:aggressive"," mood_relaxed:relaxed"," timbre:bright"," tonal_atonal:tonal"," voice_instrumental:instrumental"," mood_electronic:electronic"," mood_acoustic:acoustic"]




#
# def getTimeSeries(features,csvUrl='top100-high-level-complete.csv', fillMissingOriginalDate=True):


# data = getTimeSeries(['metadata:tags:genre', " peakPos"],'top100-high-level-complete.csv')

# print(data[:5])


# csv.dropna(subset=['metadata:tags:genre'],inplace=True)

# def lol (a):
#     # print (type(a))
#     return cleanGenres(str(a))
# # data = data[data['metadata:tags:genre']]
# data = data[pd.notnull(data['metadata:tags:genre'])]
# data['metadata:tags:genre'] = data['metadata:tags:genre'].map(lol)
# print(data[:10])
# print(data.shape)
#
# genres = list(data['metadata:tags:genre'])
#
#
# genres = [item for sublist in genres for item in sublist]
#
# genres = set(genres)
# genres = list(genres)
# print(len(genres))
# print(genres[:30])

# uniq = pd.unique(data['metadata:tags:genre'])
# print(uniq.shape)
# print(uniq[:100])




################################################################################################3
# *********** Data analysis
################################################################################################
##### year coverage
# yearCoverage('top100-low-level-complete.csv',timeWindowSize=1)
# peakPosCoverage('top100-high-level-complete.csv',20)

##### HIGH LEVEL
highlevelCSV = "top100-high-level-complete.csv"
# dateLinRegress(highlevelFeatures,highlevelCSV,plot=True)
# dateLinRegress2(highlevelFeatures,highlevelCSV,plot=True)
# for i in highlevelFeatures:
    # calculateBasicStats(highlevelCSV,i,True)
# timeRangeKruskal(highlevelFeatures, highlevelCSV,timeWindowSize=10,plot=True,dateRange=(1950,2020))
peakPosKruskal(highlevelFeatures, highlevelCSV, popSize=10,plot=True,dateRange=(1950,2020))


##### LOW LEVEL
lowlevelCSV = "top100-low-level-complete.csv"
# dateLinRegress(lowlevelFeatures[:len(lowlevelFeatures)//3],lowlevelCSV,plot=True)
# dateLinRegress(lowlevelFeatures[len(lowlevelFeatures)//3:len(lowlevelFeatures)*2//3],lowlevelCSV,plot=True)
# dateLinRegress(lowlevelFeatures[len(lowlevelFeatures)*2//3:len(lowlevelFeatures)],lowlevelCSV,plot=True)

# lowlevelFeatures = ["lowlevel:spectral_energyband_low:mean","naaa","lowlevel:spectral_energy:mean","empty nothing","t","b"]
# lowlevelFeatures = ["lowlevel:spectral_energyband_middle_high:mean","lowlevel:spectral_energyband_high:mean", "lowlevel:spectral_energyband_middle_low:mean", "lowlevel:spectral_energyband_low:mean","lowlevel:spectral_energy:mean"]
# dateLinRegress2(lowlevelFeatures,lowlevelCSV,plot=True)
#
# dateLinRegress2(lowlevelFeatures,lowlevelCSV,plot=True)

# dateLinRegress2(lowlevelFeatures[:len(lowlevelFeatures)//3],lowlevelCSV,plot=True)
# dateLinRegress2(lowlevelFeatures[len(lowlevelFeatures)//3:len(lowlevelFeatures)*2//3],lowlevelCSV,plot=True)
# dateLinRegress2(lowlevelFeatures[len(lowlevelFeatures)*2//3:len(lowlevelFeatures)],lowlevelCSV,plot=True)

# for i in lowlevelFeatures:
    # calculateBasicStats(lowlevelCSV,i,True)
# timeRangeAnova(lowlevelFeatures,lowlevelCSV, timeWindowSize=10,plot=True,dateRange=(1950,2020))
# timeRangeKruskal(lowlevelFeatures[10:],lowlevelCSV, timeWindowSize=10,plot=True,dateRange=(1950,2020))
# peakPosAnova(lowlevelFeatures, lowlevelCSV, popSize=10,plot=True,dateRange=(1950,2020))
# peakPosKruskal(lowlevelFeatures, lowlevelCSV, popSize=10,plot=True,dateRange=(1950,2020))

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

features = ["danceability:danceable", " gender:female", " mood_happy:happy", " mood_sad:sad", " mood_party:party", " mood_aggressive:aggressive"," mood_relaxed:relaxed"," timbre:bright"," tonal_atonal:tonal"," voice_instrumental:instrumental"," mood_electronic:electronic"," mood_acoustic:acoustic",' peakPos',"audio_properties:length" ]



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
        # print('returned pd date: ',r)
    except:
        return np.datetime64("NaT")
    return r

def isNaT(x):
    return (np.isnat(x.to_datetime64()))

def getTimeSeries(features,csvUrl='top100-high-level-complete.csv', fillMissingOriginalDate=True):
    csv = pd.read_csv("top100-high-level-complete.csv", sep=",",header=0)
    features = features+['tags:originaldate','tags:date']
    data = csv.loc[:,features]

    data['tags:originaldate'] = data['tags:originaldate'].apply(cleanDate)
    data['tags:date'] = data['tags:date'].apply(cleanDate)

    if (fillMissingOriginalDate):
        data['tags:originaldate'] = data['tags:date'].where(data['tags:originaldate'].map(isNaT),other=data['tags:originaldate'])

    data = data[data['tags:originaldate'].map(lambda a: not isNaT(a))]

    return data.loc[:,features]



def dateLinRegress(features, dateRange=('1900','2020'),plot=False):
    data = getTimeSeries(features)
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
        print(x[:5])
        print(y[:5])
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        if plot:
            axes[i//2][i%2].scatter(x,y,s=1)
            axes[i//2][i%2].set_title(features[i])
        print("slope: ",slope, "  p: ",p_value)
        print('_____________\n')
    if plot:
        mpl.show()


def timeRangeKruskal(features,timeWindowSize=10, plot=False,dateRange=(1950,2010)):
    data = getTimeSeries(features)
    # [ timerange( [ features(mean,var) ] ) ]
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
        print(l.shape)
        kruskal = stats.kruskal(*l2)
        post_hoc = ph.posthoc_dunn(clean_data,group_col="tags:originaldate",val_col=i)
        dunn_significant = set()
        for y1 in range(post_hoc.shape[0]):
            for y2 in range(y1):
                if post_hoc.iloc[y1,y2] < 0.05 and y1 != y2:
                    dunn_significant.add((post_hoc.index.values[y1],post_hoc.columns.values[y2]))

        print("| ",kruskal)
        print("|  significant pairs: ",dunn_significant)
        print("| ",post_hoc<0.05)
        print("|___________________\n")
        if plot:
            f, axes = mpl.subplots(nrows=1, ncols = 2, constrained_layout=True)
            for j in unique_years:
                clean_data.boxplot(i,by="tags:originaldate", ax=axes[1])
            axes[1].set_title("")
            axes[1].set_xlabel("")
            for j in unique_years:
                sns.distplot(clean_data[clean_data['tags:originaldate']==j].loc[:,i],label=str(j),bins=40,norm_hist=True,ax=axes[0])
            axes[0].legend(loc="upper right")
            axes[0].set_xlabel("")
            mpl.suptitle(i)
            mpl.xlabel("Significant (Dunn posthoc): "+str(dunn_significant), fontsize=8)
            mpl.show()
        count+=1


# jsonTypeMap = {
#   "lowlevel":
#   {"average_loudness":"num","barkbands":"d2","dissonance":"d1", "dynamic_complexity":"num", "melbands": "d2","mfcc":"d2","pitch_salience":"d1","spectral_centroid":"d1","spectral_energy":"d1","spectral_energyband_high":"d1","spectral_energyband_low":"d1","spectral_energyband_middle_high":"d1","spectral_energyband_middle_low":"d1","spectral_entropy":"d1","spectral_flux":"d1","spectral_kurtosis":"d1","spectral_rms":"d1","spectral_rolloff":"d1","spectral_skewness":"d1","spectral_spread":"d1","spectral_strongpeak":"d1"},
#
#
#   "rhythm":{"beats_count":"num", "bpm":"num"},
#
#   "tonal": {"chords_changes_rate":"num", "chords_histogram":"list", "chords_key":"str", "chords_scale":"str", "key_key":"str", "key_scale":"str", "key_strength":"num"},
#   # "metadata":{"tags":'d3', "audio_properties":"d1"},
#   # "metadata":{"tags":'d3'},
# }


jsonTypeMap = {
  "lowlevel":
  {"average_loudness":"num","dissonance":"d1", "dynamic_complexity":"num","pitch_salience":"d1","spectral_centroid":"d1","spectral_energy":"d1","spectral_energyband_high":"d1","spectral_energyband_low":"d1","spectral_energyband_middle_high":"d1","spectral_energyband_middle_low":"d1","spectral_entropy":"d1","spectral_flux":"d1","spectral_kurtosis":"d1","spectral_rms":"d1","spectral_rolloff":"d1","spectral_skewness":"d1","spectral_spread":"d1","spectral_strongpeak":"d1"},
  "rhythm":{"beats_count":"num", "bpm":"num"},

  "tonal": {"chords_changes_rate":"num", "chords_histogram":"list", "chords_key":"str", "chords_scale":"str", "key_key":"str", "key_scale":"str", "key_strength":"num"},
  # "metadata":{"tags":'d3', "audio_properties":"d1"},
  # "metadata":{"tags":'d3'},
}

# "num" - 0.0
# "d1"  - {dmean:0...}
# "d2"  - {dmean[0]...}  {cov[], ...}
# "list" - [0.0]
# "str" - "E"



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

# def get_flat_json(json_data, header_string, header, row):
#     """Parse json files with nested key-vales into flat lists using nested column labeling"""
#     for root_key, root_value in json_data.items():
#         if isinstance(root_value, dict):
#             get_flat_json(root_value, header_string + '_' + str(root_key), header, row)
#         elif isinstance(root_value, list):
#             for value_index in range(len(root_value)):
#                 for nested_key, nested_value in root_value[value_index].items():
#                     header[0].append((header_string +
#                                       '_' + str(root_key) +
#                                       '_' + str(nested_key) +
#                                       '_' + str(value_index)).strip('_'))
#                     if nested_value is None:
#                         nested_value = ''
#                     row[0].append(str(nested_value))
#         else:
#             if root_value is None:
#                 root_value = ''
#             header[0].append((header_string + '_' + str(root_key)).strip('_'))
#             row[0].append(root_value)
#     return header, row



def pull():
    print((datetime.datetime.now()-startTime).seconds)

    csv = pd.read_csv('top100-high-level-complete.csv')
    mbids = list(csv['mbid'])
    i = 0
    atATime = 5
    while (i < len(mbids)):
        # try:
        for j in range(atATime):
            r = requests.get('https://acousticbrainz.org/api/v1/low-level',params={"recording_ids":mbids[i+j]})
            val = json.loads(r.text)
            print(val.keys())
            print("\n")
        # appendJSONToLowlevelCSV(csvURL="top100-low-level-complete.csv",jsonVal=vals)
        print('count: ',i,"  -  ",i/len(mbids),"% pulled")
        i+=atATime
        if(i>5):
            return
        # except KeyboardInterrupt:
            # return
        # except:
            # print("Error on iter: ",i)
    # r = requests.get('https://acousticbrainz.org/api/v1/low-level',params={"recording_ids":mbids[:35]})
    # print("size: ",len(r.text.encode('utf-8')))
    # print(r.text)
    # j = json.loads(r.text)
    #
    # print(len([]))


# *********** Data analysis
# dateLinRegress(features,plot=True)
# for i in features:
    # calculateBasicStats("top100-high-level-complete.csv",i,True)
# timeRangeKruskal(features, timeWindowSize=10,plot=True,dateRange=(1950,2020))

# pull()

# r = requests.get('https://acousticbrainz.org/api/v1/low-level',params={"recording_ids":['00057ae3-8bdc-4be3-9820-9006a10d763e', '00062658-acfc-4bdf-806f-aa6ec85e8ddd', '0006bab8-2a26-4f8d-8289-56d66ae01c68', '0008a33b-631f-4c66-a50e-ab33f27a2961']})








# h = httplib2.Http();

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



# feature = " mood_happy:happy"
# data = getTimeSeries(feature)

# x = data['tags:originaldate'].map(lambda a: a.year).as_matrix()
# y = data[feature].as_matrix()



# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

# print("slope: ",slope, "  p: ",p_value)
# mpl.scatter(x,y,s=1)
# mpl.show()




# print(originaldate['tags:originaldate'].map(fk)[:5] ==)

# print(date[:5])
# print(originaldate.shape)
# originaldate = originaldate.where(,date)

# print(originaldate[:5])

# pd.where(originaldate['tags:originaldate'].isnat(),date)
# date = date.as_matrix()
# print(originaldate[:5,0])

# print(np.isnat(originaldate[:,0]))

# date = np.where(np.isnat(originaldate[:,0]), date, originaldate) # where original date is not defined, take 'date'
# date = date[np.logical_not(np.isnat(date))]

# print("with dates: ",date.shape)
# print(date[:5])


# csv = pd.read_csv("top100-high-level-complete.csv", sep=",",header=0)
# print(csv.shape)
# print(type(csv))
# csv.replace("",np.nan,inplace=True)
# csv.dropna(subset=[' peakPos'],inplace=True)
# data = csv.loc[:," peakPos"]
# print(data[:100])
# print(data.shape)
# print(data.mean())
# print(data.var())
# print(csv.shape)
# print("done")
# >>> df.dropna(subset=['Tenant'], inplace=True)
# defined = [i for i in data if not pd.isnull(i[1]) and i[1] >= pd.to_datetime('1900',utc=True).date() and i[2]!= None]



def getLowelevel (inputCsvPath,outputCsvPath):
    csv = pd.read_csv(inputCsvPath, sep=",",header=0)
    mbids = csv.as_matrix[:,0]


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


# generateMetadataCsv("top100-high-level-commas-removed.csv",outputCsvPath="top100-high-level-metadata.csv")

# removeEndRowCommas('top100-high-level.csv','top100-high-level-commas-removed.csv')









#
#
# print(len(data))
# print((datetime.datetime.now()-startTime).seconds)
#





# # print("genre undefined for: ",nones*100/lim, "%")
# # print("unique genres counted: ",len(uniqueGenres))
# db = column(data,2)
# db_clean = [i for i in db if i != None]
#
# defined = [i for i in data if not pd.isnull(i[1]) and i[1] >= pd.to_datetime('1900',utc=True).date() and i[2]!= None]
#
# years = [pd.to_datetime(str(i)) for i in range(1950,2020)]
#
# rock = [i for i in defined if i[0] != None and 'rock' in i[0]]
# dance = [i for i in defined if i[0] != None and 'dance' in i[0]]
#
# plotme = defined
#
#
#
# x = np.array(column(plotme,1))
# y = np.array(column(plotme,2))
#
# # for i in x:
# #     print(i)
# print("loaded")
#
#
# # stats.linregress(x,y)
# year_vectorized = np.vectorize(lambda d : d.year);
# x = year_vectorized(x)
#
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# print("slope: ",slope, "  p: ",p_value)
#
# print((datetime.datetime.now()-startTime).seconds)
#
# # mpl.scatter(x,y,s=1)
# # mpl.plot([pd.to_datetime('2000',utc=True),pd.to_datetime('2001',utc=True)],[1,2])
# # mpl.show()
#
#
#
# # df = pd.DataFrame(np.random.random((200,3)))
# # df['date'] = pd.date_range('2000-1-1', periods=200, freq='D')
# # mask = (df['date'] > '2000-6-1') & (df['date'] <= '2000-6-10')
# # print(df.loc[mask])
#
#
#
# # data  = pd.read_csv('fma_metadata/raw_tracks.csv', delimiter = ',', dtype = None)
# #
# # # data = np.genfromtxt("fma_metadata/raw_tracks.csv", delimiter=",",encoding="utf8",usecols=(1),names=True, dtype=None, max_rows=100000)
# #
# # dates = data[19,:]
# #
# # print(dates.shape);

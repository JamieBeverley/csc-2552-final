import numpy as np
import tarfile
import json
import pandas as pd
# import matplotlib.pyplot as mpl
# import matplotlib.markers as markers
from scipy import stats
import datetime
import ast


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

            #     outfile.write(s);
            #     print(s)
            # except:
            #     print("error with: ")
            #     print(s)
            #     outfile.close()
            #     return
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

features = ["danceability:danceable", " gender:female", " mood_happy:happy", " mood_sad:sad", " mood_party:party", " mood_relaxed:relaxed"," timbre:bright"," tonal_atonal:tonal"," voice_instrumental:instrumental",' peakPos']

# for i in features:
    # calculateBasicStats("top100-high-level-complete.csv",i,True)


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


# csv = pd.read_csv("top100-high-level-complete.csv", sep=",",header=0)
# csv.replace("",np.nan,inplace=True)
# csv.dropna(subset=['tags:originaldate'],inplace=True);
# originaldate = csv.loc[:,"tags:originaldate"]
# print(originaldate.shape)
# originaldate = originaldate.map(cleanDate)
# originaldate.dropna(inplace=True)
# print(originaldate.shape)
# print(originaldate[:5])

print("_________________________")
# csv = pd.read_csv("top100-high-level-complete.csv", sep=",",header=0)
# originaldate = (csv.loc[:,"tags:originaldate"]).map(cleanDate).values
# date = (csv.loc[:,'tags:date']).map(cleanDate).values

csv = pd.read_csv("top100-high-level-complete.csv", sep=",",header=0)
data = (csv.loc[:,("tags:originaldate"," peakPos",'tags:date')])
data['tags:originaldate'] = data['tags:originaldate'].apply(cleanDate)
data['tags:date'] = data['tags:date'].apply(cleanDate)
# originaldate[:,'tags:originaldate'] = originaldate[:,'tags:originaldate'].map(cleanDate)
# originaldate = originaldate.as_matrix()


def fk(x):
    return (np.isnat(x.to_datetime64()))

data['tags:originaldate'] = data['tags:date'].where(data['tags:originaldate'].map(fk),other=data['tags:originaldate'])
data = data[data['tags:originaldate'].map(lambda a: not fk(a))]


print(data[:70])
print("@@@@@@@@@@@@")



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

import numpy as np
import tarfile
import json
import pandas as pd
# import matplotlib.pyplot as mpl
# import matplotlib.markers as markers
from scipy import stats
import datetime

startTime = datetime.datetime.now()

tar = tarfile.open("high-level.tar.bz2",mode="r:bz2")

top100 = open('top-100.json')
top100 = json.load(top100)
top100 = list(top100.values())
print("unique top 100 songs: ",len(top100))

i = 0
lim = 40000
b = True

d= {}

uniqueGenres = set()

# genre, date, danceability
nones = 0

# data = np.zeros((lim,3))
data = [None]*lim


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

def cleanDate(s):
    if (s == None):
        return None
    s = s.replace(" ","")
    r = None
    try:
        r = pd.to_datetime(s,utc=True).date();
    except:
        return None
        # print("warning could not clean date: ", s)
    return r

# slope:  0.0051008503915541655   p:  1.8362819149172405e-14
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


print('reading...')
# tar.list()
# print(len(tar.list()))
# thing = []
# test = 0;
# for member in tar:
#     test+=1;
#     if(test%100==0):
#         print(test)
# print(test)


def compare(s1,s2):
    if (s1.lower()==s2.lower()):
        return True

    s1 = s1.lower().replace(" ","")
    s2 = s2.lower().replace(" ","")

    if(s1==s2):
        # print("match without spaces")
        return True
    return False

    # return (s1.lower()==s2.lower() or s1.lower().replace(" ","") == s2.lower().replace(" ",""))

def generateTop100MBIDJSON (outputURL='top100-high-level.csv',lim=None,startAt=0):

    hasPutHeader = False;
    count = 0;
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
                                        headerStr = headerStr +i+":"+j+", "
                                headerStr += "\n"
                                outfile.write(headerStr)
                                hasPutHeader = True

                            s = mbid+", \""+title.replace(",","")+"\", \""+artist.replace(",","")+"\", "
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


generateTop100MBIDJSON()


        # # content = json.load(s)
        # # test = datetime.datetime.now()
        # # thing.append(datetime.datetime.now()-test)
        #
        # genre = content.get('metadata',d).get('tags',d).get('genre')
        # date = content.get('metadata',d).get('tags',d).get('date')
        #
        # genre = cleanGenres(safe_first(genre))
        # date = cleanDate(safe_first(date))
        #
        #
        # data[i] = [genre, date]
        #
        # if (type(genre) == list):
        #     for l in genre:
        #         uniqueGenres.add(l)
        # else:
        #     uniqueGenres.add(genre);
        # if (genre == None):
        #     nones += 1






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

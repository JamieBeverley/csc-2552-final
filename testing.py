import numpy as np
import tarfile
import json
import pandas as pd
import matplotlib.pyplot as mpl
import matplotlib.markers as markers
from scipy import stats
import datetime

starTime = datetime.datetime.now()
tar = tarfile.open("acousticbrainz-highlevel-json-20150130.tar.bz2",mode="r:bz2")

i = 0
lim = 4000
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

for member in tar:
    f=tar.extractfile(member)
    content = json.load(f)
    genre = content.get('metadata',d).get('tags',d).get('genre')
    date = content.get('metadata',d).get('tags',d).get('date')
    danceability = content.get('highlevel',d).get('danceability',d).get('probability')

    gender = content.get('highlevel',d).get('gender',d).get('probability')
    sad = content.get('highlevel',d).get('mood_sad',d).get('probability')

    genre = cleanGenres(safe_first(genre))
    date = cleanDate(safe_first(date))
    danceability = safe_first(danceability)
    gender = safe_first(gender)
    sad = safe_first(sad)


    data[i] = [genre,date,danceability,gender,sad]

    if (type(genre) == list):
        for l in genre:
            uniqueGenres.add(l)
    else:
        uniqueGenres.add(genre);
    if (genre == None):
        nones += 1

    i=i+1
    # print(member.isfile())
    if (i>=lim):
        print("breaking on i")
        break



# print("genre undefined for: ",nones*100/lim, "%")
# print("unique genres counted: ",len(uniqueGenres))
db = column(data,2)
db_clean = [i for i in db if i != None]

defined = [i for i in data if not pd.isnull(i[1]) and i[1] >= pd.to_datetime('1900',utc=True).date() and i[2]!= None]

years = [pd.to_datetime(str(i)) for i in range(1950,2020)]

rock = [i for i in defined if i[0] != None and 'rock' in i[0]]
dance = [i for i in defined if i[0] != None and 'dance' in i[0]]

plotme = defined



x = np.array(column(plotme,1))
y = np.array(column(plotme,2))

# for i in x:
#     print(i)
print("loaded")


# stats.linregress(x,y)
year_vectorized = np.vectorize(lambda d : d.year);
x = year_vectorized(x)

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print("slope: ",slope, "  p: ",p_value)
print((datetime.datetime.now()-starTime).seconds)

# mpl.scatter(x,y,s=1)
# mpl.plot([pd.to_datetime('2000',utc=True),pd.to_datetime('2001',utc=True)],[1,2])
# mpl.show()




# df = pd.DataFrame(np.random.random((200,3)))
# df['date'] = pd.date_range('2000-1-1', periods=200, freq='D')
# mask = (df['date'] > '2000-6-1') & (df['date'] <= '2000-6-10')
# print(df.loc[mask])



# data  = pd.read_csv('fma_metadata/raw_tracks.csv', delimiter = ',', dtype = None)
#
# # data = np.genfromtxt("fma_metadata/raw_tracks.csv", delimiter=",",encoding="utf8",usecols=(1),names=True, dtype=None, max_rows=100000)
#
# dates = data[19,:]
#
# print(dates.shape);

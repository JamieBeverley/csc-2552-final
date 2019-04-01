import billboard
import json

chart = billboard.ChartData('hot-100')


file = open('top-100.json')
d = json.load(file)


# l = []

lim = 30;

errs = []

while chart.previousDate:
    # print(chart.previousDate)
    try:
        print(chart.date)
        for i in chart.entries:
            d[i.title+" - "+i.artist] = {"title":i.title,"artist":i.artist,'peakPos':i.peakPos}
            # l.append({'title':i.title,'artist':i.artist,'peakPos':i.peakPos,'rank':i.rank})

        with open('top-100.json', 'w') as outfile:
            json.dump(d, outfile)
        # print(chart.entries[0].title+" - "+chart.entries[0].artist)
        chart = billboard.ChartData('hot-100', chart.previousDate)
    except KeyboardInterrupt:
        print("keyboard intterupt, terminating at date: ", chart.date)
        break;
    except:
        print("**************** something wrong at date: ",chart.date)
        errs.append(chart.date)

print("errors at: ",errs)

print(len(d))

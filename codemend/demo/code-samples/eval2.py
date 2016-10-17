import matplotlib.pyplot as plt

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)

plt.pie(sizes, explode=explode, labels=labels,
        colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
text = plt.title('Raining hogs and dogs', y=1.1)
text.set_bbox(dict(facecolor='lightskyblue',
                   edgecolor='grey',
                   pad=5))
plt.axis('equal')

import matplotlib.pyplot as plt

intervals = [30, 60, 120, 300, 720]
media_data = {
    "CREA":   [0.9973, 0.9735, 0.8675, 0.7635, 0.6490],
    "PDA":    [0.9993, 0.9951, 0.9527, 0.8042, 0.7222],
    "YES":    [0.9970, 0.9810, 0.9019, 0.7828, 0.6554],
    "CYA":    [0.9992, 0.9929, 0.9365, 0.8065, 0.7130],
    "MEAox":  [0.9973, 0.9820, 0.9165, 0.8018, 0.7050],
    "OAT":    [0.9993, 0.9929, 0.9216, 0.7901, 0.6790]
}

color_map = {
    "CREA": "#AEC6CF",
    "PDA": "#FFB347",
    "YES": "#77DD77",
    "CYA": "#FF6961",
    "MEAox": "#B39EB5",  
    "OAT": "#FDFD96"    
}

plt.figure()
for medium, values in media_data.items():
    plt.plot(intervals, values, marker='o',
             color=color_map[medium],
             label=medium)

plt.xlabel("Time-lapse Interval (minutes)")
plt.ylabel("Average Accuracy")
plt.title("Media Accuracy Across Time-lapse Intervals")
plt.legend()
plt.tight_layout()
plt.show()

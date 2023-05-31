    
ywerte_all = []
Hz = np.linspace(50,51.6,1601)
for i in [50.25, 50.3, 50.35, 50.4, 50.45, 50.5, 50.55, 50.6, 50.65, 50.7, 50.75, 50.8, 50.85, 50.9, 50.95, 51]:
    ywerte = SysStabV(i)
    ywerte_all.append(ywerte)
    
colors = ['darkorange', 'burlywood', 'antiquewhite', 'tan', 'navajowhite', 'blanchedalmond', 'papayawhip', 'moccasin', 'orange', 'wheat', 'oldlace', 'floralwhite', 'cornsilk', 'linen', 'peru', 'peachpuff']
plt.figure(figsize=(15, 8))
for y in zip(ywerte_all,colors):
    plt.plot(Hz,y[0], y[1], linewidth = 3)


plt.show()

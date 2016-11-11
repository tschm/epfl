import matplotlib
import matplotlib.pyplot as plt
plt.style.use = 'default'

    
def config_configManager():
    from notebook.services.config import ConfigManager
    cm = ConfigManager()
    cm.update('livereveal', {
              'theme': 'serif',
              'start_slideshow_at': 'selected',
              'width': 1680,
              'height': 768,
    })


def config_matplotlib():
    params = {'legend.fontsize': 'large', 
              'lines.linewidth':3, 
              'font.size':16, 
              'axes.labelsize':'large', 
              'axes.labelcolor':'black', 
              'xtick.labelsize':'large', 
              'ytick.labelsize':'large', 
              'figure.figsize':(20, 6)}

    plt.rcParams.update(params)
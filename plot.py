import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(mean_scores)
    plt.legend(['Score moyen sur 100 parties'])
    plt.ylim(ymin=0)
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)

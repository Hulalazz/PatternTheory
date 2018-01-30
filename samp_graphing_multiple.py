# adding more stuff to the graphing template

from matplotlib.pylab import *
import matplotlib.animation as animation
import samp_graphing as gp


class graphing_multiple(gp.graphing_multiple):
    def __init__(self, N):
        super(graphing_multiple, self).__init__(N)

    def data_placeholder(self):
        self.yp1 = zeros(int(self.N / 2) - 1)
        self.yp2 = zeros(int(self.N / 2) - 1)
        self.yp3 = zeros(int(self.N / 2) - 1)
        self.yp4 = zeros(int(self.N / 2) - 1)
        self.yv1 = zeros(self.N)
        self.yv2 = zeros(self.N)
        self.x01 = np.arange(int(self.N / 2) - 1)
        self.x02 = np.arange(self.N)

    def set_plots(self):
        self.p011, = self.ax01.plot(self.x01, self.yp1, 'g-', lw=0.2)
        self.p012, = self.ax01.plot(self.x01, self.yp2, 'c-', lw=0.2)
        self.p013, = self.ax01.plot(self.x01, self.yp3, 'r-', lw=0.2)
        self.p014, = self.ax01.plot(self.x01, self.yp4, 'k-', lw=0.2)
        self.p021, = self.ax02.plot(self.x02, self.yv1, 'b-', lw=0.2)
        self.p022, = self.ax02.plot(self.x02, self.yv2, 'g-', lw=0.2)

    def updateData(self, i):
        print("Override Expected")
        self.p011.set_data(self.x01, self.yp1)
        self.p012.set_data(self.x01, self.yp2)
        self.p013.set_data(self.x01, self.yp3)
        self.p014.set_data(self.x01, self.yp4)
        self.p021.set_data(self.x02, self.yv1)
        self.p022.set_data(self.x02, self.yv2)
        return self.p011, self.p012, self.p013, self.p014, self.p021, self.p022
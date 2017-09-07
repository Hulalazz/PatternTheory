from matplotlib.pylab import *
import matplotlib.animation as animation
# as per graphing multiple under package

class graphing_multiple(object):
    def __init__(self, N):
        self.N = N
        self.setup()
        self.set_titles()
        self.define_limits_sound()
        self.define_limits_spectrum()
        self.data_placeholder()
        self.set_plots()

    def setup(self):
        self.font = {'size': 9}
        matplotlib.rc('font', **self.font)

        # setup figure and subplots
        self.f0 = figure(num=0, figsize=(16, 8))  # , dpi = 100)
        self.ax01 = subplot2grid((2, 1), (0, 0))
        self.ax02 = subplot2grid((2, 1), (1, 0))

    def set_titles(self):
        self.ax01.set_title('Spectrum')
        self.ax02.set_title('Sound Wave')
        self.ax01.grid(True)
        self.ax02.grid(True)

    def data_placeholder(self):
        self.yp1 = zeros(int(self.N / 2) - 1)
        self.yv1 = zeros(self.N)
        self.x01 = np.arange(int(self.N / 2) - 1)
        self.x02 = np.arange(self.N)

    def set_plots(self):
        self.p011, = self.ax01.plot(self.x01, self.yp1, 'b-', lw=0.2)
        self.p021, = self.ax02.plot(self.x02, self.yv1, 'b-', lw=0.2)

    def define_limits_spectrum(self):
        self.ax01.set_xlim(0, int(self.N/2) - 1)
        self.ax01.set_ylim(0, int(3*np.sqrt(self.N)))

    def define_limits_sound(self):
        self.ax02.set_xlim(0, self.N)
        self.ax02.set_ylim(-1, 1)

    def updateData(self, i):
        print("Override Expected")
        self.p011.set_data(self.x01, self.yp1)
        self.p021.set_data(self.x02, self.yv1)
        return self.p011, self.p021

    def start(self):
        self.simulation = animation.FuncAnimation(self.f0, self.updateData, blit=False, interval=3000, repeat=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:33:48 2018

@author: hkmac
"""

import numpy as np
import matplotlib.pyplot as plt

# GENERATING SAMPLES FROM GROUND TRUTH
PROB_TURE = 0.5  # probability if it lies from normal distribution 2 
MEAN_1_TRUE = 0.
STDE_1_TRUE = .5
MEAN_2_TRUE = 3.
STDE_2_TRUE = 1.

LENGTH = 1000
SAMPLES = [STDE_1_TRUE * np.random.randn() + MEAN_1_TRUE
           if np.random.random() > PROB_TURE 
           else STDE_2_TRUE * np.random.randn() + MEAN_2_TRUE
           for _ in range(LENGTH)]

def calc_log_lik_(mean_1_, mean_2_):
    return np.sum(
            np.log(
                ((1-PROB_TURE)/(STDE_1_TRUE*np.sqrt(2*np.pi)))
                  *np.exp(-np.power(np.add(SAMPLES,[-mean_1_]*LENGTH),2)/(2*STDE_1_TRUE**2))
                + (PROB_TURE/(STDE_2_TRUE*np.sqrt(2*np.pi)))
                  *np.exp(-np.power(np.add(SAMPLES,[-mean_2_]*LENGTH),2)/(2*STDE_2_TRUE**2))))

calc_log_lik = np.vectorize(calc_log_lik_)
GROND_TRUTH_LOG_LIK = calc_log_lik(MEAN_1_TRUE, MEAN_2_TRUE)

print("GROND_TRUTH_LOG_LIK = {}".format(GROND_TRUTH_LOG_LIK))



import matplotlib.colors as colors
import matplotlib.cm as cm
import copy

fig1 = plt.figure(figsize=(5,5))
ax1 = fig1.add_subplot(111)


MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_START, MEAN_2_PLT_END = -2,7,-2,7
PLOT_EXTENT=[MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_END, MEAN_2_PLT_START]
CONT_EXTENT=[MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_START, MEAN_2_PLT_END]

MEAN_1_LINSPACE = np.linspace(MEAN_1_PLT_START, MEAN_1_PLT_END, 100)
MEAN_2_LINSPACE = np.linspace(MEAN_2_PLT_START, MEAN_2_PLT_END, 100)

MEAN_1_GRID, MEAN_2_GRID = np.meshgrid(MEAN_1_LINSPACE, MEAN_2_LINSPACE)

LOG_LIK_MAP = calc_log_lik(MEAN_1_GRID, MEAN_2_GRID)
LOG_LIK_MAX = np.max(LOG_LIK_MAP)
LOG_LIK_MIN = np.min(LOG_LIK_MAP)


print(LOG_LIK_MIN,LOG_LIK_MAX)
ax1.imshow(LOG_LIK_MAP,
           extent = PLOT_EXTENT,
           cmap=cm.gist_rainbow,
           norm = colors.SymLogNorm(vmax=LOG_LIK_MAX, vmin=LOG_LIK_MIN, linthresh=1, clip=True))

assert LOG_LIK_MAX < 0  # so that countour levels make sense

CS = plt.contour(LOG_LIK_MAP,
                extent = [MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_START, MEAN_2_PLT_END],
                norm = colors.SymLogNorm(vmax=LOG_LIK_MAX, vmin=LOG_LIK_MIN, linthresh=1),
                levels=[LOG_LIK_MAX*3., LOG_LIK_MAX*1.5, LOG_LIK_MAX*1.1]
                )
ax1.clabel(CS, inline=1, fontsize=10, fmt="%d")
plt.show()


## we have arbitrary starting points which we will elaborate later
#global mean_1
#global mean_2
prob, mean_1, stde_1, mean_2, stde_2 = PROB_TRUE, 2., STDE_1_TRUE, 3., STDE_2_TRUE

def expectation_step(prob,mean_1,stde_1,mean_2,stde_2):
    odds_Y1 = [((1-prob)/(stde_1*np.sqrt(2*np.pi))) 
                * np.exp(-(np.add(samples,[-mean_1]*LENGTH))**2. / (2. * stde_1**2))]
    odds_Y2 = [((prob)/(stde_2*np.sqrt(2*np.pi))) 
                * np.exp(-(np.add(samples,[-mean_2]*LENGTH))**2. / (2. * stde_2**2))]
    gamma_i = np.divide(odds_Y2,np.add(odds_Y1,odds_Y2))
    return gamma_i[0]

global gamma_i
gamma_i = expectation_step(prob,mean_1,stde_1,mean_2,stde_2)
print(np.round(gamma_i[:30],3))  # the gamma values of some of the samples



from scipy.special import xlogy  # calculates x*log(y)

def calc_log_lik_lower_(mean_1_,mean_2_):
    odds_Y1 = [((1-pi)/(stde_1*np.sqrt(2.*np.pi))) 
                * np.exp(-np.power(np.add(samples,[-mean_1_]*LENGTH),2.) / (2. * stde_1**2))]
    odds_Y2 = [((pi)/(stde_2*np.sqrt(2.*np.pi))) 
                * np.exp(-np.power(np.add(samples,[-mean_2_]*LENGTH),2.) / (2. * stde_2**2))]

    return np.sum(np.add(np.add(xlogy(gamma_i, odds_Y2),-xlogy(gamma_i,gamma_i)),
                         np.add(xlogy(1-gamma_i, odds_Y1),-xlogy(1-gamma_i,1-gamma_i))))

calc_log_lik_lower = np.vectorize(calc_log_lik_lower_)

global nll_lower_map
log_lik_lower_map = calc_log_lik_lower(MEAN_1_GRID, MEAN_2_GRID)




def plot_ell_compare(nll_lower_map, nll_map):
    
#    MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_START, MEAN_2_PLT_END = -2,7,-2,7
#    MEAN_1_LINSPACE = np.linspace(MEAN_1_PLT_START, MEAN_1_PLT_END, 100)
#    MEAN_2_LINSPACE = np.linspace(MEAN_2_PLT_START, MEAN_2_PLT_END, 100)
#    MEAN_1_GRID, MEAN_2_GRID = np.meshgrid(MEAN_1_LINSPACE, MEAN_2_LINSPACE)
#    extent=[MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_END, MEAN_2_PLT_START]
    
    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.add_subplot(121)
    ax1.imshow(nll_lower_map,
               extent = extent,
               cmap=cm.gist_rainbow,
               norm = colors.SymLogNorm(vmax=vmax, vmin=vmin, linthresh=1, clip=True)
              )
    CS = ax1.contour(nll_lower_map,
           extent = [MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_START, MEAN_2_PLT_END],
           norm = colors.SymLogNorm(vmax=vmax, vmin=vmin, linthresh=1, clip=True),
                    levels=[vmax*8.8, vmax*4.4, vmax*2.2, vmax*1.1, vmax*1.01]
                    )
    
#     ax1.axvline(x=mean_1_true, linewidth=2, color='k', linestyle='dotted')
#     ax1.axhline(y=mean_2_true, linewidth=2, color='k', linestyle='dotted')
    ax1.clabel(CS, inline=1, fontsize=10, fmt="%d")
    ax1.set_xlabel("$\mean_1$")
    ax1.set_ylabel("$\mean_2$")
    ax1.set_title("nll_lower")
    ax1.scatter(mean_1,mean_2,color="white")

    ax2 = fig1.add_subplot(122)
    ax2.imshow(nll_map,
           extent = extent,
           cmap=cm.gist_rainbow,
           norm = colors.SymLogNorm(vmax=vmax, vmin=vmin, linthresh=1, clip=True)
    # norm=colors.LogNorm(vmin=nll_map.min(), vmax=nll_map.max())
          )
    ax2.scatter(mean_1,mean_2,color="white")

    CS = plt.contour(nll_map,
           extent = [MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_START, MEAN_2_PLT_END],
                     norm = colors.SymLogNorm(vmax=vmax, vmin=vmin, linthresh=1, clip=True),
                     levels=[vmax*8.8, vmax*4.4, vmax*2.2, vmax*1.1, vmax*1.01]
                    )
    ax2.clabel(CS, inline=1, fontsize=10, fmt="%d")
    ax2.set_xlabel("$\mean_1$")
    ax2.set_ylabel("$\mean_2$")
    ax2.set_title("nll")
    plt.show()
    
%matplotlib inline
plot_ell_compare(nll_lower_map, nll_map)
print(mean_1,mean_2)
print(calc_nll(mean_1,mean_2))
print(nll_lower_(mean_1,mean_2))




%matplotlib tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import copy


# fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.25, bottom=0.25)
# t = np.arange(0.0, 1.0, 0.001)
# a0 = 5
# f0 = 1.2
# delta_f = 5.0
# s = a0*np.sin(2*np.pi*f0*t)
# l, = plt.plot(t, s, lw=2, color='red')
# plt.axis([0, 1, -10, 10])


MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_START, MEAN_2_PLT_END = -2,7,-2,7
MEAN_1_LINSPACE = np.linspace(MEAN_1_PLT_START, MEAN_1_PLT_END, 100)
MEAN_2_LINSPACE = np.linspace(MEAN_2_PLT_START, MEAN_2_PLT_END, 100)
MEAN_1_GRID, MEAN_2_GRID = np.meshgrid(MEAN_1_LINSPACE, MEAN_2_LINSPACE)
extent=[MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_END, MEAN_2_PLT_START]

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
nll_map_ = copy.deepcopy(nll_map)
nll_map_[nll_map_<-5000.] = np.nan
global surf


surf = ax.plot_surface(MEAN_1_GRID, MEAN_2_GRID, nll_map_,
                cmap=cm.gist_rainbow,
                vmin=3.*vmax,
                vmax=vmax)
# global wire
# wire = ax.plot_wireframe(MEAN_1_GRID, MEAN_2_GRID, nll_lower_map)
# wire.remove()


ax.view_init(-3, 210)
ax.set_zlim3d(vmax*4., 0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))


axcolor = 'lightgoldenrodyellow'
ax_mean_1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_mean_2 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)

mean_1_init = 5
mean_2_init = 1.2
s_mean_1 = Slider(ax_mean_1, '$\mu$_1', 0.1, 5.0, valinit=mean_1_init)
s_mean_2 = Slider(ax_mean_2, '$\mu$_2', 0.1, 5.0, valinit=mean_2_init)


print(np.round(gamma_i[:30],3))  # the gamma values of some of the samples
# print(nll_lower_map)


def update(val):
    
    MEAN_1_PLT_START, MEAN_1_PLT_END, MEAN_2_PLT_START, MEAN_2_PLT_END = -2,7,-2,7
    MEAN_1_LINSPACE = np.linspace(MEAN_1_PLT_START, MEAN_1_PLT_END, 100)
    MEAN_2_LINSPACE = np.linspace(MEAN_2_PLT_START, MEAN_2_PLT_END, 100)
    MEAN_1_GRID, MEAN_2_GRID = np.meshgrid(MEAN_1_LINSPACE, MEAN_2_LINSPACE)

    mean_1 = s_mean_1.val
    mean_2 = s_mean_2.val
    print(pi,mean_1,mean_2,stde_1,stde_2)
    gamma_i = expectation_step(prob,mean_1,stde_1,mean_2,stde_2)
    print(np.round(gamma_i[:30],3))  # the gamma values of some of the samples

    def nll_lower_(mean_1,mean_2):
        odds_Y1 = [((1-pi)/(stde_1*np.sqrt(2.*np.pi))) 
                    * np.exp(-(np.add(samples,[-mean_1]*l))**2. / (2. * stde_1**2))]
        odds_Y2 = [((pi)/(stde_2*np.sqrt(2.*np.pi))) 
                    * np.exp(-(np.add(samples,[-mean_2]*l))**2. / (2. * stde_2**2))]
        
        return np.sum(np.add(np.add(xlogy(gamma_i, odds_Y2),-xlogy(gamma_i,gamma_i)),
                           np.add(xlogy(1-gamma_i, odds_Y1),-xlogy(1-gamma_i,1-gamma_i))))
        
    nll_lower = np.vectorize(nll_lower_)
    nll_lower_map_ = nll_lower(MEAN_1_GRID, MEAN_2_GRID)
    print(nll_lower_map_[0:3,0:3])

    if len(list(ax.collections)) > 1:
        print("delet this")
        ax.collections.remove(list(ax.collections)[-1])
#         ax.collections.remove(list(ax.collections)[-1])
        
    wire = ax.plot_wireframe(MEAN_1_GRID, MEAN_2_GRID, nll_lower_map_)
    
    line = ax.plot([mean_1,mean_1],[mean_2,mean_2],[0,nll_lower(mean_1, mean_2)],
                   color = 'g')
    fig.canvas.draw()
    
s_mean_1.on_changed(update)
s_mean_2.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_mean_1.reset()
    s_mean_2.reset()
button.on_clicked(reset)

# rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
# radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

# def colorfunc(label):
#     l.set_color(label)
#     fig.canvas.draw_idle()
# radio.on_clicked(colorfunc)

plt.show()
#!/usr/bin/env python

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit

def g(r, A, B , ampl, off): 
    return ampl*1/(1+np.exp((A-r)/B))*(1+r*np.exp((A-r)/B)/(3*B*(1+np.exp((A-r)/B))))+off


class SEM_image:
    def __init__(self, filepath, Y_PXLS=-1):
        im = np.array(Image.open(filepath+'.tif'))
        if len(im.shape) > 2:
            self.image = np.sum(im, axis=2) #sum rgb to one value (grayscale)
            self.image = self.image[:Y_PXLS,:]
        else:
            self.image = np.array(im)[:Y_PXLS,:]
        self.meta = {}
        with open(filepath+'.txt') as f:
            for line in f.readlines()[1:]:
                #special treatment for Condition
                if 'Condition=' in line: key, var = line[:9], line[10:]
                else: key, var = line.split('=')
                self.meta[key] = var[:-2]
        self.name = filepath.split('/')[-1]
        
                
    def calculate_fft2D(self):
        self.fft2D = np.fft.fft2(self.image - np.mean(self.image))
        self.fft2D = np.fft.fftshift(np.abs(self.fft2D))
        self.fft2D_y = np.fft.fftfreq(self.image.shape[0], d=float(self.meta['PixelSize']))
        self.fft2D_y = np.fft.fftshift(self.fft2D_y)
        self.fft2D_x = np.fft.fftfreq(self.image.shape[1], d=float(self.meta['PixelSize']))
        self.fft2D_x = np.fft.fftshift(self.fft2D_x)

    def show_image(self, fig, ax):
        ax.cla()
        ax.imshow(self.image, cmap='Greys_r')
        ax.set_title(self.meta['Condition'])
        fig.canvas.draw()
        
    def show_fft2D(self, fig, ax):
        ax.cla()
        ax.imshow(self.fft2D, cmap='viridis')
        ax.set_title('2D FFT')
        fig.canvas.draw()
        
    def show_fft2D_scaled(self, fig, ax):
        ax.cla()
        ax.set_aspect(1)
        ax.pcolor(self.fft2D_x, self.fft2D_y, self.fft2D, cmap='viridis')
        ax.set_title('2D FFT')
        fig.canvas.draw()
        
    def calculate_radial_fft_fast(self, N):
        #inspired by https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
        max_R = np.min([np.max(np.abs(self.fft2D_x)), np.max(np.abs(self.fft2D_y))])
        
        X,Y = np.meshgrid(self.fft2D_x/max_R*N, self.fft2D_y/max_R*N)
        r = np.hypot(X,Y).astype(int)
        
        self.fft1D_old = ndimage.mean(self.fft2D, r, index=np.arange(1,N))
        print(self.fft1D_old[-1])
        #self.fft1D_old /= self.fft1D_old[-1]
        self.fft1D_r_old = 1/np.linspace(0, max_R, N)[1:]

        
        
    def calculate_radial_fft(self, N, R_min, R_max, dTheta=0.1):
        interpol = RectBivariateSpline(self.fft2D_y, self.fft2D_x, self.fft2D)
        
        max_R = np.min([np.max(np.abs(self.fft2D_x)), np.max(np.abs(self.fft2D_y))])
        self.fft1D_r = np.linspace(1/R_max, 1/R_min, N)
        
        self.fft1D = []
        theta = np.linspace(0, 360-dTheta, int(360/dTheta))
        for radius in self.fft1D_r:
            self.fft1D.append(np.mean(interpol.ev(radius*np.sin(theta), radius*np.cos(theta))))
        self.fft1D =  np.array(self.fft1D)
        self.fft1D /= self.fft1D[0]
        self.fft1D_r = 1/self.fft1D_r
    


    def fit_RDF(self, verbose=False):
        p_init = [self.fft1D_r[np.argmax(self.fft1D)], 0.85, 1, np.min(self.fft1D)]
        try:
            self.popt_RDF, self.pcov_RDF = curve_fit(g, self.fft1D_r, self.fft1D, p0=p_init)

        except: 
            print('could not fit', self.name)
            self.popt_RDF = [np.nan, np.nan, np.nan, np.nan]
            self.pcov_RDF = [np.nan, np.nan, np.nan, np.nan]


    def show_RDF(self, fig, ax):
        ax.cla()
        ax.plot(self.fft1D_r, self.fft1D, color='k')
        ax.set_xlabel('radial distance [nm]')
        ax.set_ylabel('normalized radial FFT')
        fig.canvas.draw()
    
    def show_RDF_fit(self, fig, ax):
        self.show_RDF(fig,ax)
        ax.errorbar(self.popt_RDF[0], self.popt_RDF[2]+self.popt_RDF[3], xerr=self.popt_RDF[1], color='r', marker='o', capsize=4)
        ax.axvline(self.popt_RDF[0], color='r', linestyle=':')
        ax.plot(self.fft1D_r, g(self.fft1D_r, *self.popt_RDF), color='r')
        fig.canvas.draw()
    
    
    

"""                 ParallaxBarrierCreatorFinal.py
Created on Tue Sep 13 14:34:20 2016

Creates a parallax barrier to print on a transparency film.
There are also functions pitchTest() which creates a barrier pitch test and
multiFrac() which creates a series of barriers similar to a pitch test, but instead
of varying the barrier pitch, the barrier fraction is varied.

@author: dmcauslan
"""

# Import packages
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import png
import time

# Define barrier/screen/printer parameters
pixelPitch = 0.2175                                 # Pixel pitch (mm)
nViews = 5                                          # Number of views
pixelFrac = 1                                       # Barrier width as a fraction of a pixel (slightly smaller than the pixel pitch may be better)
printerRes = (600,1200)                             # Printer resolution (DPI)
maxRes = 2e7
if printerRes[0]*printerRes[1]>maxRes:
    raise ValueError('Printer resolution is set too high, program may crash. Try a resolution that has a maximum product of {} DPI^2'.format(maxRes))
printerPitch = (25.4/printerRes[0], 25.4/printerRes[1])                      # Dot pitch of the printer (mm)
paperDim = (216, 279)                               # Dimension of page to print barrier on (mm) (Letter)

# Plots the parallax barrier. Note that for high resolution barriers this can take
# a lot of computer memory
def barrierPlot(bImage, figNum):
    figP = plt.figure(figNum)
    figP.clf()
    sns.set_style("white")
    plt.imshow(bImage, interpolation = "nearest", cmap = 'Greys_r')
    figP.show()

# Create a parallax barrier
def createBarrier(paperDim, printerPitch, pixelPitch, nViews, pixelFrac):
    xDim = np.arange(0,paperDim[1],printerPitch[1])
    yDim = np.abs(np.floor((xDim%(pixelPitch*nViews))/(pixelPitch*nViews)-pixelFrac/nViews)).astype('uint8')
    barrierImg = yDim*np.ones((round(paperDim[0]/printerPitch[0]),1),dtype = 'uint8')
    return barrierImg

# create a simple parallax barrier pattern
def singleBarrier(paperDim, printerPitch, pixelPitch, nViews, pixelFrac):
    barrierImg = createBarrier(paperDim, printerPitch, pixelPitch, nViews, pixelFrac)
    
    # Save the image, note here it is being saved as greyscale, with 1 bit per pixel - this should be all that is needed
    fName = 'H:/David/Google Drive/Canopy/Barrier Patterns/Barrier_{}Views_{}x{}mm_{}x{}DPI.png'.format(nViews, paperDim[0], paperDim[1], printerRes[0], printerRes[1])
    png.from_array(barrierImg, 'L;1').save(fName)
    return barrierImg    

# Create a pitch test with rows of multiple barriers
def pitchTest(paperDim, pixelPitch, printerPitch, nViews, pixelFrac):
    numTests = 9
    pitchStep = 0.0005
    lineDim = [divmod(paperDim[0],numTests)[0], paperDim[1]]
    # The range of different pixel pitches to create barriers of
    pxPitches = np.linspace(pixelPitch-pitchStep*math.floor((numTests-1)/2),pixelPitch+pitchStep*math.floor(numTests/2),numTests)
    
    # Loop over the number of barriers to test, creating each barrier and assembling into 1 image
    for n in range(numTests):
        tmpImg = createBarrier(lineDim, printerPitch, pxPitches[n], nViews, pixelFrac)
        #Create a black border arouund each tmpImage    
        imSize = np.shape(tmpImg)
        tmpImg[0:round(.035*imSize[0])] = np.zeros((round(.035*imSize[0]),imSize[1]))
        # Use the first iteration to figure out the size of the total image containing all rows of the pitch test
        if n == 0:
            totImg = np.zeros((numTests*imSize[0],imSize[1]), dtype='float')
        totImg[n*imSize[0] + np.arange(imSize[0]),:] = tmpImg
    
    # Save the image, note here it is being saved as greyscale, with 1 bit per pixel - this should be all that is needed
    fName = 'H:/David/Google Drive/Canopy/Barrier Patterns/PitchTest_{}Views_{}x{}mm_{}x{}DPI_{}-{}-{}.png'.format(nViews, paperDim[0], paperDim[1], printerRes[0], printerRes[1],pxPitches[0], pxPitches[-1],pitchStep)
    png.from_array(totImg, 'L;1').save(fName)
    return totImg

# Create a series of barriers with changing pixel fraction
def multiFrac(paperDim, pixelPitch, printerPitch, nViews, pixelFrac):
    numTests = 9
    fracStep = 0.05
    lineDim = [divmod(paperDim[0],numTests)[0], paperDim[1]]
    # The range of different pixel fractions to create barriers of
    pxFracs = np.arange(pixelFrac,pixelFrac-fracStep*numTests,-fracStep)
    
    # Loop over the number of barriers to test, creating each barrier and assembling into 1 image
    for n in range(numTests):
        tmpImg = createBarrier(lineDim, printerPitch, pixelPitch, nViews, pxFracs[n])
        #Create a black border arouund each tmpImage    
        imSize = np.shape(tmpImg)
        tmpImg[0:round(.035*imSize[0]),:] = np.zeros((round(.035*imSize[0]),imSize[1]))
        # Use the first iteration to figure out the size of the total image containing all rows of the pitch test
        if n == 0:
            totImg = np.zeros((numTests*imSize[0],imSize[1]), dtype='float')
        totImg[n*imSize[0] + np.arange(imSize[0]),:] = tmpImg
    
    # Save the image
    fName = 'H:/David/Google Drive/Canopy/Barrier Patterns/FracTest_{}Views_{}x{}mm_{}x{}DPI_{:.2f}-{:.2f}-{:.2f}.png'.format(nViews, paperDim[0], paperDim[1], printerRes[0], printerRes[1],pxFracs[0], pxFracs[-1],fracStep)
    png.from_array(totImg, 'L;1').save(fName)
    return totImg


t0 = time.time()
singleBar = singleBarrier(paperDim, printerPitch, pixelPitch, nViews, pixelFrac)
barrierPlot(singleBar,1)    
#pitchTest = pitchTest(paperDim, pixelPitch, printerPitch, nViews, pixelFrac)
#barrierPlot(pitchTest,2)    
#multFrac = multiFrac(paperDim, pixelPitch, printerPitch, nViews, pixelFrac)
#barrierPlot(multFrac,3) 
t1 = time.time()
print("Running time = {:.2f}s".format(t1-t0))

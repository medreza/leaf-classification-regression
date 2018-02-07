## *Green leaf and dried leaf classification using distance of centroid-to-regression curve approach*
##
## First, this was student project from "Numerical Computation" class of Dept. of Electrical Eng. and Information Tech. UGM, Yogyakarta
## This approach should not be taken seriously as research-related project or 'good way of doing it'.
## The purpose of this project was to give the class insight of what regression can do.
## So we thought simple idea of classifying by comparing nearest distance of one's centroid to two distinct regression curve
## Then we determine which regression the centroid is closer to.
## If a point of centroid is closer to curve A than curve B, we classify that centroid as A, right? Yup that's the idea.
## These centroids are obtained from applying k-means clustering to every test image feature.
## Each image feature of test image can only have 1 centroid.
## Image feature used was Red and Green channel relation.
##
## Credit goes to all team members:
## - Ahmed Reza Rafsanzani
## - Awal Bakhtera Suhiyar
## - Indra Kurniawan
## - Nurman Setiawan
## - R. Cahya Hidayat
##


## Language used in the code (variable name, desc, etc.) was Bahasa Indonesia, we know it's not properly translated yet.
## What you need to note:
## Daun Hijau = Green leafs
## Daun Kering = Dried leafs

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans, MiniBatchKMeans
import shapely.geometry as geom

np.set_printoptions(threshold='nan')

print "---------------------------------------------------------------"
print "---Penerapan Regresi Linear dan Polynomial pada Citra Latih----"
print "---------------------------------------------------------------"


red1 = []
blue1 = []
green1 = []
red2 = []
blue2 = []
green2 = []
lst = []
lst2 = []
sizepic = 16
counter = 0

## -------------------- Training Daun Kering ------------------

citralatih_dk = ['012','013','014','015']
for cdk in citralatih_dk:
    imopen = Image.open('img\\training\\img'+cdk+'.jpg')
    im = imopen.resize( [int(0.125 * s) for s in imopen.size] )
    rgb_im = im.convert('RGB')
    for y in range(0, sizepic):
        row = ""
        for x in range(0, sizepic):
            RGB = rgb_im.getpixel((x,y))
            R,G,B = RGB
            counter+=1
            if counter%1 == 0: #stack all RGB value of all pixel to arrays
                red1.append(R)
                blue1.append(B)
                green1.append(G)

for xx in range(min(green1),max(green1)):
    lst.append(xx)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(green1, red1, c = 'r', marker='.')
ax1.set_title("Daun Hijau")
ax1.set_ylabel("Channel Red")
ax1.set_xlabel("Channel Green")

## ------------------- Training Daun Hijau -----------------------

citralatih_dh = ['001','002','003','004'] 

for cdh in citralatih_dh:
    counter = 0
    imopen = Image.open('img\\training\img'+cdh+'.jpg')
    im = imopen.resize( [int(0.125 * s) for s in imopen.size] )
    rgb_im = im.convert('RGB')
    for y in range(0, sizepic):
        row = ""
        for x in range(0, sizepic):

            RGB = rgb_im.getpixel((x,y))
            R,G,B = RGB
            counter+=1
            if counter%1 == 0: #stack all RGB value of all pixel to arrays
                red2.append(R)
                blue2.append(B)
                green2.append(G)

for xx in range(min(green2),max(green2)):
    lst2.append(xx)

    
ax2.scatter(green2, red2, c = 'r', marker='.')
ax2.set_title("Daun Kering")
ax2.set_ylabel("Channel Red")
ax2.set_xlabel("Channel Green")

#---------------------- Regresi Linear -------------------

dhcoord = np.column_stack((green1, red1)) #stack all of four training image feature of daun hijau
dkcoord = np.column_stack((green2, red2)) #stack all of four training image feature of daun kering

red1 = np.asarray(red1)
green1 = np.asarray(green1)
red2 = np.asarray(red2)
green2 = np.asarray(green2)
lst = np.asarray(lst)
lst2 = np.asarray(lst2)

#Linear regression for Daun Hijau
reg = linear_model.LinearRegression()
reg.fit(green1.reshape(len(green1),1),red1.reshape(len(green1),1))
print "Linear regression Formula of Daun Hijau Test Images: "
print str(reg.coef_[0][0])+" x + "+str(reg.intercept_[0])
X1=green1.reshape(len(green1),1)
Y1=red1.reshape(len(green1),1)
ax1.plot(green1.reshape(len(green1),1), reg.predict(green1.reshape(len(green1),1)),color='b')

#Linear regression for Daun Kering
reg2 = linear_model.LinearRegression()
reg2.fit(green2.reshape(len(green2),1),red2.reshape(len(green2),1))
print "Linear regression Formula of Daun Kering Test Images: "
print str(reg2.coef_[0][0])+" x + "+str(reg2.intercept_[0])
X2=green2.reshape(len(green2),1)
Y2=red2.reshape(len(red2),1)
ax2.plot(green2.reshape(len(green2),1), reg2.predict(green2.reshape(len(green2),1)),color='b')

#Print correlation coefficient
print "R^2 of Linear regression formula of Daun Hijau Test Images: "+str(reg.score(X1, Y1))
print "R^2 of Linear regression formula of Daun Kering Test Images: "+str(reg2.score(X2, Y2))

## -------------------------- Regresi Polynomial --------------------
polyhijau = np.polyfit(green1,red1, 2)
ppolyhijau = np.poly1d(polyhijau)
print "Polynomial regression Formula of Daun Hijau Test Images: "
print ppolyhijau
ax1.plot(lst, ppolyhijau(lst))

def polyfitx(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


polykering = np.polyfit(green2,red2, 2)
ppolykering = np.poly1d(polykering)
print "Polynomial regression Formula of Daun Kering Test Images:: "
print ppolykering
ax2.plot(lst2, ppolykering(lst2))

## Shortest distance from  centroid point to linear curve formula
def distance(p0, p1, p2):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    nom = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = ((y2 - y1)**2 + (x2 - x1) ** 2) ** 0.5
    result = nom / denom
    return result

hasil_ppolykering = []
hasil_ppolyhijau = []

for xa in ppolykering(lst):
    hasil_ppolykering.append(xa)

for xb in ppolyhijau(lst2):
    hasil_ppolyhijau.append(xb)

polycoord_dk = np.column_stack((lst.reshape(len(lst),1), hasil_ppolykering))
polycoord_dh = np.column_stack((lst2.reshape(len(lst2),1), hasil_ppolyhijau))

## Define regression curve as LineString from Shapely library..
## ..so we can calculate shortest distance from a centroid point to nonlinear curve using this library
linepolydk = geom.LineString(polycoord_dk)
linepolydh = geom.LineString(polycoord_dh)

citrauji = ['05','06','07','08','09','10','11','16','17','18','19','20','21','22']
print ""
print "---------------------------------------------------------"
print "----------------- Klasifikasi Citra Uji------------------"
print "---------------------------------------------------------"

## Classification
countercu = 0
listcentroid = np.empty((0,2))

for pg in citrauji:
    countercu += 1
    reduji = []
    greenuji = []
    counter = 0
    imopen = Image.open('img\\test\img0'+str(pg)+'.jpg')
    im = imopen.resize( [int(0.125 * s) for s in imopen.size] )
    rgb_im = im.convert('RGB')
    for y in range(0, sizepic):
        row = ""
        for x in range(0, sizepic):

            RGB = rgb_im.getpixel((x,y))
            R,G,B = RGB
            counter+=1
            if counter%1 == 0:
                reduji.append(R)
                greenuji.append(G)

    ujicoord = np.column_stack((greenuji, reduji))
    kmeans3 = KMeans(n_clusters=1, random_state=10,n_init=1).fit(ujicoord)
    centroiduji = kmeans3.cluster_centers_
    listcentroid = np.append(listcentroid, np.array([[centroiduji[0][0],centroiduji[0][1]]]), axis=0)
    print " "
    print " "
    print str(countercu)+". TEST FILE img0"+str(pg)+".jpg"
    print "Centroid Coordinate of Test Image = "+str(centroiduji[0][0])+", "+str(centroiduji[0][1])
    teslindaunhijau = distance((centroiduji[0][0], centroiduji[0][1]),(X1[0][0],Y1[0][0]), (X1[256][0],Y1[256][0]))
    teslindaunkering = distance((centroiduji[0][0], centroiduji[0][1]),(X2[0][0],Y2[0][0]), (X2[256][0],Y2[256][0]))
    print "Distance from test image centroid to nearest point of linear regression curve of daun hijau: "+str(teslindaunhijau)
    print "Distance from test image centroid to nearest point of linear regression curve of daun kering: "+str(teslindaunkering)
    if teslindaunkering<teslindaunhijau:
        print "-Reg. Linear: File img0"+str(pg)+".jpg classified as Daun Kering"
    else:
        print "-Reg. Linear: File img0"+str(pg)+".jpg classified as Daun Hijau"
    tespoldaunhijau =  geom.Point(centroiduji[0][0], centroiduji[0][1]).distance(linepolydh)
    tespoldaunkering =  geom.Point(centroiduji[0][0], centroiduji[0][1]).distance(linepolydk)
    print ""
    print "Distance from test image centroid to nearest point of poly regression curve of daun hijau: "+str(tespoldaunhijau)
    print "Distance from test image centroid to nearest point of poly regression curve of daun kering: "+str(tespoldaunkering)
    if tespoldaunkering<tespoldaunhijau:
        print "-Reg. Poly: File img0"+str(pg)+".jpg classified as Daun Kering"
    else:
        print "-Reg. Poly: File img0"+str(pg)+".jpg classified as Daun Hijau"

##------------------ Plot Persebaran Centroid Citra Uji -------------------
listcentroid = listcentroid.reshape(14,2)
plt.figure(2).canvas.set_window_title('Centroids Distribution')
plt.title("Distribution of Test Images' Centroids and their relation with Training Images' Regression Curve ")
plt.plot(green2.reshape(len(green2),1), reg2.predict(green2.reshape(len(green2),1)), label='Reg. Lin. Test Image of Daun Kering')
plt.plot(lst2, ppolykering(lst2),label='Reg. Pol. Test Image of Daun Kering')
plt.plot(green1.reshape(len(green1),1), reg.predict(green1.reshape(len(green1),1)), label='Reg. Lin. Test Image of Daun Hijau')
plt.plot(lst, ppolyhijau(lst),label='Reg. Pol. Test Image of Daun Hijau')
plt.legend(loc='upper left')
plt.xlabel('Green Channel')
plt.ylabel('Red Channel')
plt.scatter(listcentroid[:,0],listcentroid[:,1], color='r', marker=',')
for i, txt in enumerate(citrauji):
    plt.annotate(txt, (listcentroid[:,0][i],listcentroid[:,1][i]))


plt.show()

print ""
print ""
print polyfitx(green1,red1, 2)
print polyfitx(green2,red2, 2)


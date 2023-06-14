# -*- coding: utf-8 -*-
"""NAIPimg.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install raster_geometry

import ee

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()

ee.Initialize()

import requests
import json

uri = requests.get("https://api.globe.gov/search/v1/measurement/protocol/measureddate/userid/?protocols=land_covers&startdate=1995-01-01&enddate=2025-01-01&userid=8267248&geojson=TRUE&sample=FALSE")

dat = json.loads(uri.text)

dat.keys()

imgs = {}
coords = {}
acc = {}
hor_rot = {}

from pprint import pprint
## 120 deg
directions = ["North", "South", "East", "West", "Upward", "Downward"]

def split_extra(p):
  dd = "."
  d = dict()
  if "source: app" in p:
    p = p.replace("(source: app, ", "").replace("))", ")").replace("(", "{").replace(")", "}")
    for extra_param in ["compassData.horizon", "compassData.heading"]:
      p = p.replace(extra_param, f"'{extra_param.split(dd)[1]}'")
    d = eval(p)

  for extra_param in ["horizon", "heading"]:
    if extra_param not in d.keys():
      d[extra_param] = 0.0

  return d

for feature in dat["features"]:
  props = feature["properties"]
  acc[props["siteId"]] = {"m": props["landcoversLocationAccuracyM"], "at": props["landcoversMeasuredAt"], "method": props["landcoversLocationMethod"]}
  coords[props["siteId"]] = (props["landcoversMeasurementLatitude"], props["landcoversMeasurementLongitude"])
  img_urls = {x:props[x] for x in list(filter(lambda x: "PhotoUrl" in x, props))}
  imgs[props["siteId"]] = img_urls

  for direction in directions:
    params = split_extra(props[f'landcovers{direction}ExtraData'])
    if props["siteId"] not in hor_rot.keys():
      hor_rot[props["siteId"]] = {}
    hor_rot[props["siteId"]][direction[0]] = params

props["landcoversDownwardExtraData"]

imgs.keys()

dataset = ee.ImageCollection('USDA/NAIP/DOQQ').filter(ee.Filter.date('2014-01-01', '2024-12-31'))
elevation_dataset = ee.ImageCollection("USGS/3DEP/1m")
land_cover_10m_dataset = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
bounds = ee.Geometry.Rectangle([[-105.53,40.75],[-105.17,40.56]]);
import math

"""

*   https://www.cs.cornell.edu/projects/megadepth/
*   https://github.com/zhengqili/MegaDepth/tree/master
*   https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EA001175

"""

# Commented out IPython magic to ensure Python compatibility.
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))

START = 40 # index of imgs.keys() above to start at
N = 100 # number of sites to visualize
sat_chip_size = 2000

import matplotlib
import io, os
import gc
matplotlib.rcParams["figure.max_open_warning"] = 100
import matplotlib.pylab as plt
params = {'axes.titlesize':'x-large'}
plt.rcParams.update(params)
# %matplotlib inline
import numpy as np
# import matplotlib.pylab as plt
import math

from PIL import Image
import urllib
import pandas as pd
from matplotlib.patches import Circle

import numpy as np
import raster_geometry as rg
import cv2 as cv

def orbkp(site_id, imgg, fmt=None, depth=0, typ="None"):
  if depth > 10:
    print(f"depth exceeded: orbkp for: {site_id}")
    return None
  # try:
  if fmt == "npy":
    print(f"getting numpy img format from {imgg}")
    response = requests.get(imgg)
    tn = np.load(io.BytesIO(response.content)).astype(np.float32)
    os.makedirs(f"raw/{site_id}", exist_ok = True)
    np.save(f"raw/{site_id}/{typ}.npy", tn)

    rimg = tn.astype(np.uint8)
    # rimg = Image.fromarray(rimg)
    fig = plt.figure()
    plt.imshow(rimg)
    plt.savefig(f"raw/{site_id}/{typ}_fig.png")
    plt.close()
    rimg = Image.fromarray(rimg)

  elif fmt is None:
    response = requests.get(imgg)
    rimg = Image.open(io.BytesIO(response.content))




  orb = cv.ORB_create()
  imga = np.asarray(rimg)
  kp = orb.detect(imga, None)
  kp, des = orb.compute(imga, kp)
  img2 = cv.drawKeypoints(imga, kp, None, color=(255,0,0), flags=0)

  return img2, rimg

def imghist(imgg, ax=None):
  if isinstance(imgg, str):
    imgg = np.asarray(Image.open(requests.get(imgg, stream=True).raw))
  im = cv.cvtColor(imgg,cv.COLOR_BGR2HSV)


  hbins = 180
  sbins = 255
  hrange = [0,180]
  srange = [0,256]
  ranges = hrange+srange

  histbase = cv.calcHist(im,[0,1],None,[180,256],ranges)
  cv.normalize(histbase,histbase,0,255,cv.NORM_MINMAX)

  color = ('b','g','r')

  for i,col in enumerate(color):
    histr = cv.calcHist([im],[i],None,[256],[0,256])
    histr /= im.shape[0] * im.shape[1]
    if ax != None:
      ax.plot(histr,color = col)
  if ax != None:
    ax2=ax.twinx()
    ax2.plot(histbase / im.shape[0] * im.shape[1],color = "orange")

  return histbase

def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)

def full_triangle(a, b, c):
  ab = rg.bresenham_line(a, b, endpoint=True)
  for x in set(ab):
      yield from rg.bresenham_line(c, x, endpoint=True)

def get_points(width, height, theta):

    twoPI = math.pi * 2.0
    PI = math.pi
    theta %= twoPI

    aa = width
    bb = height

    rectAtan = math.atan2(bb,aa)
    tanTheta = math.tan(theta)

    xFactor = 1
    yFactor = 1

    # determine regions
    if theta > twoPI-rectAtan or theta <= rectAtan:
        region = 1
    elif theta > rectAtan and theta <= PI-rectAtan:
        region = 2

    elif theta > PI - rectAtan and theta <= PI + rectAtan:
        region = 3
        xFactor = -1
        yFactor = -1
    elif theta > PI + rectAtan and theta < twoPI - rectAtan:
        region = 4
        xFactor = -1
        yFactor = -1
    else:
        print(f"region assign failed : {theta}")
        raise

    # print(region, xFactor, yFactor)
    edgePoint = [0,0]
    ## calculate points
    if (region == 1) or (region == 3):
        edgePoint[0] += xFactor * (aa / 2.)
        edgePoint[1] += yFactor * (aa / 2.) * tanTheta
    else:
        edgePoint[0] += xFactor * (bb / (2. * tanTheta))
        edgePoint[1] += yFactor * (bb /  2.)

    return region, (int(edgePoint[0]), int(edgePoint[1]))

def extractFOV(img_arr, heading, deg_fov):
  # extract a cone facing in heading with spread deg_fov from img_arr
  r, a = get_points(img_arr.shape[0], img_arr.shape[1], theta=(heading - deg_fov) * math.pi/180)
  r, c = get_points(img_arr.shape[0], img_arr.shape[1], theta=(heading + deg_fov) * math.pi/180)
  b = (img_arr.shape[0]//2, img_arr.shape[1]//2)

  coords = set(full_triangle(a, b, c))
  print(coords)
  arr = rg.render_at(img_arr.shape, coords)
  arr = arr.astype(int)

  return img_arr[arr]


def comp(hists, sat, ax_arr, site_id):
  sat = np.asarray(Image.open(requests.get(sat, stream=True).raw))

  for n, direct in enumerate(["N", "S", "E", "W"]):
    print(hor_rot[site_id][direct])
    if hor_rot[site_id][direct]["heading"] != 0:
      sat = extractFOV(sat, hor_rot[site_id][direct]["heading"], 13).astype(np.uint8)
      Image.fromarray(sat).save("test.jpg")

    else:
      if direct == "N":
        # split sat to just N fov
        sat = sat[:sat.shape[0]//2, sat.shape[1]//5:4*sat.shape[1]//5]

      elif direct == "S":
        sat = sat[sat.shape[0]//2:, sat.shape[1]//5:4*sat.shape[1]//5]

      elif direct == "W":
        sat = sat[sat.shape[0]//5:4*sat.shape[0]//5, :sat.shape[1]//2]

      elif direct == "E":
        sat = sat[sat.shape[0]//5:4*sat.shape[0]//5, sat.shape[1]//2:]

    sat_hist = imghist(sat)
    diff = cv.compareHist(hists[direct],sat_hist,cv.HISTCMP_CORREL)
    print(direct, diff)

    if direct == "N":
      ax_arr[0, 1].title.set_text(f"correl {diff}")

    elif direct == "S":
      ax_arr[2, 1].title.set_text(f"correl {diff}")

    elif direct == "W":
      ax_arr[1, 0].title.set_text(f"correl {diff}")

    elif direct == "E":
      ax_arr[1, 2].title.set_text(f"correl {diff}")

for site_id in list(imgs.keys())[START:START+N]:
  gc.collect()
# for site_id in []:
  # init plot
  f, ax_arr = plt.subplots(3, 3, figsize=(25, 25))
  ff, ax_arr_hist = plt.subplots(3, 3, figsize=(25, 25))

  lat, lon = [float(x) for x in coords[site_id]]
  img = imgs[site_id]

  gps_acc = int(acc[site_id]["m"])
  toy = acc[site_id]["at"]

  R = 6378137

  dn = 160
  de = 160

  dLat = dn/R
  dLon = de/(R*math.cos(math.pi*lat/180))

  latO = dLat * 180/math.pi
  lonO = dLon * 180/math.pi

  bounds = ee.Geometry.BBox(lon-lonO, lat-latO, lon+lonO, lat+latO);
  year = int(toy.split("-")[0])
  found = False




  imgg = dataset.filterBounds(bounds).first()

  satprops = imgg.getInfo()

  imgg = imgg.getThumbURL({'dimensions': f"{sat_chip_size}", "region": bounds, "bands": ["R", "G", "B"], "format": "jpg", "region": bounds})

  img_dataset = dataset.filterBounds(bounds).first()
  elev_dataset = elevation_dataset.filterBounds(bounds)
  elev_dataset = elev_dataset.first()
  lc10m = land_cover_10m_dataset.filterBounds(bounds).first()


  satprops = img_dataset.getInfo()

  imgg = img_dataset.getThumbURL({'dimensions': f"{sat_chip_size}", "region": bounds, "bands": ["R", "G", "B"], "format": "jpg", "region": bounds})
  kpts_rgb, rimg_rgb = orbkp(site_id, imgg)

  fig = plt.figure()
  plt.imshow(np.asarray(rimg_rgb))
  os.makedirs(f"raw/{site_id}", exist_ok=True)
  np.save(f"raw/{site_id}/satellite_fig.npy", np.asarray(rimg_rgb))
  plt.savefig(f"raw/{site_id}/satellite_fig.png")
  plt.close()

  imgg_elev = elev_dataset.getDownloadURL({'dimensions': f"{sat_chip_size}", "region": bounds, "bands": ["elevation"], "format": "NPY"})
  kpts_elevation, rimg_elevation = orbkp(site_id, imgg_elev, fmt="npy", typ="elevation")

  imgg_lc = lc10m.getDownloadURL({'dimensions': f"{sat_chip_size}", "region": bounds, "bands": ["label"], "format": "NPY"})
  kpts_lc, rimg_lc = orbkp(site_id, imgg_lc, fmt="npy", typ="lc10m")

  sat = imgg
  st = pd.to_datetime(satprops["properties"]["system:time_start"], unit="ms")
  end = pd.to_datetime(satprops["properties"]["system:time_end"], unit="ms")

  ax_arr[1, 1].title.set_text(f"{st}:{end}")
  # imgg, rimg = orbkp(site_id, imgg)

  ax_arr[1, 1].imshow(kpts_rgb)

  ax_arr[1, 1].add_patch(Circle((sat_chip_size//2, sat_chip_size//2), radius=12, color='red'))
  ax_arr[1, 1].add_patch(Circle((sat_chip_size//2, sat_chip_size//2), radius=(sat_chip_size/dn)*gps_acc, color="blue", fill=False))
  ax_arr[0, 0].axis('off')
  ax_arr[2, 0].axis('off')

  ax_arr_hist[0, 0].axis('off')
  ax_arr_hist[2, 0].axis('off')

  hists = {}

  hists["sat"] = imghist(imgg, ax_arr_hist[1, 1])

  for i in img:
    kk, rimg_gp = orbkp(site_id, img[i])
    if i[10] == "D":
      ax_arr[2, 2].imshow(kk)
      hists["D"] = imghist(img[i], ax_arr_hist[2, 2])
    elif i[10] == "U":
      ax_arr[0, 2].imshow(kk)
      hists["U"] = imghist(img[i], ax_arr_hist[0, 2])
    elif i[10] == "N":
      ax_arr[0, 1].imshow(kk)
      hists["N"] = imghist(img[i], ax_arr_hist[0, 1])
    elif i[10] == "S":
      ax_arr[2, 1].imshow(kk)
      hists["S"] = imghist(img[i], ax_arr_hist[2, 1])
    elif i[10] == "W":
      ax_arr[1, 0].imshow(kk)
      hists["W"] = imghist(img[i], ax_arr_hist[1, 0])
    elif i[10] == "E":
      ax_arr[1, 2].imshow(kk)
      hists["E"] = imghist(img[i], ax_arr_hist[1, 2])

    os.makedirs(f"raw/{site_id}", exist_ok=True)
    plt.figure()
    plt.imshow(np.asarray(rimg_gp))
    # np.save(f"raw/{site_id}/{i[10]}_fig.npy", np.asarray(rimg_gp))
    plt.savefig(f"raw/{site_id}/{i[10]}_fig.png")
    plt.close()

  # comp(hists, sat, ax_arr, site_id)

  f.show()
  ff.show()



# from google.colab import drive
# drive.mount('/content/drive')

# display(rimg_elevation)

# from pprint import pprint
# pprint(np.asarray(rimg_elevation))

# elev = np.load("raw/288749/elevation.npy")
# pprint(elev)

# elev.shape

# display(Image.fromarray(elev.astype(np.uint8)))

# from matplotlib import cm

# x, y = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]

# ax = plt.gca(projection='3d')
# ax.plot_surface(x, y, elev[:,:], rstride=2, cstride=2, cmap=cm.jet)
# plt.show()

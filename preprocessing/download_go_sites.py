import os
import json
import numpy as np
import requests

directions = ["North", "South", "East", "West", "Upward", "Downward"]

def get_peder_json(force=False):
    if not os.path.exists("peder_sites.json") or force:
        cookies = {'rxVisitor': '1665795314013GD9N89GBBJAMTAFP8O2VAAQ62CF3FKN0','dtSa': '-','dtLatC': '1','dtCookie': 'v_4_srv_3_sn_ACB3D48CF324BB1335E85456A3B649B7_perc_100000_ol_0_mul_1_app-3A91c4300ee97a533c_1_app-3Acd4cdd7a69881165_1_app-3Af262232919e45f43_1_app-3A632d46d7945593bd_1','rxvt': '1667843478640|1667841678640',}

        headers = {
            'authority': 'api.globe.gov',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'accept-language': 'en-US,en;q=0.8',
            'cache-control': 'max-age=0',
            # Requests sorts cookies= alphabetically
            # 'cookie': f"rxVisitor=1665795314013GD9N89GBBJAMTAFP8O2VAAQ62CF3FKN0; dtSa=-; dtLatC=1; dtCookie=v_4_srv_3_sn_ACB3D48CF324BB1335E85456A3B649B7_perc_100000_ol_0_mul_1_app-3A91c4300ee97a533c_1_app-3Acd4cdd7a69881165_1_app-3Af262232919e45f43_1_app-3A632d46d7945593bd_1; dtPC=3{419738529_837h-vVDFLWFOECERPEAADGHMPMCSRFCRCFMKE-0e0;} rxvt=1667843478640|1667841678640",
            'sec-ch-ua': '"Brave";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'sec-gpc': '1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
        }

        params = {
            'protocols': 'land_covers',
            'startdate': '2020-01-01',
            'enddate': '2025-01-01',
            'userid': '8267248',
            'geojson': 'TRUE',
            'sample': 'FALSE',
        }

        uri = requests.get('https://api.globe.gov/search/v1/measurement/protocol/measureddate/userid/', params=params, cookies=cookies, headers=headers)
        dat = json.loads(uri.text)
        with open("peder_sites.json", "w") as f:
            json.dump(dat, f)
    else:
        with open("peder_sites.json", "r") as f:
          dat = json.load(f)
    return dat

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

def parse_peder_json(n):
    imgs = {}
    coords = {}
    acc = {}
    hor_rot = {}
    dat = get_peder_json()
    for feature in dat["features"][:n]:
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

    return imgs, coords, acc, hor_rot



import os
import glob
import json

# import pil img
from PIL import Image

import html
import numpy as np
import base64
import io
import re
from io import BytesIO

inpath = "HTMLS/"
GP_PATH = "GP/"
NAIP_PATH = "SAT/"
WORLDCOVER_PATH = "WORLDCOVER/"
ELV_PATH = "ELV/"
JSON_PATH = "JSONS/"
UNLABELED_PATH = "STRIP_PNGS/"

TEST_GP_PATH = "TEST_GP/"
TEST_NAIP_PATH = "TEST_SAT/"
TEST_WORLDCOVER_PATH = "TEST_WORLDCOVER/"
TEST_ELV_PATH = "TEST_ELV/"

OUTPUT_JSONS = "PARSED_JSONS/"


os.makedirs(UNLABELED_PATH, exist_ok=True)
os.makedirs(GP_PATH, exist_ok=True)
os.makedirs(NAIP_PATH, exist_ok=True)
os.makedirs(WORLDCOVER_PATH, exist_ok=True)
os.makedirs(ELV_PATH, exist_ok=True)
os.makedirs(JSON_PATH, exist_ok=True)

os.makedirs(TEST_GP_PATH, exist_ok=True)
os.makedirs(TEST_NAIP_PATH, exist_ok=True)
os.makedirs(TEST_WORLDCOVER_PATH, exist_ok=True)
os.makedirs(TEST_ELV_PATH, exist_ok=True)
os.makedirs(OUTPUT_JSONS, exist_ok=True)

outpath = "STRIP_PNGS/"

# os.makedirs(outpath, exist_ok=True)

# for html_path in glob.glob(inpath + "*.html"):
#     # extract the <img tag and load as base64 and save to png
#     with open(html_path, 'r', encoding='utf-8') as file:
#         content = file.read()

#         # Use regex to find the img tag
#         img_tag_pattern = re.compile(r'<img[^>]*src="data:image/[^;]+;base64,([^"]+)"', re.IGNORECASE)
#         match = img_tag_pattern.search(content)

#         if match:
#             img_data = match.group(1)
#             img_data = base64.b64decode(img_data)
#             img = Image.open(BytesIO(img_data))

#             # Save the image as PNG
#             img_name = os.path.splitext(os.path.basename(html_path))[0] + '.png'
#             img.save(os.path.join(outpath, img_name))
#         else:
#             print(f'No img tag found in {html_path}')

def get_pts(parsed):
    gp_pts = []
    sat_pts = []
    for pa, pb in zip(parsed[::2], parsed[1::2]):
        print(f"comparing {pa=} and {pb=}")
        if pa[1] < pb[1]:
            # pa is in gp img
            gp_pts.append(pa)
            pb_ = (pb[0], pb[1] - 5000)
            sat_pts.append(pb_)
        else:
            pa_ = (pa[0], pa[1] - 5000)
            sat_pts.append(pa_)
            gp_pts.append(pb)
    return gp_pts, sat_pts

def get_labeled():
    jsons = glob.glob(JSON_PATH + "*.json")

    images = ("ground_image_1", "naip_image_1", "worldcover_image_1", "elevation_image_1")

    # find matching images for the jsons
    for json_path in jsons:
        parsed = json.load(open(json_path))
        print(f"parsed: {parsed}")
        img_name = outpath + os.path.basename(json_path).replace(".json", ".png")
        img = Image.open(img_name)
        # plot the points over the image
        for point_pair in parsed:
            print(f"point_pair: {point_pair}")
            try:
                pa, pb = point_pair
            except ValueError:
                print("invalid point pair", point_pair)
                exit(0)
        img = np.asarray(img)
        print(img.shape, parsed)

        gp_img = img[:5000, :]
        naip_img = img[5000:10000, :]
        worldcover_img = img[10000:15000, :]
        elevation_img = img[15000:, :]
        gp_pts, sat_pts = get_pts(parsed)
        same_points = list(zip(gp_pts, sat_pts))
        print(f"gp_img: {gp_img.shape}, sat_img: {naip_img.shape}, worldcover_img: {worldcover_img.shape}, elevation_img: {elevation_img.shape}")

        gp_img = Image.fromarray(gp_img)
        naip_img = Image.fromarray(naip_img)
        worldcover_img = Image.fromarray(worldcover_img)
        elevation_img = Image.fromarray(elevation_img)

        gp_img.save(GP_PATH + os.path.basename(json_path).replace(".json", ".png"))
        naip_img.save(NAIP_PATH + os.path.basename(json_path).replace(".json", ".png"))
        worldcover_img.save(WORLDCOVER_PATH + os.path.basename(json_path).replace(".json", ".png"))
        elevation_img.save(ELV_PATH + os.path.basename(json_path).replace(".json", ".png"))

        # save json
        json.dump(same_points, open(OUTPUT_JSONS + os.path.basename(json_path), "w"))


def get_unlabeled():
    jsons = glob.glob(UNLABELED_PATH + "*.png")
    print(jsons)

    images = ("ground_image_1", "naip_image_1", "worldcover_image_1", "elevation_image_1")

    # find matching images for the jsons
    for json_path in jsons:
        img = Image.open(json_path)
        img = np.asarray(img)

        gp_img = img[:5000, :]
        naip_img = img[5000:10000, :]
        worldcover_img = img[10000:15000, :]
        elevation_img = img[15000:, :]
        # gp_pts, sat_pts = get_pts(parsed)
        # same_points = list(zip(gp_pts, sat_pts))
        # print(f"gp_img: {gp_img.shape}, sat_img: {naip_img.shape}, worldcover_img: {worldcover_img.shape}, elevation_img: {elevation_img.shape}")

        gp_img = Image.fromarray(gp_img)
        naip_img = Image.fromarray(naip_img)
        worldcover_img = Image.fromarray(worldcover_img)
        elevation_img = Image.fromarray(elevation_img)

        json_path = os.path.basename(json_path)

        gp_img.save(TEST_GP_PATH + json_path)
        naip_img.save(TEST_NAIP_PATH + json_path)
        worldcover_img.save(TEST_WORLDCOVER_PATH + json_path)
        elevation_img.save(TEST_ELV_PATH + json_path)

        # save json
        # json.dump(same_points, open(JSON_PATH + os.path.basename(json_path), "w"))


# def load_labels(json_paths):
#     output = []  # x1_g1, y1_g1, x1_s1, y1_s1, x2_g1, y2_g1, x2_s1, y2_s1, x3_g1, y3_g1, x3_s1, y3_s1, x4_g1, y4_g1, x4_s1, y4_s1, x5_g1, y5_g1, x5_s1, y5_s1
#     for json_path in json_paths:
#         p = json_path.split("/")[-1].replace(".json", ".png")
#         tmp = dict(pts=[], ground_image_1=p, naip_image_1=p, worldcover_image_1=p, elevation_image_1=p)
#         parsed = json.load(open(json_path))
#         for x in parsed:
#             for y in x:
#                 for z in y:
#                     tmp["pts"].append(z)
#         if len(tmp["pts"]) < 20:
#             tmp["pts"] += [0, 0, 0, 0] * (20 - len(tmp["pts"])) / 4
#         if len(tmp["pts"]) > 20:
#             tmp["pts"] = tmp["pts"][:20]
#         output.append(tmp)
#     return output

# print(load_labels(jsons))

if __name__ == "__main__":
    get_labeled()
    # get_unlabeled()
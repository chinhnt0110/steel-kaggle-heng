import os
import cv2
import csv
import numpy as np
import json
import urllib.request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
img_path = ROOT_DIR + '/data/scratch/train_images/'
anno_path = ROOT_DIR + '/data/scratch/'
saved_csv_path = ROOT_DIR + '/data/scratch/train.csv'


def load_json_from_dataturks():
    dataturks_data = []
    dataturks_path = anno_path + 'dataturks'
    for json_file in sorted(os.listdir(dataturks_path)):
        with open(os.path.join(dataturks_path, json_file)) as f:
            data = f.readlines()
            data = [img for img in data if '"scratch","big"' in img or '"big","scratch"' in img]
            dataturks_data.extend(data)
    return dataturks_data


def extract_params_from_dataturks(img):
    annotation = []
    img = json.loads(img)
    url = img['content']
    img_id = url.replace('/', '_')
    for anno in img['annotation']:
        if anno['label'] == ['scratch', 'big'] or anno['label'] == ['big', 'scratch']:
            annotation.append(anno)
    points = [anno['points'] for anno in annotation]
    w = h = 0
    try:
        w = annotation[0]['imageWidth']
        h = annotation[0]['imageHeight']
        for area in points:
            for point in area:
                point[0] *= w
                point[1] *= h
    except:
        print(url)
    return img_id, points, h, w


def load_json_from_scalabel():
    scalabel_data = []
    scalabel_path = anno_path + 'scalabel'
    for json_file in sorted(os.listdir(scalabel_path)):
        with open(os.path.join(scalabel_path, json_file)) as f:
            data = json.load(f)
            scalabel_data.extend(data)
    return scalabel_data


def extract_params_from_scalabel(img):
    points = []
    for anno in img['labels']:
        if anno['category'] == 'scratch':
            poly2d_data = anno['poly2d']
            for p in poly2d_data:
                points.append(p['vertices'])
    image_name = img['name'].replace('/', '_')
    img = cv2.imread(img_path + image_name)
    (h, w) = img.shape[:2]
    return image_name, points, h, w


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def download_images_from_s3():
    dataturks_data = load_json_from_dataturks()
    for index, line in enumerate(dataturks_data):
        img = json.loads(line)
        image = url_to_image(img['content'])
        image_name = img['content'].replace('/', '_')
        cv2.imwrite(img_path + image_name, image)
        print(index, len(dataturks_data))

    scalabel_data = load_json_from_scalabel()
    for index, img in enumerate(scalabel_data):
        count = 0
        for anno in img['labels']:
            if anno['category'] == 'scratch':
                count += 1
        if count == 0:
            continue
        image = url_to_image(img['name'])
        image_name = img['name'].replace('/', '_')
        cv2.imwrite(img_path + image_name, image)
        print(index, len(scalabel_data))


def convert_coordinates_to_masks(imageWidth, imageHeight, points):
    defect = []
    image = np.zeros((imageHeight, imageWidth), np.uint8)
    # print(points)
    for area in points:
        # print(area)
        defect.append(np.array(area, dtype=np.int32))

    cv2.fillPoly(image, defect, (255, 255))
    indices = np.argwhere(image == [255])
    return indices


def convert_masks_to_pixels(imageWidth, indices):
    position = []
    for index in indices:
        pos = index[0] * imageWidth + index[1] + 1
        position.append(pos)

    encoded_pixels = ''
    length = 0
    for i in range(0, len(position) - 1):
        if i == 0:
            start = position[i]
            length += 1
        elif position[i] == position[i - 1] + 1:
            length += 1
        elif position[i] != position[i - 1] + 1:
            encoded_pixels += str(start) + ' ' + str(length) + ' '
            start = position[i]
            length = 1

    encoded_pixels = encoded_pixels[:-1]
    # rle= [[imageId, classId, encoded_pixels]]
    return encoded_pixels


def write_to_csv(rle):
    with open(saved_csv_path, 'w') as f:
        writer = csv.writer(f)
        header = ['ImageId', 'ClassId', 'EncodedPixels']
        writer.writerow(header)
        for r in rle:
            writer.writerow(r)


def main():
    rle = []
    dataturks_data = load_json_from_dataturks()
    scalabel_data = load_json_from_scalabel()
    for img in dataturks_data:
        image_id, points, image_height, image_width = extract_params_from_dataturks(img)
        indices = convert_coordinates_to_masks(image_width, image_height, points)
        encoded_pixels = convert_masks_to_pixels(image_width, indices)
        class_id = '1'
        rle.append([image_id, class_id, encoded_pixels])

    for img in scalabel_data:
        image_name = img['name'].replace('/', '_')
        if os.path.isfile(img_path + image_name):
            image_id, points, image_height, image_width = extract_params_from_scalabel(img)
            indices = convert_coordinates_to_masks(image_width, image_height, points)
            encoded_pixels = convert_masks_to_pixels(image_width, indices)
            class_id = '1'
            rle.append([image_id, class_id, encoded_pixels])
    write_to_csv(rle)


if __name__ == '__main__':
    # download_images_from_s3()
    main()

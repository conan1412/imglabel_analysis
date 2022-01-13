# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
from xml.etree import cElementTree as ET
import argparse

def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

def plot_wh(labels, cls_n, save_dir=''):
    # plot dataset labels
    b = labels[:, :].transpose().astype(np.int32)  # boxes


    b = b[2]/(np.where(b[3]==0, 0.001, b[3]))
    b = np.where(b>100, 1, b)
    plt.hist(b, bins=[x/10 for x in range(50)], rwidth=0.8)
    plt.title("histogram")

    plt.grid(linestyle="-.", axis='y', alpha=0.4)
    plt.ylabel('nums', fontsize=12)
    plt.xlabel('w/h', fontsize=12)
    # 此处图像大小定死了，需要修改
    plt.suptitle('class: %s, mean w/h %f'
                 % (cls_n,
                    np.mean(b[2]/b[3]),
                    ), fontsize=14)
    plt.savefig(os.path.join(save_dir, 'wdivh_%s.png' % (cls_n)), dpi=200)
    plt.close()

def plot_labels(labels, cls_n, save_dir=''):
    # plot dataset labels
    b = labels[:, :].transpose().astype(np.int32)  # boxes

    # matplotlib labels
    ax = plt.subplots(2, 3, figsize=(16, 8), tight_layout=True)[1].ravel()

    ax[0].hist(b[0], bins=50, rwidth=0.8)
    ax[0].set_xlabel('x_center', fontsize=12)
    ax[0].set_ylabel('nums', fontsize=12)
    ax[1].hist(b[1], bins=50, rwidth=0.8)
    ax[1].set_xlabel('y_center', fontsize=12)
    ax[1].set_ylabel('nums', fontsize=12)
    ax[2].scatter(b[0], b[1], c=hist2d(b[0], b[1], 100), cmap='Greens')
    ax[2].set_xlabel('x_center', fontsize=12)
    ax[2].set_ylabel('y_center', fontsize=12)

    ax[3].hist(b[2], bins=50, rwidth=0.8)
    ax[3].set_xlabel('width', fontsize=12)
    ax[3].set_ylabel('nums', fontsize=12)
    ax[4].hist(b[3], bins=50, rwidth=0.8)
    ax[4].set_xlabel('height', fontsize=12)
    ax[4].set_ylabel('nums', fontsize=12)
    ax[5].scatter(b[2], b[3], c=hist2d(b[2], b[3], 100), cmap='Purples')
    ax[5].set_xlabel('width', fontsize=12)
    ax[5].set_ylabel('height', fontsize=12)

    #此处图像大小定死了，需要修改
    plt.suptitle('class: %s, img_w = %d, img_h = %d'
              '\nmean_x = %d, mean_y = %d, mean_w = %d, mean_h = %d'
                 # '\nmean_w_ratio = %.6f, mean_h_ratio = %.6f, mean_area_ratio = %.6f'
              % (cls_n, 1920, 1080,
                 int(np.mean(b[0])), int(np.mean(b[1])), int(np.mean(b[2])), int(np.mean(b[3])),
                 # int(np.mean(b[2]))/1280, int(np.mean(b[3]))/720, np.mean(b[2]) * np.mean(b[3]) / (1280*720)
                 ), fontsize=14)
    plt.savefig(os.path.join(save_dir, 'xywh_%s.png'%(cls_n)), dpi=200)
    plt.close()


def plot_classes(labels, save_dir=''):
    # plot dataset labels
    # 将类别转化为字典，{类别：数量}
    cs = {}
    for i in labels:
        cs[i] = len(labels[i])

    #对字典进行按值排序
    sort_val_dic_instance = dict(sorted(cs.items(), key=operator.itemgetter(1), reverse=True))  # 按照value值降序




    cls, nums = sort_val_dic_instance.keys(), sort_val_dic_instance.values()

    rects = plt.bar(cls, nums, color=['purple','red','deepskyblue','orange','tomato','greenyellow'], width=0.5)
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, 1.01 * height, '%s' % int(height), ha='center', va='bottom')

    plt.grid(linestyle="-.", axis='y', alpha=0.4)
    plt.title('classes-nums')
    plt.ylabel('nums', fontsize=12)
    plt.xlabel('classes', fontsize=12)
    plt.legend(rects, tuple(cls), fontsize=12)
    plt.savefig(os.path.join(save_dir, 'classes-nums.png'), dpi=200)
    plt.close()

# 读取xml，返回[[name, x, y, w, h]]的列表
def load_xml_resize(xmlfile):
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    xmlbox = []
    for obj in objs:
        name = obj.find('name').text
        xmin = int(obj.find('bndbox')[0].text)
        ymin = int(obj.find('bndbox')[1].text)
        xmax = int(obj.find('bndbox')[2].text)
        ymax = int(obj.find('bndbox')[3].text)
        x = int((xmin + xmax) / 2)
        y = int((ymin + ymax) / 2)
        w = xmax - xmin
        h = ymax - ymin
        xmlbox.append([name, x, y, w, h])
    return xmlbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/Users/videopls/Desktop/工业检测/初赛数据/Annotations_ori/', help='the xml dir', type=str)  # xml路径
    parser.add_argument('--save_dir', default='/Users/videopls/Desktop/工业检测/初赛数据/analyze/', help='where to save the figure', type=str)  # 保存路径
    args = parser.parse_args()

    root_dir = args.root_dir
    save_dir = args.save_dir
    # label_name = ['car', 'bus-truck', 'person', 'bike-motor', 'rider']
    # label_name = ['Crack', 'Net', 'AbnormalManhole', 'Pothole', 'Marking']
    lable_list = {'yhtp':[], 'lwxqp':[], 'jzzqyh':[], 'bhzxjz':[], 'tph':[]}
    xml_boxes = []

    print('开始读取xml，存成字典')
    # 把每个类别和对应的box存成一个字典，类似{'car': [[79, 329, 159, 169],...], 'bus-truck': [[79, 329, 159, 169],...]}
    for xmlname in os.listdir(root_dir):
        xml_box_list = load_xml_resize(os.path.join(root_dir, xmlname)) # [[name, x, y, w, h]]的列表
        for xml_box in xml_box_list:
            cls_name = xml_box[0] # 类别，如'car'
            # if cls_name not in ['traffic light']:
            box = xml_box[1:] # bbox，如[79, 329, 159, 169]
            if cls_name in lable_list:
                lable_list[cls_name].append(box)
            # if cls_name not in lable_list:
            #     lable_list[cls_name] = [box]
            # else:
            #     lable_list[cls_name].append(box)

    print('开始画类别-数量图')
    # 画类别-数量图
    plot_classes(lable_list, save_dir=save_dir)

    print('开始画每个类别的box分布图')
    ## 每个类别中的box分布图
    for cls_n in lable_list:
        plot_labels(np.array(lable_list[cls_n]), cls_n, save_dir=save_dir)
        xml_boxes.extend(lable_list[cls_n])
    #
    print('开始画所有类别的box分布图')
    plot_labels(np.array(xml_boxes), 'all', save_dir=save_dir)


    print('开始画每个类别的wh分布')
    for cls_n in lable_list:
        plot_wh(np.array(lable_list[cls_n]), cls_n, save_dir=save_dir)
        xml_boxes.extend(lable_list[cls_n])

    print('开始画所有类别的wh分布')
    plot_wh(np.array(xml_boxes), 'all', save_dir=save_dir)




import cv2
import os
import shutil

# visualize to check the points
# image = cv2.imread("volleyball/volleyball/maJM7QoN7-2935.532_1.jpg")
# with open("volleyball/volleyball/maJM7QoN7-2935.532_1.txt","r") as file:
#     for data in file:
#         data = data.strip()
#         data = data.split(" ")
#         label, xmin, ymin, width, height = data
#         print(xmin)
#         x_cent = float(xmin) * 768
#         y_cent = float(ymin) * 768
#         height = float(height)* 768
#         width = float(width) * 768
#         xmin = int(x_cent - width/2)
#         ymin = int(y_cent - height/2)
#         xmax = int(x_cent + width/2)
#         ymax = int(y_cent + height/2)
#         cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax),(0,0,255),thickness= 3)


# create a folder that fit yolov5_1 requirements
if os.path.isdir("volleyball_yolo"):
    shutil.rmtree("volleyball_yolo")
if not os.path.isdir("volleyball_yolo"):
    os.mkdir("volleyball_yolo")
    os.mkdir("volleyball_yolo/images")
    os.mkdir("volleyball_yolo/labels")
    os.mkdir("volleyball_yolo/images/train")
    os.mkdir("volleyball_yolo/labels/train")
    os.mkdir("volleyball_yolo/images/val")
    os.mkdir("volleyball_yolo/labels/val")

root = "volleyball/volleyball/"
data_files = os.listdir(root)
num_frames = len(data_files)
for i in range(1, num_frames):

    # 90% for train dataset and 10% for val dataset
    mode = "train" if i < int(num_frames * 0.9) else "val"

    if data_files[i].endswith(".txt"):
        source = os.path.join(root,data_files[i])
        destination = os.path.join("volleyball_yolo/labels/{}".format(mode),data_files[i])
        shutil.copyfile(source, destination)
    else:
        source = os.path.join(root, data_files[i])
        destination = os.path.join("volleyball_yolo/images/{}".format(mode),data_files[i])
        shutil.copyfile(source, destination)

print(len(os.listdir("volleyball_yolo/labels/train")))
print(len(os.listdir("volleyball_yolo/labels/val")))





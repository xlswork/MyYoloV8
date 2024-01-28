from PIL import Image
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# results = model.train(data=r'E:\Study\yolov8\datasets\game_model\data.yaml', epochs=100, imgsz=640)

model = YOLO(r"E:\Study\yolov8\runs\detect\train8\weights\best.pt")  # load a custom model

# # Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95

# Run batched inference on a list of images
results = model(['img1.jpg'])  # return a list of Results objects

# Process results list
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image
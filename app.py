import gradio as gr

from ultralytics import YOLO
model = YOLO('best.pt')  # load your custom trained model
import torch
#from ultralyticsplus import render_result
from render import custom_render_result
def yoloV8_func(image: gr.Image = None,
                image_size: int = 640,
                conf_threshold: float = 0.4,
                iou_threshold: float = 0.5):
    # Load the YOLOv8 model from the 'best.pt' checkpoint
    model_path = "yolov8n.pt"
    # model = torch.hub.load('ultralytics/yolov8', 'custom', path='/content/best.pt', force_reload=True, trust_repo=True)

    # Perform object detection on the input image using the YOLOv8 model
    results = model.predict(image,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            imgsz=image_size)

    # Print the detected objects' information (class, coordinates, and probability)
    box = results[0].boxes
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)

    # Render the output image with bounding boxes around detected objects
    render = custom_render_result(model=model, image=image, result=results[0])
    return render


inputs = [
    gr.Image(type="filepath", label="Input Image"),
    gr.Slider(minimum=320, maximum=1280, step=32, label="Image Size", value=640),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Confidence Threshold"),
    gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="IOU Threshold"),
]

outputs = gr.Image(type="filepath", label="Output Image")

title = "YOLOv8 For Traffic Sign, person and Cars Detection 🚦🚶🚗"



yolo_app = gr.Interface(
    fn=yoloV8_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    cache_examples=False,
)

# Launch the Gradio interface in debug mode with queue enabled
yolo_app.launch(debug=True, share=True).queue()

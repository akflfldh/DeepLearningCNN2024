from flask import Flask,render_template,jsonify,request,make_response
import io
import subprocess
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import base64
from main import Net,transform
import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("./cifar_net.pth"))
model.eval()


app=Flask(__name__)


@app.route('/',methods=["GET"])
def index():
    return render_template("input.html")


@app.route('/upload',methods=["POST"])
def upload():
    if 'image' not in request.files:
        return "No image selected!"
    
    #Get the uploaded image file
    image_file=request.files['image']   #file storage
    
     # Check if a valid image file
    if image_file.filename.lower().endswith(('.png', '.jpg', '.gif', '.jpeg')):
        # Save the image to a designated folder (implement security checks)
        image_data = io.BytesIO(image_file.read())
        from PIL import Image  # Install Pillow library: pip install Pillow
        image = Image.open(image_data)
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally

        # Save the edited image back to the in-memory buffer
        buffer = io.BytesIO()
        flipped_image.save(buffer, format=image.format)

        # Set response headers for image content
        response = make_response(buffer.getvalue())
        response.headers.set('Content-Type', f'image/{image.format}')
        response.headers.set('Content-Disposition', 'attachment; filename=flipped_image.' + image.format)
        return response
    else:
        return "Invalid image format!"

@app.route("/predict",methods=["POST"])
def predict():
    #이미지가져오고, 모델의 predict사용해서 분류하여 그결과출력하면된다.
    #prediction결과를 매개변수로보내서 출력하면된다.
    if 'image' not in request.files:
        return "No image selected!"

    #Get the uploaded image file
    image_file=request.files['image']   #file storage

    if image_file.filename.lower().endswith(('.png', '.jpg', '.gif', '.jpeg','bmp')):
        image = Image.open(image_file).convert('RGB')
       # transform = transforms.Compose([
        #    transforms.Resize((256, 32)),
         #   transforms.ToTensor(),
          #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #])
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)

        # 예측된 클래스
        class_idx = predictions.argmax(dim=1).item()
        classname=str()
        if class_idx==0:
            classname="cat"
        else:
            classname="dog"
        print(f"Predicted class: {classname}")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()


    return render_template("input.html", className=classname,imageData=img_str)


if __name__ =='__main__':
    app.run()

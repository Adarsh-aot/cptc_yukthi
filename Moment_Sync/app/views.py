from django.shortcuts import render     

# Create your views here.
from django.shortcuts import render, redirect
from .models import UploadedImage , Uploaded , UserProfile , imageimage
from datetime import datetime
# def upload_image(request):
#     if request.method == 'POST':
#         images = request.FILES.getlist('images')  # Get a list of uploaded images
#         title = request.POST['title']
#         date = datetime.now().strftime('%Y-%m-%d %H:%M')
#         user_id = request.user.id
        
#         for image1 in images:
#             UploadedImage.objects.create(title=title, date=date, image=image1, u_id=user_id)

        
#         return redirect(upload_success)  # Redirect to a success page
#     return render(request, 'app/upload.html')

from datetime import datetime
from .models import UploadedImage, Uploaded
from django.shortcuts import render, redirect
from deepface import DeepFace

def upload_image(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')  # Get a list of uploaded images
        title = request.POST['title']
        date = datetime.now().strftime('%Y-%m-%d %H:%M')
        user_id = request.user.id
        
        for image1 in images:
            UploadedImage.objects.create(title=title, date=date, image=image1, u_id=user_id)

        # Perform image verification and save verified images
        try:
            uploaded_image = UserProfile.objects.all()
            for image in UploadedImage.objects.all():
                for i in uploaded_image:
                    print(len(uploaded_image))
                    result = DeepFace.verify(img1_path=image.image.path, img2_path=i.profile_picture.path)
                    if result['verified'] and result['distance'] < 0.5:
                        try:
                            print(result['distance'])
                            Uploaded.objects.create(title=image.title, date=image.date, image=image.image, k_id=i.user.id, un=str(image.id) + '_' + str(i.user.id))
                        except Exception as e:      
                            print(e)
        except Exception as e:
            print(e)

        return redirect(verified_image, request.user.id)  # Redirect to verified image page
    return render(request, 'app/upload.html')


def upload_success(request):
    return render(request, 'app/success.html')



# from django.http import HttpResponse , JsonResponse
# from django.shortcuts import render
# from PIL import Image
# import os
# from deepface import DeepFace


# def check_image(request):
#     if request.method == 'POST':
#         uploaded_image = request.FILES['image']
#         uploaded_image_path = os.path.join(r'C:\Users\aotir\OneDrive\Documents\GitHub\cptc_yukthi\media', uploaded_image.name)  # Adjusted path
        
#         # Ensure that the directory where the uploaded image should be saved exists
#         os.makedirs(os.path.dirname(uploaded_image_path), exist_ok=True)

#         with open(uploaded_image_path, 'wb+') as destination:
#             for chunk in uploaded_image.chunks():
#                 destination.write(chunk)

#         dataset_path = r'C:\Users\aotir\OneDrive\Documents\GitHub\cptc_yukthi\Moment_Sync\images'  # Adjust the path to your dataset
#         results = []
#         n = UploadedImage.objects.all()
#         for filename in os.listdir(dataset_path):
#             if filename.endswith('.jpg') or filename.endswith('.png'):
#                 image_path = os.path.join(dataset_path, filename)
#                 result = DeepFace.verify(img1_path=image_path, img2_path=uploaded_image_path)
#                 print(filename)
#                 results.append({
#                     'image': filename,
#                     'result': result['verified']
#                 })
        

#         return JsonResponse(results, safe=False)

#     return render(request, 'app/file.html')


import os
import zipfile
from django.http import FileResponse
from django.core.files import File
from django.shortcuts import render
from deepface import DeepFace
from .models import Uploaded
from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login


def check_image(request):
    if request.method == 'POST':
        id = request.user
        print(id)

        
                

        return redirect (verified_image , request.user.id )
        
        
    return render(request, 'app/file.html')


def verified_image(request , id = id ):
    n = Uploaded.objects.filter(k_id = id)
    return render ( request , 'app/verified_images.html' , {'n' : n })



from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        profile_picture = request.FILES.get('profile_picture')

        # Create a new user
        user = User.objects.create_user(username=username, password=password)

        # Save the profile picture (if provided)
        if profile_picture:
            # Handle profile picture upload here
            user_profile = UserProfile(user=user, profile_picture=profile_picture)
            user_profile.save()
            

        # Authenticate and login the user
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Redirect to success page or homepage
            return redirect(home)

    return render(request, 'app/register.html')


def home(request):
    user = request.user.id
    return render ( request , 'app/home.html' , {'n' : user })

def myimage(request , id  = id ):
    p = UploadedImage.objects.filter(u_id = id)
    user = request.user.id
    return render (request , 'app/myimage.html' , {'p' : p , 'id' : user})

def delete_pic(request , id = id ):
    n = Uploaded.objects.filter(id = id ).first()
    n.delete()
    return redirect ( verified_image , request.user.id )

def delete_picc(request , id = id ):
    n = UploadedImage.objects.filter(id = id ).first()
    n.delete()
    return redirect ( myimage , request.user.id )


def delete_p(request , id = id ):
    n = Uploaded.objects.filter(id = id ).first()
    n.delete()
    return redirect ( f_man )




def loginn(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        request.session['un'] = 1000

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Redirect to success page or homepage
            return redirect(delay) 
    return render ( request , 'app/login.html')

from django.http import HttpResponse
def camera(request):
    return render(request, 'app/camera.html')
import base64
from django.core.files.base import ContentFile
from .models import UploadedImage
from datetime import datetime

def upload_p(request):
    if request.method == 'POST':
        image_data_uri = request.POST.get('image_data')
        title = request.POST.get('title')
        date = datetime.now()
        user_id = request.user.id

        if image_data_uri:
            # Extract the base64-encoded data part from the data URL
            image_data_base64 = image_data_uri.split(',')[1]
            # Decode the base64-encoded string into binary image data
            image_data = base64.b64decode(image_data_base64)
            # Create a ContentFile object from the binary image data
            image_content = ContentFile(image_data, name=f'{title}.jpg')
            # Create an UploadedImage object and save it to the database
            uploaded_image = UploadedImage.objects.create(title=title, date=date, image=image_content, u_id=user_id)
            
            return redirect('camera')
        else:
            # Handle case where image data is not provided
            pass

    return HttpResponse('Method not allowed', status=405)


def past_images(request):
    if request.method == 'POST':
        id = request.user
        print(id)

        uploaded_image = UserProfile.objects.filter(user=request.user).first()

        verified_images = []

        # Assuming you have a model named UploadedImage with a field named image
        images = UploadedImage.objects.all()

        for image in images:
            print(image.u_id)
            result = DeepFace.verify(img1_path=image.image.path, img2_path=uploaded_image.profile_picture.path)

            if result['verified'] and result['distance'] < 0.5:
                print(result['distance'])
                print("yes")
                verified_images.append(image.image.name)
                try:
                    # Concatenate the user ID and image ID as strings
                    un = str(image.id) + '_' + str(request.user.id)
                    Uploaded.objects.create(title=image.title, date=image.date, image=image.image, k_id=request.user.id, un=un)
                except Exception as e:
                    print(e)

        return redirect(verified_image, request.user.id)

        
                
    return render(request, 'app/fil.html')



import os
import requests
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings


import os
import requests
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings

def upload_file(request):
    if request.method == 'POST':
        # Get the uploaded file from the request
        uploaded_file = request.FILES['file']
        
        # Set the API endpoint URL
        api_url = 'http://127.0.0.1:8000/uploadfile/'
        
        # Create a dictionary with the file data
        files = {'file': uploaded_file}
        
        # Send a POST request to the API endpoint with the file data
        response = requests.post(api_url, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the response content to a text file
            response_text = response.text
            file_path = os.path.join(settings.MEDIA_ROOT, 'output.txt')
            with open(file_path, 'w') as file:
                file.write(response_text)
            
            # Redirect to a new page with a download button for the text file
            return redirect(download_file)
        else:
            return HttpResponse('Error uploading file to the API!')
    else:
        return render(request, 'app/upload_file.html')
    

from django.http import HttpResponse
from django.conf import settings
import os

def download_file(request):
    # Get the path to the text file
    file_path = os.path.join(settings.MEDIA_ROOT, 'output.txt')
    
    # Open the text file for reading
    with open(file_path, 'rb') as file:
        # Create an HTTP response with the file content as attachment
        response = HttpResponse(file.read(), content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename=output.txt'
    
    # Delete the temporary text file
    os.remove(file_path)
    
    return response



from django.shortcuts import render
import cv2
import numpy as np

def apply_filter(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image from the request
        uploaded_image = request.FILES['image'].read()

        # Convert the uploaded image data to a numpy array
        nparr = np.frombuffer(uploaded_image, np.uint8)

        # Decode the numpy array to an OpenCV image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Apply a Gaussian blur filter
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Encode the filtered image back to JPEG format
        _, encoded_image = cv2.imencode('.jpg', blurred_image)

        # Convert the encoded image data to bytes
        blurred_image_bytes = encoded_image.tobytes()

        # Pass the filtered image data to the template
        return render(request, 'app/filtered_image.html', {'image_data': blurred_image_bytes})

    return render(request, 'app/upload_f.html')


def f_man(request):
    n = Uploaded.objects.all()
    return render ( request , 'app/f_man.html' , {'n' : n})


import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def question_answer(question, text):
    
    #tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
                
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
    print("\nPredicted answer:\n{}".format(answer.capitalize()))
    return "\nPredicted answer:\n{}".format(answer.capitalize())


def query(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        image_data_uri = request.POST.get('question')
        api_url = 'http://127.0.0.1:8000/uploadfile/'

        # Create a dictionary with the file data
        files = {'file': uploaded_file}

        # Send a POST request to the API endpoint with the file data
        response = requests.post(api_url, files=files)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Extract the response content (assuming it's JSON)
            response_data = response.json()

            # Access the specific data you need from the response
            result = response_data.get('Result', '')
            
            # Process the result as needed
            output = question_answer(result, image_data_uri)
            
            return render(request, 'app/query.html', {'output': output})

        else:
            # Handle the case where the request was not successful
            return HttpResponse('Error: Failed to upload file to the API')

    # Handle GET requests or other cases
    return render(request, 'app/query.html', {'output': ''})

import io
import torch
from PIL import Image
import torchvision.transforms as transforms
from django.http import HttpResponse
from django.shortcuts import render

norm_layer = torch.nn.InstanceNorm2d

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      torch.nn.ReLU(inplace=True),
                      torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)]

        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [torch.nn.ReflectionPad2d(3),
                  torch.nn.Conv2d(input_nc, 64, 7),
                  norm_layer(64),
                  torch.nn.ReLU(inplace=True)]
        self.model0 = torch.nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [torch.nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                       norm_layer(out_features),
                       torch.nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = torch.nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = torch.nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [torch.nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       norm_layer(out_features),
                       torch.nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = torch.nn.Sequential(*model3)

        # Output layer
        model4 = [torch.nn.ReflectionPad2d(3),
                  torch.nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [torch.nn.Sigmoid()]

        self.model4 = torch.nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out

# Load models
model1 = Generator(3, 1, 3)
model1.load_state_dict(torch.load(r'C:\Users\aotir\OneDrive\Documents\GitHub\cptc_yukthi\Moment_Sync\model.pth', map_location=torch.device('cpu')))
model1.eval()

model2 = Generator(3, 1, 3)
model2.load_state_dict(torch.load(r'C:\Users\aotir\OneDrive\Documents\GitHub\cptc_yukthi\Moment_Sync\model2.pth', map_location=torch.device('cpu')))
model2.eval()

from django.core.files.base import ContentFile

def predict(request):
    if request.method == 'POST':
        # Get input image and version from the request
        input_img = request.FILES['file']
        ver = request.POST.get('version')

        # Load the input image
        input_image = Image.open(input_img)

        # Transform the input image
        transform = transforms.Compose([transforms.Resize(256, Image.BICUBIC), transforms.ToTensor()])
        input_image = transform(input_image)
        input_image = torch.unsqueeze(input_image, 0)

        # Perform prediction based on the selected version
        with torch.no_grad():
            if ver == 'Simple Lines':
                drawing = model2(input_image)[0].detach()
            else:
                drawing = model1(input_image)[0].detach()

        # Convert the output to a PIL image
        drawing = transforms.ToPILImage()(drawing)

        # Save the drawing image to the database
        drawing_bytes = io.BytesIO()
        drawing.save(drawing_bytes, format='PNG')
        drawing_bytes = drawing_bytes.getvalue()

        # Create a new ImageImage object and save the image data
        drawing_instance = imageimage(image=ContentFile(drawing_bytes, name='drawing.png'))
        drawing_instance.save()

        # Return a response or redirect as needed
        return render(request, 'app/l.html' , {'n' : drawing_instance})
    else:
        # Handle GET requests (if necessary)
        return render(request, 'app/q.html')
    


def delay(request):
    return render ( request , 'app/delay.html')
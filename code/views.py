from django.shortcuts import render
from django.views import View
from django.http.response import HttpResponseRedirect
from .file_upload import upload_file
import os
from django.core.files.storage import FileSystemStorage
from .utils import cvrt2rgb
import PIL


# Create your views here.
class ImageUploadView(View):
    def get(self,request):
        return render(request,'base.html')
    
    def post(self,request):
        
        if not request.FILES.get('animal'):
            context = {'validation_error':'Image is required'}
            return render(request,'base.html',context)
        file_url = upload_file(request.FILES.get('animal'))
        colored_img = cvrt2rgb(file_url)
   
        list_of_names = []
        FileSystemStorage().delete(file_url)
        return render(request,'base.html',{'colored_img':colored_img})

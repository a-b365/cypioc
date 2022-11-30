from xml.etree.ElementInclude import include
from django.contrib import admin
from django.urls import path,include

app_name = 'code'

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('',include('code.urls',namespace='code'))
]

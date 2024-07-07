from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video, name='upload_video'),
    path('annotate/<int:video_id>/', views.annotate_frame, name='annotate_frame'),
    path('inference/<int:video_id>/', views.get_inference, name='get_inference'),
    path('view/<int:video_id>/', views.view_annotated_video, name='view_annotated_video'),
    path('convex_hull/', views.convex_hull, name='convex_hull'),
    path('store_data/<int:video_id>/', views.store_data, name='store_data'),
    path('get_fps/<int:video_id>/', views.get_fps, name='get_fps'),
    path('getresult/<int:video_id>/', views.get_result, name='get_result'),
    path('getoutput/<int:video_id>/', views.get_output, name='get_output'),
    path('getoutputframe/<int:video_id>/<int:frame_number>/', views.get_output_frame, name='get_output_frame'),
    path('videoname/<int:video_id>/', views.get_output_file, name='get_video_name'),
    path('output/<int:video_id>/', views.get_output_video, name='get_output_video'),
    path('outputpage/<int:video_id>/', views.get_output_page, name='get_output_page'),
    # Add more paths as needed for additional functionalities
]
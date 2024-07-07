import cv2
import os
import base64
import torch
from scipy.spatial import ConvexHull
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from .models import Video, Queue, PolygonAnnotation, OutputVideo, Inference
import json
from .forms import VideoForm
from django.conf import settings
from django.urls import reverse
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from collections import defaultdict
import numpy as np
import time
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            return redirect('annotate_frame', video_id=video.id)
    else:
        form = VideoForm()
    return render(request, 'upload.html', {'form': form})


def annotate_frame(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    
    # Path to the uploaded video file (assuming you have stored the video in 'media' directory)
    video_path = video.video_file.path
    
    # Logic to extract the 30th frame using OpenCV
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 29)  # 0-based index, so 30th frame is at index 29
    ret, frame = cap.read()
    cap.release()
    
    # Convert the frame to base64 to pass to the template
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    if request.method == 'POST':
        # Handle form submission to save polygon coordinates
        frame_number = 0  # Assuming you're always showing the 30th frame
        coordinates_json = request.POST.get('coordinates')
        coordinates = json.loads(coordinates_json)
        
        # Save each polygon annotation
        for polygon_coords in coordinates:
            PolygonAnnotation.objects.create(video=video, frame_number=frame_number, coordinates=json.dumps(polygon_coords))
        # Queue.objects.create(video=video)

        
        # Redirect to another view or page after saving (example: get_inference view)
        return redirect('get_inference', video_id=video.id)
    
    return render(request, 'annotate.html', {'video': video, 'frame_base64': frame_base64})


def store_data(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    if request.method == 'POST':
        data = json.loads(request.body)
        frame_number = data.get('frame_number', 0)  # Default frame number to 0
        boxes = data.get('boxes')
        
        queue, created = Queue.objects.get_or_create(video=video)
        queue.enqueue(frame_number, boxes)
        return JsonResponse({'message': 'Data stored successfully'})
    return JsonResponse({'error': 'Invalid request'}, status=400)


def convex_hull(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        points = data.get('points')
        hull = ConvexHull(points)
        hull_points = [points[i] for i in hull.vertices]
        return JsonResponse(hull_points, safe=False)

def get_inference(request, video_id):
    video = Video.objects.get(id=video_id)
    # annotations = PolygonAnnotation.objects.filter(video=video)
    curr_queue, created = Queue.objects.get_or_create(video=video)
    print(curr_queue.display())
    annotations = curr_queue.front()['boxes']
    print(annotations)
    return render(request, 'video.html', {'video': video, 'annotations': annotations})

def view_annotated_video(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    if request.method == 'POST':
        data = json.loads(request.body)
        frame_number = data.get('frame_number', 0)
        boxes = data.get('boxes')
        
        queue, created = Queue.objects.get_or_create(video=video)
        queue.enqueue(frame_number, boxes)
        print("Data stored successfully")
        return JsonResponse({'redirect_url': reverse('get_inference', args=[video.id])})
    return JsonResponse({'error': 'Invalid request'}, status=400)




# route to get fps of the video
def get_fps(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    video_path = video.video_file.path  # Assuming Video model has a 'video_file' FileField

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JsonResponse({'error': 'Cannot open video file.'}, status=500)

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return JsonResponse({'fps': fps})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Function to convert as original coordinates
def convert_coordinates(coordinates, width, height):
    for coord in coordinates:
        coord[0] = int(coord[0] * width/800)
        coord[1] = int(coord[1] * height/500)
    return coordinates

# views to get result using yolo

def get_result(request, video_id):
    st_time = time.time()
    video = Video.objects.get(id=video_id)
    filename = video.video_file.name
    filename = filename.split('/')[-1]
    video_path = video.video_file.path  # Assuming Video model has a 'video_file' FileField

    # Some of the parameters
    classes = [0]
    track_history = defaultdict(list)
    line_thickness = 2
    track_thickness = 2
    region_thickness = 2
    if torch.cuda.is_available():
        device = '0'
    else:
        device = 'cpu'
    model = YOLO('yolov8n.pt')
    
    # Getting the queue object for the video
    queue, created = Queue.objects.get_or_create(video=video)
    data = queue.dequeue()
    if data is None:
        return JsonResponse({'error' : 'Video not annotated or it may be already processed'})
    boxes = data['boxes']
    frame_number = data['frame_number']
            
    vc = cv2.VideoCapture(video_path)
    fh = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fw = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    counting_regions = []
    frame_no = 0
    if(frame_no == frame_number):
        for idx, box in enumerate(boxes):
            rec = {
                "name": f"box{idx}",
                "polygon": Polygon(convert_coordinates(box, fw, fh)),  # Polygon points
                "counts": 0,
            }
            counting_regions.append(rec)
    # video writer for saving the video
    filename = filename.split('.')[0] + '.webm'
    video_writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'VP80'),
        30,
        (int(fw), int(fh)),
    )
    output, created = Inference.objects.get_or_create(video=video)
    if(not created):
        return JsonResponse({'error': 'Inference already exists for the video'})
    names = model.model.names
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        # Check the queue for any updation in the plygon
        queue, created = Queue.objects.get_or_create(video=video)
        if not queue.is_empty():
            data = queue.front()
            # print(data)
            if data['frame_number'] <= frame_no:
                print("Detected changes in the polygon..+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                boxes = data['boxes']
                for idx, box in enumerate(boxes):
                    counting_regions[idx]["polygon"] = Polygon(convert_coordinates(box, fw, fh))
                queue.dequeue()
                print("Polygon updated successfully++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        results = model.track(frame, persist=True, classes=classes, device=device)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # Check if detection inside region
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            # region_color = region["region_color"]
            region_color = (0, 255, 0)
            # region_text_color = region["text_color"]
            region_text_color = (0, 0, 0)

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
        video_writer.write(frame)
        output_list = []
        for region in counting_regions:  # Reinitialize count for each region
            output_list.append(region["counts"])
            region["counts"] = 0
        # Adding result to Inferenece model
        output.add_inference(frame_no, output_list)

        frame_no += 1
    vc.release()
    video_writer.release()
    # Saving output video to the database
    output_video = OutputVideo(video=video)
    output_video.output_file.save(filename, open(filename, 'rb'))
    # Deleting the file after saving
    os.remove(filename)
    ed_time = time.time()
    print(f"Time taken for inference: {ed_time-st_time}++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return JsonResponse({'message': 'Video processed successfully'})

# Function to send the inference data to the frontend
def get_output(request, video_id):
    video = Video.objects.get(id=video_id)
    inference = Inference.objects.get(video=video)
    data = inference.get_inference()
    return JsonResponse(data)

# Function to send the inference data for every frame to the frontend
def get_output_frame(request, video_id, frame_number):
    video = Video.objects.get(id=video_id)
    inference = Inference.objects.get(video=video)
    # print(inference.get_inference())
    # print(frame_number,"+++++++++++++++++++++++++++++++")
    data = inference.get_frame_result(frame_number)
    # print(data)
    if data == 'KeyError':
        return JsonResponse({'error': 'Frame not found'})
    return JsonResponse({'frame' : frame_number, 'count': data})

# Function to get the output video file name
def get_output_file(request, video_id):
    video = Video.objects.get(id=video_id)
    output_video = OutputVideo.objects.get(video=video)
    return JsonResponse({'output_video': output_video.output_file.name})

def get_output_video(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    output_video = get_object_or_404(OutputVideo, video=video)
    context = {
        'output_video_url': output_video.output_file.url
    }
    return render(request, 'output.html', context)

# # Function to redirect to the output video page
def get_output_page(request, video_id):
    return redirect('get_output_video', video_id=video_id)
        

        


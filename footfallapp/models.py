from django.db import models
from collections import deque

class Video(models.Model):
    title = models.CharField(max_length=100)
    video_file = models.FileField(upload_to='videos/')

class PolygonAnnotation(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    frame_number = models.IntegerField()
    coordinates = models.TextField()  # JSON-encoded list of points

# creating model with queue data structure which stores the data in the form {frame_number: <frame_no>, box<i>: <box_coordinates>(i=1,2,3,4...)}
class Queue(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    data = models.JSONField(default=list)

    def enqueue(self, frame_number, boxes):
        self.data.append({'frame_number': frame_number, 'boxes': boxes})
        self.save()

    def dequeue(self):
        if self.data:
            item = self.data.pop(0)
            self.save()
            return item
        return None
    
    def front(self):
        if self.data:
            return self.data[0]
        return None
    
    def display(self):
        return self.data

    def size(self):
        return len(self.data)

    def is_empty(self):
        return len(self.data) == 0
    
# creating model to store the output video from yolo result
class OutputVideo(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    output_file = models.FileField(upload_to='output_videos/')


# Creating model to store every frame inference result using dictionary data structure {frame_number: <count list(list of counts in the n boxes)>}
class Inference(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    data = models.JSONField(default=dict)

    def add_inference(self, frame_number, count_list):
        self.data[frame_number] = count_list
        self.save()

    def get_frame_result(self, frame_number):
        try:
            return self.data[f'{frame_number}']
        except KeyError:
            return 'KeyError'
    
    def display(self):
        return self.data
    
    def size(self):
        return len(self.data)
    
    def get_inference(self):
        return self.data

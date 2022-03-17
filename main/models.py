import os
from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    doc = models.FileField(upload_to='documents')
    processed = models.BooleanField(default=False)
    user = models.ForeignKey(User, on_delete = models.CASCADE, default=1)

    def filename(self):
        return os.path.basename(self.doc.name)

class Processed(models.Model):
    fileid = models.IntegerField()
    file = models.FileField(upload_to='processed')
    detections_by_frame = models.TextField(default='')
    detections_by_frame_reduced = models.TextField(default='')
    time_reduced = models.TextField(default='')
    user = models.ForeignKey(User, on_delete = models.CASCADE, default=1)

    def filename(self):
        return os.path.basename(self.file.name)

class Logging(models.Model):
    login = models.CharField(max_length=30)
    action = models.CharField(max_length=255, default='')
    datetime = models.DateTimeField()

    def __str__(self):
        return f'Time: {self.datetime}, User: {self.login}'




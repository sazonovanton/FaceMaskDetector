from django.shortcuts import redirect, render
from .models import *
from .forms import DocumentForm
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.contrib.auth.decorators import user_passes_test

from .detector import run
from .data_processing import *
import json
import shutil
import os

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent


def main(request):
    context = {}
    return render(request, 'main/main.html', context)


def list(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newfile = Document(doc=request.FILES['doc'])
            newfile.user = request.user
            newfile.save()
            ext = f'File %s uploaded' % str(request.FILES['doc'])
        else:
            ext = 'Error while uploading file'
        dl = Logging.objects.create(login=str(request.user), action='[UPLOAD] ' + ext, datetime=timezone.now())
        dl.save()
        return redirect('main:list')
    else:
        form = DocumentForm()
    if request.user.is_superuser:
        documents = Document.objects.all().order_by('-pk')
    else:
        documents = Document.objects.all().filter(user=request.user).order_by('-pk')
    context = {'documents': documents, 'form': form, 'user': request.user}
    return render(request, 'main/list.html', context)


def process(request):
    if request.method != 'POST':
        raise HTTP404
    fileID = request.POST.get('doc', None)
    file_to_proc = get_object_or_404(Document, pk = fileID)
    if (request.user.is_superuser) or (request.user == file_to_proc.user):
        save_path, detections_by_frame, inf_speed, total_time, fps = run(str(Path(BASE_DIR / 'media' / 'documents' / str(file_to_proc.doc).split('/')[-1].split('\\')[-1])))
        procfile = Processed(fileid=fileID)
        procfile.file.name = 'processed/' + '.'.join(str(save_path).split('/')[-1].split('\\')[-1].split('.')[:-1]) + '.mp4' # relative to MEDIA_ROOT
        procfile.fileid=fileID
        procfile.detections_by_frame = json.dumps(detections_by_frame)
        dbfr, tr = reduce_output(detections_by_frame, fps)
        procfile.detections_by_frame_reduced = json.dumps(dbfr)
        procfile.time_reduced = json.dumps(tr)
        procfile.user = request.user
        procfile.save()
        Document.objects.filter(pk=fileID).update(processed=True)
        dl = Logging.objects.create(login=str(request.user), action=f'[PROCESS] %s; frames: %s; total speed: %s fps; inferense time: %s ms' % (str(file_to_proc.doc), str(len(detections_by_frame)), str(round(len(detections_by_frame)/total_time)), str(round(inf_speed, 1))), datetime=timezone.now())
        dl.save()
    else:
        dl = Logging.objects.create(login=str(request.user), action=f'[INVALID PROCESS] %s;' % str(file_to_proc.doc), datetime=timezone.now())
        dl.save()
    return HttpResponseRedirect(reverse('main:list'))


def view(request):
    if request.method != 'POST':
        raise HTTP404
    FileID = request.POST.get('doc', None)
    procfile = get_object_or_404(Processed, fileid = FileID)

    if (request.user.is_superuser) or (request.user == procfile.user):
        labels = json.loads(procfile.time_reduced)
        values = json.loads(procfile.detections_by_frame_reduced)
        events = {
                'labels': labels,
                'values': values,
            }
    else:        
        dl = Logging.objects.create(login=str(request.user), action=f'[INVALID ACCESS] %s' % str(procfile.file), datetime=timezone.now())
        dl.save()
        procfile = ''
        events={}
    context = {'procfile': procfile, 'user': request.user, 'events': events}
    return render(request, 'main/view.html', context)


def delete(request):
    if request.method != 'POST':
        raise HTTP404
    FileID = request.POST.get('doc', None)
    FileToDel = get_object_or_404(Document, pk = FileID)
    if (request.user.is_superuser) or (request.user == FileToDel.user):
        FileToDel.doc.delete()
        FileToDel.delete()
        ext = ' Deleted uploaded file'

        try:
            procfile = get_object_or_404(Processed, fileid = FileID)
            fileurl = procfile.file.name
            procfile.file.delete()
            procfile.delete()

            ext += '; Deleted processed file'
        except Exception as e:
            ext += '; Error while deleteing processed file.'

        try:
            path = Path(BASE_DIR / 'media' / 'processed' / Path('.'.join(str(os.path.basename(fileurl)).split('.')[:-1]) + '_crops'))
            shutil.rmtree(path, ignore_errors=True)
            ext += '; Deleted cropped images'
        except Exception as e:
            ext += '; Error while deleteing cropped images.'
            print(e)

        dl = Logging.objects.create(login=str(request.user), action=f'[DELETE] %s; ' % str(FileToDel.doc) + ext, datetime=timezone.now())
        dl.save()
    else:
        dl = Logging.objects.create(login=str(request.user), action=f'[INVALID DELETE] %s' % str(FileToDel.doc), datetime=timezone.now())
        dl.save()
    return HttpResponseRedirect(reverse('main:list'))


@user_passes_test(lambda u: u.is_superuser)
def logs(request):
    logs = Logging.objects.all().order_by('-pk')
    context = {'logs': logs}
    return render(request, 'main/logs.html', context)


def handler404(request, exception):
    context = {'status': 404, 'exception': exception}
    return render(request, 'error/404.html', context)
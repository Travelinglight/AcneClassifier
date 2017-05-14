# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.cache import never_cache
from predictor import *
import base64
import hashlib
import os
import json

predictor = Predictor(
    model_file = settings.BASE_DIR + '/acne/keras_model/model.json',
    weights_file = settings.BASE_DIR + '/acne/keras_model/weights.h5'
)

@never_cache
def index(request):
    return render(request, 'acne/index.html')


def predict(request):
    if request.method == "POST":
        filedata = base64.b64decode(request.POST['file'])
        filehash = hashfile(filedata)

        # create the folder if it doesn't exist.
        filepath = settings.BASE_DIR + '/acne/static/uploads/' + filehash[:2]
        try:
            os.mkdir(filepath)
        except:
            pass

        filepath = filepath + '/' + filehash[2:4]
        try:
            os.mkdir(filepath)
        except:
            pass

        # write file to disk
        filename = filepath + '/' + filehash[4:]
        fout = open(filename, 'wb')
        try:
            fout.write(filedata)
            fout.close()
            try:
                return JsonResponse({'status': 0, 'result': predictor.predict(filename)})
            except:
                return JsonResponse({'status': 2, 'message': 'predictor failed'})
        except:
            return JsonResponse({'status': 1, 'message': 'image saving failed'})


# Helpers
def hashfile(f):
    sha1 = hashlib.sha1()
    sha1.update(f)
    return sha1.hexdigest()

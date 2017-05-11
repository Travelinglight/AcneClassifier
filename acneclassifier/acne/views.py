# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.cache import never_cache

@never_cache
def index(request):
    return render(request, 'acne/index.html')

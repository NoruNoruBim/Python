from django.shortcuts import render
from django.http import *
from .forms import UserForm, AdminForm


def index(request):
    if request.method == "POST":
        name = request.POST.get("name")
        pswd = request.POST.get("pswd")
        text = request.POST.get("tmp")
        if pswd == "1234":
            with open(text, 'r', encoding='utf8') as file:
                text = file.read()
            return HttpResponse("<h5>{0}</h5>".format(text))
        else:
            return HttpResponse("<h2>Bad password</h2>")
    else:
        userform = UserForm()
        return render(request, "index.html", {"form": userform})

def admin(request):
    if request.method == "POST":
        url = request.POST.get("url")
        pswd = request.POST.get("pswd")
        if pswd == "4321":
            return HttpResponse("<h2>Here will be next part of project soon</h2>")
        else:
            return HttpResponse("<h2>Bad password</h2>")
    else:
        adminform = AdminForm()
        return render(request, "index.html", {"form": adminform})

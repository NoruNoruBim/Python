from django import forms


class UserForm(forms.Form):
    name = forms.CharField(label="Имя")
    pswd = forms.CharField(label="Пароль", widget=forms.PasswordInput)
    tmp = forms.FilePathField(path="C:/hello_world/Python/try_django/first_try/hello/static/text/", label="Файл", help_text="выберите файл")

class AdminForm(forms.Form):
    pswd = forms.CharField(label="Пароль", widget=forms.PasswordInput)
    url = forms.CharField(help_text="enter link to parse and save")

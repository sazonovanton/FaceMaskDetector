from django import forms
# from .models import Datetimelog

class DocumentForm(forms.Form):
    doc = forms.FileField(
        label='Выберите файл',
    )

    # def __init__(self, *args, **kwargs):
    #     super(DocumentForm, self).__init__(*args, **kwargs)
    #     self.fields['doc'].widget.attrs.update({'class': 'button'})


# class DatetimelogForm(forms.Form):
#     class Meta:
#         model = Datetimelog
#         fields = ['datetime', 'login', 'etc']
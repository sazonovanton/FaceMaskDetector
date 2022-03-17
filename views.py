from django.shortcuts import render


def handler404(request, exception):
    context = {
        'status_code': 404
        }
    return render(request, '404.html', context)

# def handler500(request):
#     context = {
#         'status_code': 500
#         }
#     return render(request, '500.html', context)
# Foolbox_Wavmark
I am researching the results of applying Foolbox to WavMark using competitive attacks based on numerical perturbations, such as FGSM, PGD, and DeepFool.

1️⃣ Скачать WinDivert

Скачайте последнюю версию с официального GitHub:
https://github.com/basil00/WinDivert/releases

Скачайте архив, например: WinDivert-2.x.x-A.zip

После распаковки структура будет выглядеть примерно так:

<img width="287" height="224" alt="image" src="https://github.com/user-attachments/assets/04b55032-f247-4a7e-8f8d-49411a5fe132" />

2️⃣ Добавить файлы в проект (Visual Studio)

Предположим, что вы используете x64 проект.

Скопируйте следующие файлы в папку проекта:

<img width="261" height="121" alt="image" src="https://github.com/user-attachments/assets/5e83a0ba-6278-43fc-b788-b1da37cd6fc2" />


Файлы необходимо взять из следующих папок архива WinDivert:

include/windivert.h

x64/WinDivert.lib

x64/WinDivert.dll

x64/WinDivert64.sys

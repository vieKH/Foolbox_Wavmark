# Foolbox_Wavmark
I am researching the results of applying Foolbox to WavMark using competitive attacks based on numerical perturbations, such as FGSM, PGD, and DeepFool.

1️⃣ Скачать WinDivert

Скачайте последнюю версию с официального GitHub:
https://github.com/basil00/WinDivert/releases

Скачайте архив, например:

WinDivert-2.x.x-A.zip

После распаковки структура будет выглядеть примерно так:

WinDivert/
 ├─ include/
 │   └─ windivert.h
 ├─ x64/
 │   ├─ WinDivert.lib
 │   ├─ WinDivert.dll
 │   └─ WinDivert64.sys
 ├─ x86/
 │   ├─ WinDivert.lib
 │   ├─ WinDivert.dll
 │   └─ WinDivert32.sys
2️⃣ Добавить файлы в проект (Visual Studio)

Предположим, что вы используете x64 проект.

Скопируйте следующие файлы в папку проекта:

project/
 ├─ main.cpp
 ├─ windivert.h
 ├─ WinDivert.lib
 ├─ WinDivert.dll
 └─ WinDivert64.sys

Файлы необходимо взять из следующих папок архива WinDivert:

include/windivert.h
x64/WinDivert.lib
x64/WinDivert.dll
x64/WinDivert64.sys

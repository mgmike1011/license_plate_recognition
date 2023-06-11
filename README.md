# License plate recognition
## Systemy wizyjne - laboratorium - projekt - 2023 - Miłosz Gajewski
# Struktura działania algorytmu
Algorytm został podzielony na trzy części realizujące kolejne etapy przetwarzania obrazu.
## Wyznaczenie(znalezienie) tablicy rejestracyjnej
1. Skalowanie zdjęcia do rozdzielczości 600px x 400px,
2. Konwersja zdjęcia do skali szarości,
3. Rozmywanie z wykorzystaniem filtru Gausa,
4. Zastosowanie algorytmu Canny,
5. Dylacja,
6. Znajdowanie konturów na zdjęciu,
7. Dla znalezionych konturów:
   1. Przybliżanie krzywych dla wieloboków - cv2.approxPolyDP()
   2. Wyznaczanie prostokąta dla przybliżenia - cv2.boundingRect()
   3. Wyznaczenie współczynnika proporcji dla konturu,
   4. Jeśli kontur ma cztery boki, a jego współczynnik proporcji jest powyżej 4.5 tablica została znaleziona,
8. Jeśli kontur tablicy nie został znaleziony powraca się do kroku drugiego i stosowana jest metoda:
   1. Wyliczenie filtru Sobela na zdjęciu,
   2. Zastosowanie progowania,
   3. Znajdowanie konturów na zdjęciu,
   4. Dla znalezionych konturów:
      1. Przybliżanie krzywych dla wieloboków - cv2.approxPolyDP()
      2. Jeśli kontur ma cztery boki, to tablica została znaleziona.
9. Sortowanie wierzchołków znalezionego konturu według współrzędnej x,
10. Rektyfikacja - prostowanie konturu tablicy(zdjęcie w skali szarości).
## Wyznaczenie poszczególnych liter w tablicy rejestracyjnej na uzyskanym obrazie z poprzedniego kroku
1. Zastosowanie progowania,
2. Operacja otwarcia,
3. Znajdowanie konturów na zdjęciu,
4. Dla znalezionych konturów:
   1. Wyliczenie pola powierzchni poszczególnych konturów,
   2. Jeżeli pole powierzchni jest w zadanym przedziale litera została znaleziona i wyznaczany jest dla niej prostokąt obudowujący,
5. Sortowanie znalezionych konturów według współrzędnej x (ustawienie liter w kolejności od lewej do prawej),
6. Sprawdzenie, czy wykryte kontury nie są wewnątrz siebie (przykład: dla litery O znaleziony został kontur zewnętrzny jak i wewnętrzny, który jest zbędny i można go usunąć),
7. Wycięcie poszczególnych liter.
## Klasyfikacja znalezionych liter
1. Wykonanie operacji progowania,
2. Skanowanie znalezionej litery do rozdzielczości 128px x 151px
3. Wykorzystanie dedykowanego narzędzia z biblioteki scikit-image - [structural_similarity](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_simi).
4. Klasyfikacja litery - [zbiór wzorców liter](processing/MIX),
5. Weryfikacja i walidacja znalezionego ciągu znaków
6. Zwrócenie napisu.

# Wywołanie programu
```console
python3 Gajewski_Milosz.py /path/to/directory/train /path/to/output.txt
```
# Inne testowane podejścia i metody
Podczas tworzenia programu przetestowane zostały różne sposoby wyznaczania konturu tablicy na zdjęciu, jak również klasyfikacji znaków. Miały one jednak niższą skuteczność działania niż prezentowany finalny algorytm.
## Inne testowane metody wyznaczania tablicy na zdjęciu:
- Rozmywanie -> Canny -> Znajdowanie konturów -> Sortowanie konturów względem area -> Długość łuku -> Aproksymacja do prostokąta -> Maska,
- Progowanie -> Dylacja -> Znajdowanie konturów -> Długość łuku -> Aproksymacja do prostokąta -> Sortowanie kształtów,
- Adaptacyjne progowanie -> Dylatacja -> Znajdowanie konturów -> Długość łuku -> Aproksymacja do prostokąta -> Sortowanie kształtów,
- Stworzenie dwóch rozmytych obrazów GaussianBlur o różnych parametrach -> Odjęcie stworzonych obrazów od siebie -> Canny -> Znajdowanie konturów -> Długość łuku -> Aproksymacja do prostokąta -> Sortowanie kształtów,
- Stworzenie dwóch rozmytych obrazów GaussianBlur o różnych parametrach -> Odjęcie stworzonych obrazów od siebie -> Erozja -> Otwarcie ->Canny -> Znajdowanie konturów -> Długość łuku -> Aproksymacja do prostokąta -> Sortowanie kształtów,
- Rozmywanie -> Canny -> Dylacja -> Znajdowanie konturów -> Długość łuku -> Aproksymacja do prostokąta -> Sortowanie kształtów.
## Inne testowane metody klasyfikacji znaków
### Szablon znaków - cv.matchTemplate()
![Szablon liter](processing/Template_5.png "Szablon znakow")

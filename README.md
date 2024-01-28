# Hand Gesture Tracking and Mouse Control

## Opis projektu

Projekt składa się z dwóch głównych skryptów: `main.py` i `mouse.py`. Skrypt `main.py` wykorzystuje bibliotekę MediaPipe do śledzenia gestów ręki w czasie rzeczywistym, natomiast `mouse.py` umożliwia kontrolę kursora myszy na podstawie pozycji środka dłoni wykrytej przez `main.py`.

### Wymagane biblioteki

- `mediapipe`
- `numpy`
- `opencv-python (cv2)`
- `pyautogui`

## Jak uruchomić

Upewnij się, że wszystkie wymagane biblioteki są zainstalowane.
Uruchom skrypt `main.py` w celu rozpoczęcia śledzenia gestów ręki.
W nowym terminalu uruchom `mouse.py`, aby połączyć się z serwerem utworzonym przez `main.py` i zacząć kontrolować kursor myszy za pomocą ruchów ręki.

## Zmienne do edycji

### `main.py`

- `MAX_NUM_HANDS`: Maksymalna liczba dłoni do wykrycia (domyślnie 4).
- `DETECTION_CONFIDENCE`: Prog pewności detekcji dłoni (domyślnie 0.4).
- `TRACKING_CONFIDENCE`: Prog pewności śledzenia dłoni (domyślnie 0.5).
- `DRAW_OVERLAY_PALM`: Włącza/wyłącza rysowanie nakładki na środku dłoni (domyślnie True).
- `DRAW_OVERLAY_FINGERS`: Włącza/wyłącza rysowanie nakładek na palcach (domyślnie False).
- `DRAW_LANDMARKS`: Włącza/wyłącza rysowanie znaczników na dłoni (domyślnie False).
- `PINCH_REQUIRE_LAST`: Liczba ostatnich próbek do rozważenia przy wykrywaniu szczypania (domyślnie 4).

### `mouse.py`

- `addr`: Adres IP serwera do połączenia (domyślnie 'localhost:9111').
- `width`, `height`: Rozdzielczość ekranu, na którym ma być kontrolowany kursor. Domyślnie ustawiane na rozdzielczość głównego ekranu.

## Uwagi

Aby zmienić zachowanie śledzenia gestów ręki lub kontrolowania kursora myszy, odpowiednie zmienne można dostosować w kodzie źródłowym.
Należy upewnić się, że skrypt `mouse.py` jest uruchomiony na tym samym urządzeniu co `main.py` lub w sieci, która pozwala na komunikację między urządzeniami.


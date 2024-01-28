# Hand Gesture Tracking and Mouse Control

## Opis projektu

Projekt skÅ‚ada siÄ™ z dwÃ³ch gÅ‚Ã³wnych skryptÃ³w: main.py i mouse.py. Skrypt main.py wykorzystuje bibliotekÄ™ MediaPipe do Å›ledzenia gestÃ³w rÄ™ki w czasie rzeczywistym, natomiast mouse.py umoÅ¼liwia kontrolÄ™ kursora myszy na podstawie pozycji Å›rodka dÅ‚oni wykrytej przez main.py.

### Wymagane biblioteki

- mediapipe
- numpy
- opencv-python (cv2)
- pyautogui
- socket
- threading
- queue
- json

## Jak uruchomiÄ‡

Upewnij siÄ™, Å¼e wszystkie wymagane biblioteki sÄ… zainstalowane.
Uruchom skrypt main.py w celu rozpoczÄ™cia Å›ledzenia gestÃ³w rÄ™ki.
W nowym terminalu uruchom mouse.py, aby poÅ‚Ä…czyÄ‡ siÄ™ z serwerem utworzonym przez main.py i zaczÄ…Ä‡ kontrolowaÄ‡ kursor myszy za pomocÄ… ruchÃ³w rÄ™ki.

## Zmienne do edycji

### main.py

- MAX_NUM_HANDS: Maksymalna liczba dÅ‚oni do wykrycia (domyÅ›lnie 4).
- DETECTION_CONFIDENCE: PrÃ³g pewnoÅ›ci detekcji dÅ‚oni (domyÅ›lnie 0.4).
- TRACKING_CONFIDENCE: PrÃ³g pewnoÅ›ci Å›ledzenia dÅ‚oni (domyÅ›lnie 0.5).
- DRAW_OVERLAY_PALM: WÅ‚Ä…cza/wyÅ‚Ä…cza rysowanie nakÅ‚adki na Å›rodku dÅ‚oni (domyÅ›lnie True).
- DRAW_OVERLAY_FINGERS: WÅ‚Ä…cza/wyÅ‚Ä…cza rysowanie nakÅ‚adek na palcach (domyÅ›lnie False).
- DRAW_LANDMARKS: WÅ‚Ä…cza/wyÅ‚Ä…cza rysowanie znacznikÃ³w na dÅ‚oni (domyÅ›lnie False).
- PINCH_REQUIRE_LAST: Liczba ostatnich prÃ³bek do rozwaÅ¼enia przy wykrywaniu szczypania (domyÅ›lnie 4).

### mouse.py

- addr: Adres IP serwera do poÅ‚Ä…czenia (domyÅ›lnie 'localhost:9111').
- width, height: RozdzielczoÅ›Ä‡ ekranu, na ktÃ³rym ma byÄ‡ kontrolowany kursor. DomyÅ›lnie ustawiane na rozdzielczoÅ›Ä‡ gÅ‚Ã³wnego ekranu.

## Uwagi

Aby zmieniÄ‡ zachowanie Å›ledzenia gestÃ³w rÄ™ki lub kontrolowania kursora myszy, odpowiednie zmienne moÅ¼na dostosowaÄ‡ w kodzie ÅºrÃ³dÅ‚owym.
NaleÅ¼y upewniÄ‡ siÄ™, Å¼e skrypt mouse.py jest uruchomiony na tym samym urzÄ…dzeniu co main.py lub w sieci, ktÃ³ra pozwala na komunikacjÄ™ miÄ™dzy urzÄ…dzeniami.

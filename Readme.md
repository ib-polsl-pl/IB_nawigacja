# Wtyczka do programu 3D Slicer


## Przygotowanie środowiska
- Slicer stable release (4.11.20210226)
- Zainstalować rozszerzenie `DeveloperToolsForExtensions`, `DebuggingTools`, `SlicerIGT`, `SlicerOpenIGTLink`
- Menu Edit -> Settings -> Developer -> Enable developer mode


### Utworzenie nowej wtyczki
- moduł Extension Wizard
- Create Extension
- Add Module to Extension

### Konfiguracja PyCharm

Przydatne materiały:
- <https://slicer.readthedocs.io/en/latest/developer_guide/debugging/overview.html>
- <https://github.com/SlicerRt/SlicerDebuggingTools>
- [Podłączenie do jupytera](https://github.com/SlicerRt/SlicerDebuggingTools)


Kolejne etapy działania:

1. Aktywacja zdalnego debuggera w Pycharm
    - zgodnie z informacjami z https://github.com/SlicerRt/SlicerDebuggingTools 
    - Run -> Edit configurations -> Add new configuration
    - Select Python debug server
    - Name: Slicer remote debugger, Port: 5678
2. Ustawienie interpretera
    - File -> Settings -> Project -> Python Interpreter -> Existing Virtual Environment 
    - Podać ścieżkę do zainstalowanego Slicera: `${SLICER_INSTALL}/Slicer-4.11.20210226-linux-amd64/bin/PythonSlicer`
3. Uruchomienie 
    - Pycharm -> Debug -> Slicer remote debugger (pojawia się w konsoli: `Waiting for process connection...`)
    - Slicer -> Python debugger -> Pycharm, ścieżka do pliku egg, port 5678, connect

Uwagi

1. Można debugować nawet sam początek (w `__init__` jest `connect("startupCompleted()", registerSampleData)` i już można zatrzymać break-point wewnątrz tej funkcji `registerSampleData`. Do tego trzeba:
    - Pycharm: zresetować debuger (pojawia się komunikat "Waiting for process connection")
    - Slicer: w module "Python Debugger" zaznczyć checkbox "Auto-connect on next app startup", potem przejść do modułu np. "Developer Tools For Extensions" i wybrać "Restart Slicer". 
    - W trakcie restartu aktywije się break point w Pycharm, a sam Slicer uruchamia tylko małe okienko "SLicer is paused until Pycharm accepts connection".
    - Po zwolnieniu break-pointa program się uruchamia bez przeszkód



### Architektura wtyczki

Następujące funkcje wywoływane są automatycznie w odpowiedzi na zdarzenia (zakładam że moduł nazywa się `ModelOffset`:

- W trakcie uruchamiania Slicera - `ModelOffset.__init__()`
- Natychmiast po zakończeniu uruchamiania Slicera - funkcje, które wewnątrz powyższej `__init__` zostały połączone z sygnałem: `slicer.app.connect("startupCompleted()", local_foo)`
- Przy pierwszym otwarciu okna modułu - `ModelOffsetWidget.__init__()` oraz `ModelOffsetWidget.setup()`, w tej drugej wywoływane też `ModelOffsetLogic.__init__()`

Ważniejsze momenty
- Tworzenie elementów UI na podstawie pliku XML Qt w funkcji `ModelOffsetWidget.setup()`


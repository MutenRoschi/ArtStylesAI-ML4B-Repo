# ArtStylesAI for ML4B

 ML4B – Bildtransformation mit KI (Bilder in andere Epochen verwandeln)

In diesem Projekt wollen wir mit Hilfe des CycleGAN Models eine Streamlit App entwickeln, die Bilder in den Stil verschiedener Kunstepochen überträgt.  

Beispiel: Ein modernes Foto soll im Stil des Impressionismus oder Barock dargestellt werden.

--> Link zur Streamlit App: https://artstylesai-ml4b-repo-zwxfdtk5cctppa2wbupqvd.streamlit.app/

Dokumentation wichtiger Schritte:

1) Datensatz wählen:

   Datensatz mit zahlreichen Kunstepochen:  https://www.kaggle.com/datasets/ansonnnnn/historic-art?resource=download-directory
   --> zur Referenz siehe: "display-european-art.ipynb" (originales Notebook des Autors auf kaggle)
 
2) Data Preparation:

   Um später das CycleGAN Model nutzen zu können, müssen
 
   a) zwei bis drei Epochen gefiltert werden

    b) Die Bilder auf ein einheitliches Format gebracht werden.

    --> siehe: "Art-Styles-AI.ipynb". (Notebook zur Data Preparation via JupyterLab) und "environment_full.yml" (Umgebung mit allen wichtigen requirements)
   
3) Model:
   
   Ein lokales Training via CycleGAN ist auf Grund von Kapazitätsmangel und geringer Rechnerleistung nicht möglich.
   Es wurde deshalb entschieden, die T4-GPU von Google Colab zu nutzen.
   Folgende Probleme traten dabei auf:
1. Durchlauf:
   Die Laufzeit wird bei Inaktivität automatisch getrennt, was das Training erschwert.

   --> Lösung: einen weiteren Code während des Trainings ausführen, der alle      10 Minuten eine Ausgabe erstellt und somit das Programm am Laufen hält.
2. Durchlauf:
   Das Model läuft nur wenige Stunden (ca. 3), bevor das Limit für die GPU Nutzung erreicht wird.

    --> Lösung: Upgrade auf das Colab Pro Abo
3. Durchlauf:
   Nachdem das Training nach jedem Durchlauf abgeschlossen war, fiel auf, dass die Bilder zwar durch das Model gelaufen sind, der Fortschritt aber (also die         generierten Bilder und Gewichtungen), wurde nicht gespeichert. Das Model fing also nach jedem Durchlauf von vorne an, alle Bilder durchzugehen.

    --> Lösung:         Erstellen von Checkpoints auf Google Drive, um bei einem neuen Durchlauf nicht von vorne anzufangen.

   Siehe: "CycleGAN_Training-and-Evaluation.ipynb" (Notebook zum Model-Training und Evaluation)

   Generierte Bilder und Gewichtungen sind hier zu finden: https://drive.google.com/drive/folders/1rTzT78v7ssT4RZu1H1Lm36b3FRThT19Q?usp=drive_link

4a) Qualitative Model Evaluation:

4b) Quantitative Model Evaluation:

5) Deployment

6) Fazit




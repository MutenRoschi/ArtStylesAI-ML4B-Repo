# ArtStylesAI for ML4B

 ML4B – Bildtransformation mit KI (Bilder in andere Epochen verwandeln)

In diesem Projekt wird versucht mit Hilfe des CycleGAN Models eine Streamlit App zu entwickeln, die Bilder in den Stil verschiedener Kunstepochen überträgt.  

In diesem Beispiel wurden spezifisch Bilder aus der Kunstrichtung Barock und Realismus verwendet. 

--> Link zur  "Baroque → Realism Style Transfer" Streamlit App: https://artstylesai-ml4b-repo-zwxfdtk5cctppa2wbupqvd.streamlit.app/

Dokumentation wichtiger Schritte:

# 1) Datensatz wählen:

   Datensatz mit zahlreichen Kunstepochen:  https://www.kaggle.com/datasets/ansonnnnn/historic-art?resource=download-directory
   
   --> zur Referenz siehe: "display-european-art.ipynb" (originales Notebook des Autors auf kaggle)
 
# 2) Data Preparation:

   Um später das CycleGAN Model nutzen zu können, müssen
 
   a) zwei bis drei Epochen gefiltert werden

   b) Die Bilder auf ein einheitliches Format gebracht werden.

   c) Die Bilder müssen in Trainingsgruppen unterteil werden: trainA = Barock und trainB = Realismus

    --> siehe: "Art-Styles-AI.ipynb". (Notebook zur Data Preparation via JupyterLab) und "environment_full.yml" (Umgebung mit allen wichtigen requirements)
   
# 3) Model:
   
   Ein lokales Training via CycleGAN ist auf Grund von Kapazitätsmangel und geringer Rechnerleistung nicht möglich.
   Es wurde deshalb entschieden, die T4-GPU von Google Colab zu nutzen.
   Folgende Probleme traten dabei auf:
   
1. Durchlauf:
   Die Laufzeit wird bei Inaktivität automatisch getrennt, was das Training erschwert.

   --> Lösung: einen weiteren Code während des Trainings ausführen, der alle      10 Minuten eine Ausgabe erstellt und somit das Programm am Laufen hält.
2. Durchlauf:
   Das Model läuft nur wenige Stunden (ca. 3), bevor das Limit für die GPU Nutzung erreicht wird.

    --> Lösung: Upgrade auf das Colab Pro Abo. Aktive Überwachung, war trotzdem manchmal notwendig aufgrund von Laufzeitunterbrechungen.
   
3. Durchlauf:
   Nachdem das Training nach jedem Durchlauf abgeschlossen war, fiel auf, dass die Bilder zwar durch das Model gelaufen sind, der Fortschritt aber (also die         generierten Bilder und Gewichtungen), wurde nicht gespeichert. Das Model fing also nach jedem Durchlauf von vorne an, alle Bilder durchzugehen.

   --> Lösung: Erstellen von Checkpoints auf Google Drive, um bei einem neuen Durchlauf nicht von vorne anzufangen.

*Ergebnisse des Models*

  Es wurden jeweils 200 Bilder pro Epoche (insgesamt also 400 Bilder) durch das Modell verarbeitet. Dabei wurde immer ein Bild aus trainA (Barock) mit einem Bild aus trainB (Realismus) abgeglichen, um die Gewichtungen des Modells zu berechnen. Diese Gewichtungen wurden in der Datei „latest_net_G“ unter den Checkpoints gespeichert. Zusätzlich wurden die Bilder in vier verschiedene Kategorien unterteilt: 1. real (originales Bild), 2. fake (generiertes Bild), 3. rec (rekonstruiertes Bild, das zurück in die Ausgangskategorie transformiert wurde), und 4. idt (Identitätsbild, das den Stil des Originalbilds beibehält).
   
   Siehe: "CycleGAN_Training-and-Evaluation.ipynb" (Notebook zum Model-Training und Evaluation)

   Generierte Bilder und Gewichtungen sind hier zu finden: https://drive.google.com/drive/folders/1rTzT78v7ssT4RZu1H1Lm36b3FRThT19Q?usp=drive_link

# 4a) Qualitative Model Evaluation:

Bei der qualitativen Bewertung wurden die generierten Bilder einfach mit den Originalbildern verglichen, um zu sehen, wie gut der Stiltransfer funktioniert. Es fiel auf, dass die Qualität der Ergebnisse noch nicht ganz überzeugend war. Ein Grund dafür war, dass oft Bilder aus unterschiedlichen Kategorien zusammentrainiert wurden – zum Beispiel Porträts und Landschaftsbilder. Das führte dazu, dass die Umwandlung eher oberflächliche Farbänderungen vornahm, statt tiefere stilistische Merkmale zu übertragen. 

Eine Lösung hierfür wäre es, das Model mit mehr Bildern und über einen längeren Zeitraum zu trainieren. Dies war allerdings aufgrund von begrenztem Speicherplatz und Zeit in diesem Projekt nicht möglich.

# 4b) Quantitative Model Evaluation:

Für die quantitative Evaluation des Modells wurden mehrere Kennzahlen berechnet:

1. *FID Wert (Fréchet Inception Distance)*: 105.96
Ein höherer FID-Wert deutet auf eine größere Diskrepanz zwischen den echten und generierten Bildern hin. Der Wert zeigt an, dass es noch Verbesserungsbedarf gibt, da ein niedrigerer FID-Wert ein besseres Modell bedeutet.

2. *Inception Score (IS)*: 6.66
Der Inception Score bewertet, wie gut das Modell Bilder generiert, die sowohl klar als auch vielfältig sind. Ein Wert von 6.66 ist ein akzeptables Ergebnis, wobei ein höherer Wert auf eine bessere Bildqualität und -vielfalt hinweist.

3. *PPL (Perplexity)*: 0.82
Der PPL-Wert misst die Unsicherheit des Modells bei der Vorhersage. Ein niedriger Wert (nahe 1) deutet auf ein gutes Modell hin, das klar zwischen den verschiedenen Stilen unterscheidet.

4. *Precision*: 0.35
Die Präzision gibt an, wie genau die generierten Bilder die richtigen Kategorien treffen. Ein Wert von 0.35 zeigt, dass das Modell noch eine hohe Rate von falsch-positiven Ergebnissen hat.

5. *Recall*: 0.34
Der Recall misst, wie gut das Modell in der Lage ist, alle relevanten (echten) Bilder korrekt zu erkennen. Ein Wert von 0.34 bedeutet, dass ein erheblicher Teil der echten Bilder nicht korrekt wiedergegeben wird.

# 5) Deployment

Die Streamlit-App wurde mit GitHub verbunden, indem das Repository mit dem Code und den notwendigen Dateien auf GitHub hochgeladen wurde.

siehe zur Referenz: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix (originale CycleGAN Repository)

Der Code für die App und die zugehörigen Modell- und Trainingsdateien wurden in das Repository integriert. Beim ersten Versuch traten einige Fehler auf, insbesondere beim Laden des Modells und der Architektur. Nach mehreren Anpassungen und Tests mussten bestimmte Teile des Modells sowie die Art und Weise, wie es auf Google Drive zugreift, um das CycleGAN-Modell korrekt zu laden, angepasst werden. Zudem wurden in der Architektur kleinere Änderungen vorgenommen, um sicherzustellen, dass das Modell in der Streamlit-App richtig funktioniert und die Bilder wie gewünscht transformiert werden. 

# 6) Fazit
Das Projekt bietet wertvolle Einblicke in die Anwendung von CycleGAN zur Bildtransformation und die Herausforderungen beim Training von Modellen. Trotz anfänglicher Schwierigkeiten konnten durch kontinuierliche Anpassungen und der Nutzung von Google Colab Fortschritte erzielt werden. Es besteht weiterhin Potenzial zur Verbesserung der Bildqualität und der Erweiterung des Modells, insbesondere durch die Integration weiterer Kunstepochen und eine bessere Feinabstimmung des Trainingsprozesses.

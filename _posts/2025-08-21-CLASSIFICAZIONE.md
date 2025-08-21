---
title: "Classificazione"
description: Una spiegazione delle metriche di classificazione
date: 2025-08-21 12:00:00 +0530
categories: [Classification]
tags: [Classification, ROC, AUC, Precision, Recall, F1-score]
comments: false
mermaid: true
---


### Classificazione


La classificazione è un problema di machine learning che consiste nel catalogare un'istanza in una delle classi previste.  
Un **classificatore binario** è un modello che prende in input un'istanza e restituisce una classe tra due possibili classi; uno **multiclasse**, invece, restituisce una classe tra più di due possibili classi.  
(Un modello di **regressione**, invece, prende in input un'istanza e restituisce un valore continuo.)  
Per semplificare, ci concentreremo sui classificatori binari.  

#### Matrice di confusione

Una matrice di confusione è una tabella che mostra il numero di istanze classificate in modo corretto o errato.

![Matrice di confusione](/assets/images/confusion_matrix.svg){: width="300" height="300"}

Quando, in un dataset, il totale dei positivi reali è molto diverso da quello dei negativi reali, la matrice di confusione è **sbilanciata**.  



#### Metriche di classificazione

I veri e falsi positivi e negativi vengono usati per calcolare diverse metriche di classificazione.  
Di solito le metriche vengono calcolate con una soglia (**threshold**) fissa, per esempio 0.5.  
Se il modello restituisce una probabilità maggiore di 0.5, la classe è positiva, altrimenti è negativa.  
Dopo aver calcolato le metriche, si può cambiare la soglia per vedere come cambiano le metriche, al fine di ottenere le migliori prestazioni possibili del modello.  

#### Accuracy
> **Quante istanze sono state classificate correttamente?**
{: .prompt-tip }

L'accuracy è la proporzione di istanze correttamente classificate rispetto al totale delle istanze.  
Un modello perfetto ha un'accuracy di 1, perché tutte le istanze sono classificate correttamente.  
Per i dataset fortemente sbilanciati, l'accuracy **non è una buona metrica** per valutare la qualità di un modello perché, per classi che compaiono raramente, l'accuracy non è rappresentativa del comportamento del modello.  


#### Recall
> **Se il riferimento sono le istanze positive, quante di queste sono state classificate correttamente?**
{: .prompt-tip }

Il recall (richiamo) è la proporzione di istanze positive correttamente classificate rispetto al totale delle istanze positive reali.  
Un modello perfetto ha un recall di 1, perché tutte le istanze positive sono classificate correttamente.  

#### Precision
> **Quando il modello ha previsto la classe positiva, qual è stata la percentuale di previsioni corrette?**
{: .prompt-tip }

La precisione è la proporzione di istanze positive correttamente classificate rispetto al totale delle istanze classificate positive.  
La precisione migliora quando diminuisce il numero di falsi positivi, al contrario del recall che migliora quando diminuisce il numero di falsi negativi.  
Un modello perfetto ha una precisione di 1, perché tutte le istanze positive sono classificate correttamente.  

#### F1-score
> **Per una soglia scelta, quanto bene il modello bilancia il fatto di aver avuto ragione nell'identificare i positivi (Precision) e la percentuale di positivi trovati (Recall)?**
{: .prompt-tip }
Il F1-score è la **media armonica** di precisione e recall.  
Un modello perfetto ha un F1-score di 1, perché precisione e recall sono entrambi 1, altrimenti il F1-score è compreso tra 0 e 1 ed è simile al valore peggiore tra precisione e recall. 
Questa metrica bilancia l'importanza di precisione e recall ed è preferibile alla precisione per i set di dati con classi sbilanciate perché è più robusta.  


#### ROC
> **Quanto il modello riesce a mantenere True Positive Rates (TPR) alti e False Positive Rates (FPR) bassi variando la soglia?**
{: .prompt-tip }

ROC (**Receiver Operating Characteristic**) è una curva che mostra la capacità del modello di distinguere tra classi positive e negative.  
Una curva ROC ideale è la seguente, in quanto il modello è in grado di distinguere perfettamente tra classi positive e negative.  
![Curva ROC ideale](/assets/images/roc_ideal.svg){: width="300" height="300"}


Una pessima curva ROC è la diagonale, in quanto il modello non è in grado di distinguere tra classi positive e negative (modello random).
![Curva ROC pessima](/assets/images/roc_bad.svg){: width="300" height="300"}



In una curva ROC reale, il modello non è in grado di distinguere perfettamente tra classi positive e negative. 
Il punto che minimizza la distanza dall'angolo in alto a sinistra è la soglia migliore per quel modello.
![Curva ROC reale](/assets/images/roc_real.svg){: width="300" height="300"}

**ROC AUC** (**Area Under the Curve**) è l'area sotto una curva ROC.  

Un modello perfetto ha un ROC AUC di 1, perché la curva ROC è una linea retta che va dall'angolo in basso a sinistra all'angolo in alto a destra, chiaramente l'AUC di una curva pessima (diagonale) è 0.5.  
Una qualsiasi curva reale ha un AUC compreso tra 0.5 e 1.  


#### PR
> **Quando il modello ha previsto la classe positiva, quante volte ha avuto ragione? (Precision) e quanti veri positivi ha trovato? (Recall)**
{: .prompt-tip }

PR (**Precision-Recall**) è una curva che mostra la qualità delle classi positive predette.  
Risponde ad una domanda simile a quella del F1 score, ma mentre l'F1 sore è un punto sulla curva PR, la PR valuta l'intero comportamento del modello per tutte le soglie.

La curva ideale è la seguente, in quanto il modello ha avuto sempre ragione nell'identificare i positivi e li ha trovati tutti.  
![Curva PR ideale](/assets/images/pr_ideal.svg){: width="300" height="300"}

Nella realtà, più la curva si avvicina all'angolo in alto a destra (1.0,1.0), migliore è il modello.  
In alternativa, si può pensare che il massimo score F1 corrisponde alla migliore curva PR per il modello.
**PR AUC** (**Area Under the Curve**) è l'area sotto una curva Precision-Recall.  
Un modello perfetto ha un PR AUC di 1.











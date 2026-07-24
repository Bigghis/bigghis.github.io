---
title: "La teoria del Feature Engineering"
description: "Il Feature Engineering è cruciale per ottenere un modello di machine learning che funzioni in modo ottimale."
date: 2026-07-23 12:00:00 +0530
categories: [Machine Learning, Feature Engineering]
tags: [Machine Learning, Feature Engineering]
comments: false
protected: false
---

Il **feature engineering** è il processo di conversione dei dati in features utili per un modello di machine learning.  
Una **feature** è una singola caratteristica misurabile del dato che viene utilizzata per prevedere il risultato di un modello di machine learning.  
Tenendo presente che la qualità dei dati influisce direttamente sulle prestazioni del modello, il feature engineering è un processo fondamentale per ottenere un modello che funzioni in modo ottimale.  

> "Applied machine learning is basically feature engineering". Andrew Ng
{: .prompt-info }

### Curse of dimensionality

Se ipotizziamo che ogni feature dei dati possa essere rappresentata in una dimensione, come un vettore, è chiaro che più features ci sono, più crescono le dimensioni e di conseguenza la complessità di rappresentazione dei dati.   
Sta a noi decidere quali features sono importanti e quali secondarie che possono essere scartate, in quanto non contribuiscono al processo decisionale del modello e aumentano inutilmente la complessità di rappresentazione dei dati.  
Difatti un grande numero di features contribuisce alla creazione di un **curse of dimensionality**, ovvero un aumento dello spazio dimensionale in cui sono collocati i dati, che può portare a problemi di overfitting e di difficoltà nell'addestrare il modello.  
Per tenere sotto controllo questo problema si sceglie ponderatamente quali sono le features importanti.  
Esistono tecniche come **PCA (Principal Component Analysis)** che permettono di distillare le features in un minor numero in modo da continuare a mantenere la maggior informazione possibile.  
Altra tecnica è il **K-means clustering** che permette di raggruppare le features in base alle loro similarità.  
Queste tecniche sono algoritmi classici non algoritmi di machine learning.    


### Imputazione dei dati mancanti

Uno dei problemi più comuni nei dataset è la mancanza di alcuni dati.  
Ciò è deleterio per l'addestramento dei modelli e per questo motivo si cerca di imputare i dati mancanti con altri valori.  
Una comunissima tecnica è quella di sostituire il dato mancante con la **media aritmetica** dei valori presenti nella stessa feature (colonna).  
Ha il vantaggio di non alterare la media della distribuzione dei valori della featurema non sempre è la soluzione migliore.  
All'interno di una feature, infatti, possono esserci valori che si discostano tantissimo dalla media. Sono chiamati, per questo, **outliers**, o valori anomali.  
In presenza di outliers è preferibile utilizzare la **mediana** per sostituire il dato mancante in quanto è meno sensibile a tali valori anomali.  
Tuttavia le sostituzioni con medie e mediane sono spesso semplicistiche e non sempre sono le soluzioni migliori.  
Non coprono situazioni in cui una feature è correlata ad altre feature, per esempio.  
Inoltre non è possibile usare medie e mediane per **feature categoriche**, es.: colore dei capelli, genere, etc.  

Se si hanno grandi dataset, può anche essere utile eliminare le righe che contengono dati mancanti, a patto che queste non alterino in alcun modo la distribuzione dei dati. Per esempio una riga da eliminare può contenere relazioni tra feature che possono essere utili per il modello, oltre al dato mancante.

Si può anche usare il machine learning per imputare i dati mancanti, esempio tramite algoritmo di ML **KNN (K-Nearest Neighbors)**. L'idea è quella di prendere una riga che ha un dato mancante e cercare le k righe più vicine (in base a una **distanza euclidea**) e usare la media dei dati di queste righe per sostituirla al dato mancante.  
Anche questo sistema, tuttavia, è più adatto per dati numerici e non per dati categorici.  
Infatti è meglio usare reti neurali per feature categoriche, che possono imparare a sostituire i dati mancanti in base alle feature correlate.  

Esiste una tecnica molto avanzata per imputare i dati mancanti, chiamata **MICE (Multiple Imputation by Chained Equations)**.  
Ad oggi è la tecnica migliore, ma è anche la più complessa da implementare.   

In generale, quando siamo in presenza di molti dati mancanti all'interno di un dataset, dovremmo sempre chiederci se fosse possibile **allargare il dataset con nuovi dati** che possano aiutare a mitigare il problema.  


### Gestione dei dati sbilanciati

Nei dataset che contengono feature categoriche, è possibile che ci siano classi che sono rappresentate in modo sbilanciato.
In questi casi, prendendo ad esempio un sistema di classificazione binaria, si hanno notevoli discrepanze tra i casi positivi ed i casi negativi
   
> Esempio: nel caso di rilevamento delle frodi, il 99% delle transazioni documentate nel dataset sono legittime e solo l'1% sono frodi.
> Ciò significa che l'alto numero di casi negativi influenza il modello a predire sempre il caso negativo, non individuando mai i casi positivi, cioè le frodi.
{: .prompt-info }

Per cercare di mitigare il problema si possono usare varie tecniche:


##### Oversampling 

Per cercare di riequilibrare lo sbilanciamento si può usare la tecnica dell'oversampling. Questa tecnica consiste nel duplicare casi positivi (frodi) per bilanciare il numero di casi negativi (transazioni legittime).  
Si è sperimentato che aggiungere casi positivi minoritari all'interno del dataset funziona bene per addestrare le reti neurali.


##### Undersampling

Un'altra tecnica è l'undersampling, che consiste nell'eliminare casi negativi (transazioni legittime) per bilanciare il numero di casi positivi (frodi).
Tuttavia, di solito, eliminare dati non è una buona soluzione, in quanto si perdono informazioni potenzialmente utili. 
In sostanza, diciamo che è fattibile se il dataset è davvero molto grande. L'arte del feature engineering sta anche in questo: trovare il giusto equilibrio tra informazioni utili e dati da eliminare.


##### SMOTE generazione di dati sintetici

**SMOTE (Synthetic Minority Over-sampling Technique)** è una tecnica che genera nuovi casi positivi (frodi) usando gli algoritmi KNN (K-Nearest Neighbors) che abbiamo visto prima. Calcolando la distanza euclidea dai casi positivi, vengono generati nuovi casi positivi che sono simili ai casi positivi originali perché hanno le stesse caratteristiche, ma non sono copie esatte.  
Ovviamente la tecnica puà essere usata anche per creare casi negativi simili a quelli originali.
Questo aiuta molto durante l'addestramento del modello. 


##### Modifica del threshold di classificazione

Prendiamo per semplicità un sistema di classificazione binaria.  Il modello predice caso positivo e caso negativo sottoforma di una certa percentuale di probabilità.  
Questa probabilità viene confrontata con un **threshold** (valore soglia) al di sotto del quale il modello predice caso negativo e al di sopra del quale predice caso positivo.  
Nel feature engineering, il threshold può essere modificato per bilanciare il numero di casi positivi e negativi, per cercare di migliorare le predizioni dei casi positivi. Ovviamente il settaggio del giusto threshold è una scelta che deve essere fatta con molta attenzione e deve essere testata con cura, perché può anche sensibilmente peggiorare le prestazioni del modello.


### Gestione dei dati anomali (outliers)

Per comprendere le questioni relative ai dati anomali, è utile far riferimento a feature numeriche.  
Abbiamo visto che nei dataset possono esserci valori anomali, che si discostano tantissimo dalla **media** di tutto l'insieme dei valori di cui fanno parte.  
E' anche molto semplice individuare dati anomali in una distribuzione di probabilità visualizzandola in un istogramma.  
Misurare di quanto i valori si discostano dalla media vuol dire calcolarne la **varianza** o la **deviazione standard**.  
Difatti **la varianza indica la dispersione dei valori rispetto alla media**.  
Essa utilizza valori al quadrato, per questo motivo non è sempre molto intuitiva da leggere  (ad esempio, se i dati sono in Euro, la varianza sarà in Euro al quadrato).   
Per questo motivo si usa più spesso la **deviazione standard**, che è semplicemente la radice quadrata della varianza. Riportando il numero all'unità di misura originale (di nuovo in Euro), ci fa capire subito di quanto i dati si allontanano dalla media.

In una **distribuzione normale** (la classica "campana"), la maggior parte dei valori si concentra intorno alla media: circa il 68% cade entro **±1σ** e circa il 95% entro **±2σ**. I valori che restano fuori da queste fasce sono i candidati naturali a essere considerati anomali.

Esempio di distribuzione del reddito annuo:

![Distribuzione normale con media e fasce ±1σ / ±2σ](/assets/images/normal_distribution.svg){: width="720"}

> **Un esempio pratico:**
> Immaginiamo di avere questi 6 valori in un dataset: `18, 20, 22, 19, 21, 200`.
> - La **media** di questi numeri è `50`.
> - La **deviazione standard** (calcolata matematicamente) è circa `67`.
> 
> Come troviamo l'anomalia? Una regola comune è considerare "normali" i valori che rientrano in un intervallo di una o due deviazioni standard dalla media. 
> Se prendiamo la media (50) e aggiungiamo/sottraiamo una deviazione standard (67), otteniamo un intervallo "normale" che va da `-17` a `117`. 
> Il valore `200` è enormemente al di fuori di questo intervallo, rendendolo un **dato anomalo (outlier)** impossibile da non notare.
{: .prompt-tip }

Anche i dati anomali possono essere gestiti in vari modi, è possibile **eliminarli**, **sostituirli** con la media o la mediana dei dati presenti nella stessa feature, o con un valore predefinito, o **lasciarli** così com'è...
Tutto sta nel saper individuare la giusta soluzione in base alla natura del dato anomalo e alla natura del dataset e non è sempre semplice!

> **Quando conviene eliminare un dato anomalo?**
> Immaginiamo un dataset con i tempi di consegna (in giorni) di un corriere: `2, 3, 2, 4, 3, 365`.
> Il valore `365` è sospetto: un pacco consegnato dopo un anno intero. Indagando, scopriamo che quel record appartiene a un **ordine di test** creato dagli sviluppatori in ambiente di produzione, mai cancellato, e lasciato "aperto" per mesi.
> Non rappresenta quindi una consegna reale: è **rumore operativo**. In questo caso è bene **eliminarlo**, perché altrimenti falserebbe le metriche del servizio (es. alzando artificialmente il tempo medio di consegna).
> 
> Diverso è invece il caso di un tempo di consegna di `15` giorni dovuto a uno sciopero dei trasporti o a un pacco spedito all'estero: lì l'outlier è un evento **reale**, e va tenuto (o gestito a parte) perché racconta qualcosa di vero sul business.
{: .prompt-tip }

All'interno dell'ecosistema AWS esistono algoritmi come **AWS Random Cut Forest** che possono aiutare a identificare i dati anomali.  
Possiamo quindi usare Random Cut Forest all'interno di **Kinesis Analytics** o dentro **SageMaker**, per esempio.

##### Esempio: distribuzione dei redditi

Supponiamo di voler modellare la distribuzione dei redditi annuali di una popolazione: stiamo semplicemente osservando quanto guadagna ogni persona all'anno e cosa questo significa per l'insieme dei dati nel suo complesso.  
Immaginiamo una distribuzione di redditi "normali" centrata intorno ai **27.000 €** all'anno. Ora aggiungiamo un singolo **miliardario** e tracciamo l'istogramma.  
Quel miliardario manda a monte la scala della distribuzione: tutte le persone comuni finiscono ammassate in una piccola fascia a sinistra, mentre l'outlier (quel dato estremo che a volte a occhio non si nota nemmeno sulla scala) distorce enormemente la lettura dei dati.

![Istogramma dei redditi distorto da un miliardario](/assets/images/income_with_outlier.svg){: width="720"}

In questo scenario è la **media** a diventare poco significativa: un solo reddito altissimo la trascina verso valori bizzarri, lontani dai 27.000 € da cui eravamo partiti. La **mediana**, invece, resta vicina a quel valore centrale, perché è meno sensibile agli estremi.  

Calcolando media e deviazione standard del dataset riusciamo agevolmente a identificare gli outlier per poi **scartare** i valori che si trovano al di fuori, ad esempio, di **due deviazioni standard**. Applicando questa regola e riprovando a tracciare l'istogramma, otteniamo una distribuzione molto più leggibile: scartando il miliardario, anche la media torna più vicina ai 27.000 € di partenza.

![Istogramma dei redditi dopo aver scartato l'outlier](/assets/images/income_without_outlier.svg){: width="720"}  

Ma qui arriva la domanda davvero importante: **è opportuno sbarazzarsi di quel miliardario?** Dipende dall'obiettivo di business. Se stiamo modellando il reddito del "lavoratore tipico", rimuoverlo ha senso. Se invece stiamo studiando disuguaglianza, fiscalità o segmenti VIP, quel dato è reale e prezioso: eliminarlo altererebbe proprio il fenomeno che ci interessa.  
Algoritmi come Random Cut Forest aiutano a **individuare** le anomalie; la decisione di scartarle, sostituirle o tenerle resta sempre una scelta di dominio.


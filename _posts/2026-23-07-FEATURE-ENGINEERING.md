---
title: "La teoria del Feature Engineering"
description: "Senza un'attenta e corretta preparazione dei dati, anche il miglior modello fatica a fornire risultati ottimali."
date: 2026-07-23 12:00:00 +0530
categories: [Machine Learning, Feature Engineering]
tags: [Machine Learning, Feature Engineering]
comments: false
protected: false
---

Il **feature engineering** è il processo che trasforma i dati grezzi in informazioni più chiare e utili per addestrare correttamente un modello di machine learning.  
Una **feature** è una singola proprietà o caratteristica misurabile dei dati, che il modello utilizza come input per prendere decisioni e fare previsioni.  
Tenendo presente che la qualità dei dati influisce direttamente sulle prestazioni del modello, il feature engineering è un processo fondamentale per ottenere un modello che funzioni in modo ottimale.  

> "Applied machine learning is basically feature engineering". Andrew Ng
{: .prompt-info }

### Curse of dimensionality

Ogni feature può essere vista come una dimensione: all’aumentare delle features crescono le dimensioni e, di conseguenza, la complessità con cui i dati vengono rappresentati.     
Sta a noi decidere quali features sono importanti e quali secondarie che possono essere scartate, in quanto non contribuiscono al processo decisionale del modello e aumentano inutilmente la complessità di rappresentazione dei dati.  
Difatti un grande numero di features contribuisce alla creazione di un **curse of dimensionality**, ovvero un aumento dello spazio dimensionale in cui sono collocati i dati, che può portare a problemi di overfitting e difficoltà nell'addestrare il modello.  
Per tenere sotto controllo questo problema si sceglie ponderatamente quali sono le features importanti.  
Esistono tecniche come **PCA (Principal Component Analysis)** che permettono di distillare le features in un minor numero in modo da continuare a mantenere la maggior informazione possibile.  
Altra tecnica è il **K-means clustering** che permette di raggruppare le features in base alle loro similarità.  
Queste tecniche sono algoritmi classici non algoritmi di machine learning.    


### Imputazione dei dati mancanti

Uno dei problemi più comuni nei dataset è la mancanza di alcuni dati.  
Ciò è deleterio per l'addestramento dei modelli e per questo motivo si cerca di sostituire i dati mancanti con altri valori.  
Una tecnica comune è quella di sostituire il dato mancante con la **media aritmetica** dei valori presenti nella stessa feature (colonna).  
Ha il vantaggio di non alterare la media della distribuzione dei valori della feature ma non sempre è una buona soluzione.    
All'interno di una feature, infatti, possono esserci valori che si discostano tantissimo dalla media. Sono chiamati, per questo, **outliers**, o valori anomali.  
In presenza di outliers è preferibile utilizzare la **mediana** per sostituire il dato mancante in quanto è meno sensibile a tali valori anomali.  
Tuttavia le sostituzioni con medie e mediane sono spesso semplicistiche e non sempre sono le soluzioni migliori.  
Non coprono situazioni in cui una feature è correlata ad altre feature, per esempio.  
Inoltre non è possibile usare medie e mediane per **feature categoriche**, es.: colore dei capelli, genere, etc.  

Se si hanno grandi dataset, può anche essere utile eliminare le righe che contengono dati mancanti, a patto che queste non alterino in alcun modo la distribuzione dei dati. Per esempio una riga da eliminare può contenere relazioni tra feature che possono essere utili per il modello, oltre al dato mancante e quindi non andrebbe scartata.  

Si può anche usare il machine learning per imputare i dati mancanti, esempio tramite algoritmo di ML **KNN (K-Nearest Neighbors)**.    
L'idea è quella di prendere una riga che ha un dato mancante e cercare le k righe più vicine (in base a una **distanza euclidea**) e usare la media dei dati di queste righe per sostituirla al dato mancante.  
Anche questo sistema, tuttavia, è più adatto per dati numerici e non per dati categorici.  
Infatti è meglio usare reti neurali per feature categoriche, che possono imparare a sostituire i dati mancanti in base alle feature correlate.  

Esiste una tecnica molto avanzata per imputare i dati mancanti, chiamata **MICE (Multiple Imputation by Chained Equations)**.  
Ad oggi è la tecnica migliore, ma è anche la più complessa da implementare.   

In generale, quando siamo in presenza di molti dati mancanti all'interno di un dataset, dovremmo sempre chiederci se sia possibile **allargare il dataset con nuove informazioni** che possano aiutare a ridurre il problema.


### Gestione dei dati sbilanciati

Nei dataset che contengono feature categoriche, è possibile che ci siano classi che sono rappresentate in modo sbilanciato.  
In questi casi, prendendo ad esempio un sistema di **classificazione binaria**, si hanno notevoli discrepanze tra i casi positivi ed i casi negativi.
   
> Esempio: nel caso di rilevamento delle frodi, il 99% delle transazioni documentate nel dataset sono legittime e solo l'1% sono frodi.  
> Ciò significa che l'alto numero di casi negativi influenza il modello a predire sempre il caso negativo, non individuando mai i casi positivi, cioè le frodi.
{: .prompt-tip }

Per cercare di mitigare il problema si possono usare varie tecniche:


##### Oversampling 

Per cercare di riequilibrare lo sbilanciamento si può usare la tecnica dell'oversampling.  
Questa tecnica consiste nel duplicare casi positivi (frodi) per bilanciare il numero di casi negativi (transazioni legittime).  
Si è sperimentato che aggiungere casi positivi minoritari all'interno del dataset funziona bene per addestrare le reti neurali.


##### Undersampling

Un'altra tecnica è l'undersampling, che consiste nell'eliminare casi negativi (transazioni legittime) per bilanciare il numero di casi positivi (frodi).  
Tuttavia, di solito, eliminare dati non è una buona soluzione, in quanto si perdono informazioni potenzialmente utili.  
In sostanza, diciamo che è fattibile se il dataset è davvero molto grande.  
L'arte del feature engineering consiste anche in questo: trovare il giusto equilibrio tra informazioni utili e dati da eliminare.


##### SMOTE generazione di dati sintetici

**SMOTE (Synthetic Minority Over-sampling Technique)** è una tecnica che genera nuovi casi positivi (frodi) usando l'algoritmo KNN (K-Nearest Neighbors) che abbiamo visto prima.  
Il concetto è simile al classico **data augmentation** che si usa per generare dati sintetici, per migliorare le prestazioni del modello.  
Calcolando la distanza euclidea dai casi positivi, vengono generati nuovi casi che sono simili ai casi positivi originali perché hanno le stesse caratteristiche, ma non sono copie esatte.   
Ovviamente la tecnica puà essere usata anche per creare casi negativi simili a quelli originali.  
Questo aiuta molto durante l'addestramento del modello.  


##### Modifica del threshold di classificazione

Usiamo sempre come esempio un sistema di classificazione binaria. 
Il modello predice caso positivo e caso negativo sotto forma di una certa **percentuale di probabilità**.  
Questa probabilità viene confrontata con un **threshold** (valore soglia) al di sotto del quale il modello predice caso negativo e al di sopra predice caso positivo.  
Nel feature engineering, il threshold può essere modificato per bilanciare il numero di casi positivi e negativi, per cercare di migliorare le predizioni dei casi positivi.  
Ovviamente il settaggio del giusto threshold è una scelta che deve essere fatta con molta attenzione e deve essere testata con cura, perché può anche sensibilmente peggiorare le prestazioni del modello.


### Gestione dei dati anomali (outliers)

Per comprendere le questioni relative ai dati anomali, è utile far riferimento a feature numeriche.  
Abbiamo visto che nei dataset possono esserci valori anomali, che si discostano tantissimo dalla **media** di tutto l'insieme dei valori di cui fanno parte.  
E' molto semplice individuare dati anomali in una distribuzione di probabilità visualizzandola in un istogramma.  
Misurare di quanto i valori si discostano dalla media vuol dire calcolarne la **varianza** o la **deviazione standard**.  
Difatti **la varianza indica la dispersione dei valori rispetto alla media**.  
Essa utilizza valori al quadrato, per questo motivo non è sempre molto intuitiva da leggere (ad esempio, se i dati sono in Euro, la varianza sarà in Euro al quadrato).   
Per questo motivo si usa più spesso la **deviazione standard**, che è semplicemente **la radice quadrata della varianza**.  
Riportando il numero all'unità di misura originale (di nuovo in Euro), ci fa capire subito di quanto i dati si allontanano dalla media.

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
> Il valore `200` è enormemente al di fuori di questo intervallo, ed è un **dato anomalo (outlier)** impossibile da non notare.
{: .prompt-tip }

Anche i dati anomali possono essere gestiti in vari modi, è possibile **eliminarli**, **sostituirli** con la media o la mediana dei dati presenti nella stessa feature, o con un valore predefinito, o **lasciarli** così com'è.   
Tutto sta nel saper individuare la corretta soluzione in base alla natura del dato anomalo, e a quella del dataset e non è sempre semplice!

> **Quando conviene eliminare un dato anomalo?**
> Immaginiamo un dataset con i tempi di consegna (in giorni) di un corriere: `2, 3, 2, 4, 3, 365`.  
> Il valore `365` è anomalo: un pacco consegnato dopo un anno intero!    
> Indagando, scopriamo che quel record appartiene a un **ordine di test** creato dagli sviluppatori in ambiente di produzione, mai cancellato, e lasciato "aperto" per mesi.  
> Non rappresenta quindi una consegna reale: è **rumore operativo**.  
> In questo caso è bene **eliminarlo**, perché altrimenti falserebbe le metriche del servizio (es. alzando artificialmente il tempo medio di consegna).
> 
> Diverso è invece il caso di un tempo di consegna di `15` giorni dovuto a uno sciopero dei trasporti o a un pacco spedito all'estero: lì l'outlier è un evento **reale**, e va tenuto perché racconta qualcosa di vero sul business.
{: .prompt-tip }

All'interno dell'ecosistema AWS esistono algoritmi come **AWS Random Cut Forest** che possono aiutare a identificare i dati anomali.  
Possiamo quindi usare Random Cut Forest all'interno di **Kinesis Analytics** o dentro **SageMaker**, per esempio.

##### Esempio: distribuzione dei redditi

Supponiamo di voler modellare la distribuzione dei redditi annuali di una popolazione: stiamo semplicemente osservando quanto guadagna ogni persona all'anno e cosa questo significa per l'insieme dei dati nel suo complesso.  
Immaginiamo una distribuzione di redditi "normali" centrata intorno ai **27.000 €** all'anno. Ora aggiungiamo un singolo **miliardario** e tracciamo l'istogramma.  
Quel miliardario modifica di molto la scala della distribuzione: tutte le persone comuni finiscono ammassate in una piccola fascia a sinistra, mentre l'outlier (quel dato estremo che non si nota nemmeno sulla scala) distorce enormemente la lettura dei dati.

![Istogramma dei redditi distorto da un miliardario](/assets/images/income_with_outlier.svg){: width="720"}

In questo scenario è la **media** a diventare poco significativa: un solo reddito altissimo la trascina verso valori bizzarri, lontani dai 27.000 € da cui eravamo partiti.  
La **mediana**, invece, rimane vicina a quel valore centrale, perché è meno sensibile agli estremi.  

Abbiamo visto che calcolando media e deviazione standard del dataset riusciamo agevolmente a identificare gli outlier per poi **scartare** i valori che si trovano al di fuori, ad esempio, di **due deviazioni standard**.  Applicando questa regola e riprovando a tracciare l'istogramma, otteniamo una distribuzione molto più leggibile: scartando il miliardario, anche la media torna più vicina ai 27.000 € di partenza.

![Istogramma dei redditi dopo aver scartato l'outlier](/assets/images/income_without_outlier.svg){: width="720"}  

Ma qui arriva la domanda davvero importante: **è opportuno "sbarazzarsi" 😅 di quel miliardario?**   
Dipende dall'obiettivo di business.  
Se stiamo modellando il reddito del "lavoratore tipico", rimuoverlo ha senso.  
Se invece stiamo studiando disuguaglianze nei redditi, o aspetti fiscali, quel dato è reale e prezioso: eliminarlo altererebbe proprio il fenomeno che ci interessa studiare.   
Algoritmi come Random Cut Forest aiutano a **individuare** le anomalie; ma la decisione di scartarle, sostituirle o tenerle resta sempre una scelta di dominio.



### Raggruppamento in classi (binning)

Talvolta per modellare meglio i dati (es dati numerici), è utile raggrupparli in classi di appartenenza basandosi su range di valori.  

> Ad esempio, invece di usare l'età esatta di una persona (es. `18, 25, 42, 65` anni),  
> possiamo creare delle classi: `18-25`, `26-40`, `41-60`, `60+`.  
> Questo aiuta il modello a cogliere schemi generali (es. i giovani adulti potrebbero avere abitudini di acquisto simili).
{: .prompt-tip }

Creare delle classi aiuta a "nascondere" i piccoli errori nei dati.  
Se l'età di una persona viene registrata per sbaglio come 27 anni anziché 26, finirà quasi sicuramente nella stessa classe (es. `26-40`) e in questo modo, l'imprecisione diventa irrilevante per il modello.  

Tuttavia, bisogna fare **attenzione**: il binning comporta inevitabilmente una **perdita di informazioni** (non sappiamo più se un utente ha 26 o 40 anni, sappiamo solo che appartiene a quella fascia). Un altro motivo per usarlo è se si è costretti a utilizzare un algoritmo che accetta solo dati categorici e non numerici, ma in generale è una scelta da ponderare bene.


##### Quantile Binning
Una variante molto utile è il **quantile binning**. Invece di creare classi basate su intervalli di valori fissi, le classi vengono create in base alla distribuzione dei dati, in modo che **ogni classe contenga esattamente lo stesso numero di campioni**.  
Questo assicura che nessuna classe sia vuota o sovrappesata rispetto alle altre.

### Trasformazione dei dati (Data Transformation)
Un'altra tecnica fondamentale è l'applicazione di funzioni matematiche alle feature per **trasformarle** o **crearne di nuove** che siano più "digeribili" per gli algoritmi.  
Molti modelli, infatti, faticano a individuare pattern in dati con andamenti non lineari.   
Un esempio pratico è la **trasformazione logaritmica**: se una feature presenta un andamento esponenziale, applicare un logaritmo può trasformare la curva in una linea retta (relazione lineare), rendendo molto più semplice per il modello coglierne il trend.  




> **Esempio: sistema di raccomandazione di YouTube**
> Un esempio reale viene da un paper pubblicato da YouTube sul funzionamento dei loro sistemi di raccomandazione.   
> Per ogni feature numerica $x$ (ad esempio, il tempo trascorso dall'ultima visualizzazione di un video), YouTube non si limita a passare al modello il valore grezzo $x$, ma fornisce anche il suo quadrato ($x^2$) e la sua radice quadrata ($\sqrt{x}$).  
> In questo modo, aiutano la rete neurale a catturare facilmente andamenti super-lineari o sub-lineari nei dati.  
{: .prompt-tip }

Naturalmente, bisogna sempre bilanciare: aggiungere troppe feature derivate rischia di farci ritornare nel problema del **curse of dimensionality**.  



Un'altra operazione molto comune durante la preparazione dei dati è la **codifica** (encoding). Spesso i modelli, in particolare nel mondo del Deep Learning, richiedono che i dati in ingresso abbiano un formato specifico e bisogna trasformarli di conseguenza.

### Encoding (One-Hot Encoding)
Encoding è il processo di conversione di dati da un formato a un altro.  
Un esempio classico e fondamentale è il **One-Hot Encoding**.  
L'idea alla base è creare un "contenitore" (o *bucket*) per ogni possibile categoria presente nei dati.    
Si assegna il valore `1` alla categoria a cui appartiene il dato (indicando che è presente, o "hot") e `0` a tutte le altre (indicando che non è quella categoria).

> **Esempio: Riconoscimento della scrittura**
> Immaginiamo di costruire un modello di Deep Learning per riconoscere numeri scritti a mano da 0 a 9. Se vogliamo rappresentare il fatto che un'immagine contenga il numero "8", non passiamo semplicemente il valore numerico `8` al modello. 
> Creiamo invece 10 "slot" (uno per ogni possibile cifra). Inseriremo un `1` nello slot corrispondente all'8 (il nono slot, se iniziamo a contare da zero) e `0` in tutti gli altri:
> `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]`
{: .prompt-tip }

Nel Deep Learning, i neuroni lavorano tipicamente su **stati di attivazione** (booleano acceso/spento), non possiamo semplicemente "alimentare" un singolo neurone di input con il numero 8 o il numero 3 e aspettarci che il modello lo interpreti correttamente come una categoria.  
Invece, utilizzando il One-Hot Encoding, il dato viene fornito a 10 input diversi, che corrispondono a 10 neuroni, dove solo uno di essi riceve il segnale di attivazione `1`, mentre tutti gli altri rimangono inattivi `0`.

### Normalizzazione dei dati

Un'altra fase quasi sempre obbligatoria nella preparazione dei dati è la **normalizzazione** dei dati.    
Se non ridimensioniamo i dati tramite normalizzazione, le feature con valori numerici assoluti più alti finiranno per avere un "peso" sproporzionato all'interno del modello.  
Normalizzare vuol dire trasformare i dati in una **scala di grandezza comparabile**.    
Molti modelli, infatti, specialmente le reti neurali, lavorano meglio se i dati di input sono distribuiti normalmente (range `[0, 1]`) in **distribuzioni gaussiane uniformi**, o per lo meno se tutte le feature si trovano su scale di grandezza comparabili.  

> Per approfondire, vedere gli appunti su ["Normalization"](https://bigghis.github.io/AI-appunti/guide/optimizations/normalize.html){: target="_blank" }.
{: .prompt-info }



> **Esempio: Reddito vs Età**
> Immaginiamo di addestrare un sistema basandoci sul reddito di una persona (es. `50.000`) e sulla sua età (es. `30`). Se non normalizziamo questi dati portandoli su scale comparabili, il valore "50.000" del reddito sovrasterà completamente il "30" dell'età. Il modello darà un'importanza enorme al reddito ignorando quasi del tutto l'età, portando a risultati scadenti.
{: .prompt-tip }

Ci sono delle eccezioni: alcuni algoritmi, come gli alberi decisionali (*Decision Trees*), sono insensibili alla scala dei dati. Tuttavia, per la maggior parte dei modelli, lo normalizzazione è fondamentale.


> **Attenzione ai risultati:** Se il modello serve a prevedere un valore numerico (e non una categoria) e abbiamo normalizzato anche la variabile target in fase di addestramento, ricordarsi sempre di **invertire la scala** (reverse scaling) sui risultati finali.  
> Altrimenti, il modello restituirà un numero normalizzato privo di significato per l'utente, invece del valore reale atteso.
{: .prompt-danger }    

### Shuffling dei dati

Infine, un'ultima pratica è lo **shuffling** (mescolamento) dei dati di addestramento.  

Spesso i dati mantengono un "segnale residuo" o un **bias** nascosto dovuto semplicemente all'ordine temporale o logico in cui sono stati originariamente raccolti.  
Mescolando casualmente l'ordine delle righe prima di fornirle al modello, si elimina questo effetto collaterale indesiderato. 

Esistono molti casi reali in cui modelli con prestazioni inizialmente pessime sono migliorati drasticamente con un semplice shuffle dell'input!

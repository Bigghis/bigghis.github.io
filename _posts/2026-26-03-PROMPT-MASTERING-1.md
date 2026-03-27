---
title: "Mastering Prompt Engineering"
description: "Le basi per costruire prompt efficaci"
date: 2026-03-26 12:00:00 +0530
categories: [Prompt Engineering]
tags: [Prompt, Mastering, Prompt Engineering]
comments: false
---

### Prompt Engineering

Per un utilizzo efficiente di un LLM è fondamentale saper progettare e sviluppare prompt ottimizzati.  
Questo processo è noto come **Prompt Engineering**.  

In genere possiamo suddividere un prompt in 4 blocchi:

* **Context**: il contesto del prompt, cioè un insieme di informazioni esterne che guidano il modello su come eseguire compiti. (**definisce *chi* è il modello e la sua competenza.**)
* **Instructions**: un task che il modello deve eseguire con la descrizione del compito da eseguire. (**descrive *cosa* deve fare.**)
* **Input data**: i dati che il modello deve utilizzare per eseguire il task. (**fornisce i *dati concreti* su cui lavorare.**)
* **Output indicator**: delle indicazioni sul formato della risposta che il modello deve produrre. (**specifica il *formato* atteso.**)

Un classico prompt può essere il seguente:  
> **[Context]**  
> Sei un nutrizionista esperto specializzato in diete mediterranee.  
> Hai 20 anni di esperienza nella creazione di piani alimentari personalizzati per pazienti con intolleranze alimentari.  
>  
> **[Instructions]**  
> Analizza il profilo del paziente fornito e crea un piano alimentare settimanale (7 giorni) che rispetti le sue intolleranze, mantenendo un apporto calorico giornaliero tra 1800 e 2000 kcal.  
>  
> **[Input data]**  
> Paziente: Marco, 35 anni, 80 kg, 175 cm.  
> Intolleranze: lattosio, glutine.  
> Obiettivo: perdita di peso graduale.  
> Attività fisica: corsa 3 volte a settimana.  
> Preferenze: cucina italiana, non ama il pesce crudo.  
>  
> **[Output indicator]**  
> Presenta il piano in formato tabella con colonne: Giorno, Colazione, Spuntino, Pranzo, Merenda, Cena.  
> Sotto la tabella, aggiungi una nota con il totale calorico medio giornaliero e le macro (proteine, carboidrati, grassi) in percentuale.  
{: .prompt-info }


A partire da questo schema dei 4 blocchi è possibile effettuare delle variazioni di prompt per ottenere risultati diversi.  

### Negative Prompting

Un **negative prompt** è un prompt che viene utilizzato per impedire al modello di produrre risposte specifiche o impedire di produrre risposte che contengono determinate parole.  
Un negative prompt può aiutare a **mantenere il focus del modello su un determinato argomento** e quindi **migliorare la chiarezza delle risposte**.  

Riprendendo l'esempio precedente, possiamo aggiungere dei negative prompting:  
> **[Context]**  
> Sei un nutrizionista esperto specializzato in diete mediterranee.  
> Hai 20 anni di esperienza nella creazione di piani alimentari personalizzati per pazienti con intolleranze alimentari.  
>  
> **[Instructions]**  
> Analizza il profilo del paziente fornito e crea un piano alimentare settimanale (7 giorni) che rispetti le sue intolleranze, mantenendo un apporto calorico giornaliero tra 1800 e 2000 kcal.  
> **Non suggerire integratori alimentari o prodotti commerciali specifici.**  
> **Non includere ricette elaborate che richiedano più di 30 minuti di preparazione.**  
> **Non proporre pasti ripetitivi: ogni giorno deve avere piatti diversi.**  
>  
> **[Input data]**  
> Paziente: Marco, 35 anni, 80 kg, 175 cm.  
> Intolleranze: lattosio, glutine.  
> Obiettivo: perdita di peso graduale.  
> Attività fisica: corsa 3 volte a settimana.  
> Preferenze: cucina italiana, non ama il pesce crudo.  
>  
> **[Output indicator]**  
> Presenta il piano in formato tabella con colonne: Giorno, Colazione, Spuntino, Pranzo, Merenda, Cena.  
> Sotto la tabella, aggiungi una nota con il totale calorico medio giornaliero e le macro (proteine, carboidrati, grassi) in percentuale.  
> **Non aggiungere disclaimer medici o avvertenze legali.**  
> **Non inserire spiegazioni o commenti al di fuori della tabella e della nota riassuntiva.**  
{: .prompt-info }


### Zero-shot Prompting
Nello **zero-shot prompting** non si fornisce alcun esempio al modello: si descrive solo il task da eseguire e ci si affida alla conoscenza già acquisita dal modello durante il training.  
Funziona bene per compiti semplici e comuni.  

> Classifica il sentimento della seguente frase come "positivo", "negativo" o "neutro".  
> Frase: "Il ristorante aveva un'atmosfera fantastica ma il cibo era mediocre."
{: .prompt-info }

### One-shot Prompting
Nel **one-shot prompting** si fornisce **un singolo esempio** prima di porre una domanda. L'esempio guida il modello sul formato e sul tipo di risposta attesi.  

> Classifica il sentimento della frase.  
>  
> Frase: "Il film mi ha commosso profondamente."  
> Sentimento: positivo  
>  
> Frase: "Il ristorante aveva un'atmosfera fantastica ma il cibo era mediocre."  
> Sentimento:
{: .prompt-info }

### Few-shot Prompting
Nel **few-shot prompting** si forniscono **più esempi** per aiutare il modello a fornire una risposta corretta.  

> Classifica il sentimento della frase.  
>  
> Frase: "Il film mi ha commosso profondamente."  
> Sentimento: positivo  
>  
> Frase: "Ho perso il treno e sono arrivato in ritardo."  
> Sentimento: negativo  
>  
> Frase: "La riunione è stata spostata a domani."  
> Sentimento: neutro  
>  
> Frase: "Il ristorante aveva un'atmosfera fantastica ma il cibo era mediocre."  
> Sentimento:
{: .prompt-info }

### Chain of Thought Prompting
Nel **Chain of Thought (CoT) prompting** si chiede al modello di esplicitare i **passaggi intermedi del ragionamento**, tramite un ragionamento a step, prima di arrivare alla risposta finale. Questo migliora le prestazioni su task che richiedono logica, calcolo o ragionamento multi-step.  

> Scriviamo una breve presentazione per un'azienda che produce olio d'oliva biologico in Puglia.  
>  
> **Prima**, descrivi l'azienda e il territorio.  
> **Poi**, presenta il metodo di produzione.  
> **Successivamente**, spiega cosa rende unico il prodotto.  
> **Infine**, concludi con un invito all'acquisto.  
>  
> Scrivi la presentazione seguendo questo piano. Ragiona passo per passo.
{: .prompt-info }


### Prompt Template
Un **prompt template** è un prompt con dei **placeholder** che vengono sostituiti dinamicamente al momento dell'uso.    
Questo permette di riutilizzare la stessa struttura di prompt per input diversi, senza doverlo riscrivere ogni volta.  
I placeholder sono tipicamente racchiusi tra doppie parentesi graffe `{{variabile}}`.  
I prompt template sono utili quando si integra un LLM all'interno di un'applicazione software, dove i valori dei placeholder vengono popolati programmaticamente.  

> **[Context]**  
> Sei un esperto scrittore di sceneggiature cinematografiche con 30 anni di esperienza a Hollywood.  
>  
> **[Instructions]**  
> In base alla descrizione del film fornita dall'utente, scrivi una sinossi professionale di massimo 10 righe, seguita da una lista di 3 personaggi principali con nome e breve descrizione.  
>  
> **[Input data]**  
> Descrizione dell'utente: "{{descrizione_film}}"  
>  
> **[Output indicator]**  
> Rispondi con due sezioni: "Sinossi" e "Personaggi principali".  
{: .prompt-info }

L'applicazione chiede all'utente: *"Descrivi il film che vuoi creare"* e inserisce la sua risposta nel placeholder `{{descrizione_film}}`.  
Ad esempio, se l'utente risponde `"Un thriller ambientato su una stazione spaziale dove l'equipaggio scopre che uno di loro non è umano"`, il prompt generato sarà:

> **[Context]**  
> Sei un esperto scrittore di sceneggiature cinematografiche con 30 anni di esperienza a Hollywood.  
>  
> **[Instructions]**  
> In base alla descrizione del film fornita dall'utente, scrivi una sinossi professionale di massimo 10 righe, seguita da una lista di 3 personaggi principali con nome e breve descrizione.  
>  
> **[Input data]**  
> Descrizione dell'utente: "Un thriller ambientato su una stazione spaziale dove l'equipaggio scopre che uno di loro non è umano"  
>  
> **[Output indicator]**  
> Rispondi con due sezioni: "Sinossi" e "Personaggi principali".  
{: .prompt-info }

E' possibile che il template contenga esempi (few shot) per aiutare il modello nella comprensione del task, e che questi siano completatamente trasparenti all'utente esterno che utilizza l'applicazione e fornisce il suo prompt.  

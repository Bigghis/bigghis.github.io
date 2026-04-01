---
title: "Reinforcement Learning"
description: "Reinforcement Learning e le sue basi"
date: 2026-03-27 12:00:00 +0530
categories: [AI]
tags: [AI, Machine Learning, Reinforcement Learning, RLHF]
comments: false
protected: true
---

### Reinforcement Learning

Il **Reinforcement Learning (RL)** è un tipo di Machine Learning in cui un **agente** impara a prendere decisioni eseguendo azioni all'interno di un **ambiente**, con l'obiettivo di **massimizzare una ricompensa cumulativa** nel tempo.

A differenza del Supervised Learning (dove il modello impara da esempi etichettati) e dell'Unsupervised Learning (dove il modello cerca pattern nei dati), nel Reinforcement Learning il modello **impara per tentativi ed errori**, simulando molte volte lo stesso scenario e imparando dai propri successi e dai propri sbagli.

#### Concetti chiave

* **Agent** - il soggetto che impara e prende le decisioni
* **Environment** - il sistema esterno con cui l'agente interagisce
* **State** - la situazione attuale dell'ambiente
* **Action** - le scelte compiute dall'agente
* **Reward** - il feedback che l'ambiente restituisce in base alle azioni dell'agente
* **Policy** - la strategia che l'agente utilizza per determinare quale azione compiere in base allo stato corrente

#### Come funziona

Il processo di apprendimento segue un ciclo continuo:

1. L'agente **osserva** lo stato corrente dell'ambiente
2. **Seleziona un'azione** in base alla propria policy
3. L'ambiente **transita in un nuovo stato** e fornisce una **ricompensa**
4. L'agente **aggiorna la propria policy** per migliorare le decisioni future

L'obiettivo finale è **massimizzare la ricompensa cumulativa nel tempo**.

#### Esempio pratico: un robot in un labirinto

Immaginiamo di addestrare un robot a navigare un labirinto:

* Il **robot** è l'agente
* Il **labirinto** è l'ambiente
* La **posizione** del robot è lo stato

Ad ogni passo il robot:
1. Osserva la propria posizione (State)
2. Sceglie una direzione in cui muoversi (Action)
3. Riceve una ricompensa:
   * **-1** per ogni passo compiuto (incentivo a trovare la via più breve)
   * **-10** per aver sbattuto contro un muro (penalità)
   * **+100** per aver raggiunto l'uscita (ricompensa massima)
4. Aggiorna la propria policy in base alla ricompensa e alla nuova posizione

**Risultato**: dopo molte simulazioni, il robot impara a navigare il labirinto in modo efficiente, evitando i muri e trovando il percorso più breve verso l'uscita.

#### Applicazioni del Reinforcement Learning

* **Gaming** - insegnare all'AI a giocare a giochi complessi (es. Scacchi, Go)
* **Robotica** - navigazione e manipolazione di oggetti in ambienti dinamici
* **Finanza** - gestione di portafogli e strategie di trading
* **Sanità** - ottimizzazione di piani terapeutici
* **Veicoli autonomi** - pianificazione del percorso e processo decisionale

---

### RLHF - Reinforcement Learning from Human Feedback

L'**RLHF** (Reinforcement Learning from Human Feedback) è una variante del Reinforcement Learning che utilizza il **feedback umano** per aiutare i modelli di ML ad apprendere in modo più efficiente e allineato alle esigenze delle persone.

Nel Reinforcement Learning classico esiste una **funzione di ricompensa** (reward function) che guida l'apprendimento. L'RLHF **incorpora il feedback umano all'interno di questa funzione di ricompensa**, così che il modello sia più allineato con gli obiettivi, i desideri e le necessità umane.

L'RLHF è utilizzato in tutta la Generative AI, inclusi i modelli LLM, e **migliora significativamente le prestazioni** del modello.

#### Come funziona l'RLHF

Prendiamo come esempio la creazione di un **chatbot aziendale** per la conoscenza interna dell'azienda:

**1. Raccolta dati**  
Viene creato un insieme di prompt e risposte generate da esseri umani.  
Esempio: *"Dov'è la sede del dipartimento HR a Boston?"*

**2. Fine-tuning supervisionato del modello**  
Si effettua il fine-tuning di un modello esistente con la conoscenza interna dell'azienda.  
Il modello genera le proprie risposte ai prompt umani, e queste vengono confrontate matematicamente con le risposte generate dagli umani.

**3. Costruzione di un reward model separato**  
Gli esseri umani indicano quale risposta preferiscono tra più alternative generate dallo stesso prompt.  
Questo crea un **modello di ricompensa** capace di stimare quale risposta un umano preferirebbe.

**4. Ottimizzazione del modello con il reward model**  
Il reward model viene usato come funzione di ricompensa per il Reinforcement Learning.  
Questa fase può essere **completamente automatizzata**: il modello continua a migliorarsi senza ulteriore intervento umano.


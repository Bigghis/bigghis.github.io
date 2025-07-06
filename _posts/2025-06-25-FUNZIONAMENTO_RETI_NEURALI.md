---
title: "How Neural Networks work"
description: A simple explanation of how neural networks work
date: 2025-06-25 12:00:00 +0530
categories: [Base, Neural Network, LLM]
tags: [Base, Neural Network, LLM]
comments: false
---

DISCLAIMER: Cerchiamo di mettere in ordine dei princìpi generali per il funzionamento delle reti neurali. Quanto elencato è una spiegazione di base e non un insieme di regole ferree, anche perché la sperimentazione in questo campo è in continua evoluzione. Non è detto, ad esempio, che l'allineamento venga sempre effettuato con l'approccio del Reinforcement Learning, piuttosto che con la discesa del gradiente.


### Funzionamento delle reti neurali

Una rete neurale è un modello matematico, composto da numeri e funzioni matematiche.  
I numeri sono detti **parametri della rete** o **pesi**, le funzioni sono dette **attivazioni**.  
Un **Large Language Model (LLM)** è una rete neurale che ha imparato a leggere e scrivere testo.  
In una prima fase la rete passa attraverso un processo di addestramento, chiamato **training**.  
In una seconda fase la rete viene utilizzata per fare delle predizioni, il processo viene chiamato **inferenza**.  

### Addestramento della rete (Training)
Prendiamo come esempio un modello di LLM come ChatGPT-4, esso possiede 1800 milioni di parametri.  
Usa una rappresentazione matematica chiamata **embedding** per rappresentare le parole che vengono convertite in numeri. 
Questi numeri viaggiano attraverso la rete e dopo calcoli matematici diventano parametri della rete.    
Alla fine i numeri verranno riconvertiti in parole e la rete produrrà una predizione di testo.  
Addestrare una rete neurale significa trovare i valori dei parametri che permettono alla stessa di fare delle predizioni corrette.   
L'addestramento consiste nel far vedere alla rete milioni di esempi di testo, in modo che la rete possa imparare a fare delle predizioni corrette, 
aggiustando i valori dei parametri.  
L'addestramento dei moderni LLM avviene in tre sottofasi:
1. **Pre-training**: viene insegnato alla rete a produrre frasi di senso compiuto.  
2. **Fine-tuning**: viene insegnato alla rete a produrre frasi sensate in base a un contesto specifico.  
3. **Alignment**: viene insegnato alla rete a produrre frasi sensate in base a un contesto specifico, con un tono specifico.

#### Pre-training (Unsupervised learning)
E' la fase più importante, può essere paragonata al processo di apprendimento del parlato da parte di un bambino.  
Prima del pre-training, la rete viene inizializzata con dei valori casuali per i parametri.  
Anche detto **unsupervised learning** perché dando in pasto alla rete milioni di frasi di testo, la rete impara da sola a trovare le relazioni tra le parole.  
L'algoritmo di ottimizzazione usato in questa fase è il **Gradient Descent** (discesa del gradiente), che trova i migliori valori possibili per i parametri.  
L'algoritmo legge milioni e milioni di frasi di testo, chiamate **batch**, e calcola la differenza tra la predizione della rete e il testo corretto.  
Questa differenza viene chiamata **loss** e l'algoritmo di ottimizzazione tenta di minimizzare questo valore.  
Al termine del pre-training, abbiamo ottenuto GPT (Generative Pre-trained Transformer), che produce la prossima parola
sulla base delle parole precedenti.  


#### Fine-tuning (Supervised learning)  
A questo punto la rete è in grado di produrre frasi, ma non sa chattare, non sa tradurre, non sa produrre riassunti, non sa rispondere a domande specifiche, per esempio.   
Per questo motivo i suoi parametri devono essere modificati, raffinandoli ancora un po' per compiti specifici.  
Anche detto **supervised learning** perché dando in pasto alla rete molti esempi di testo e le loro risposte, la rete impara a produrre risposte coerenti con i dati di addestramento.  
L'algoritmo di ottimizzazione usato in questa fase è sempre il **Gradient Descent**.  
Vengono letti anche stavolta molti esempi di testo, per task specifici, es chat di dialogo tra utenti, traduzioni di testi da una lingua a un'altra, riassunti di testi, etc.  
Al termine di questa fase abbiamo aggiunto a GPT la capacità di **chat** (ChatGPT), oltre che la capacità di leggere e scrivere testo in base a un contesto specifico. 

#### Alignment
I parametri della rete vengono aggiustati ancora un po per poter produrre testo coerente con i prncìpi umani. Di solito in questa fase si usa l'approccio del **Reinforcement Learning**. 
Le risposte di un LLM vengono valutate da umani che premiano le risposte ritenute più opportune rispetto ad altre meno opportune. (**Reinforcement Learning from Human Feedback (RLHF)**.)

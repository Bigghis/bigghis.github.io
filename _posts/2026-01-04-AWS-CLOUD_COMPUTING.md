---
title: "AWS e Cloud Computing"
description: "Panoramica dei concetti di AWS e Cloud Computing"
date: 2026-03-27 12:00:00 +0530
categories: [AI]
tags: [AWS, Cloud Computing]
comments: false
protected: true
---

### AWS e Cloud Computing


### L'approccio IT tradizionale e i suoi problemi

Tradizionalmente, per mettere online un'applicazione, un'azienda doveva costruire e gestire la propria infrastruttura IT: un data center fisico (anche solo un garage o un ufficio con dei server).

Questo approccio comporta diversi problemi:

- **Costi fissi elevati**: affitto dei locali, alimentazione elettrica, raffreddamento, manutenzione.
- **Tempi lunghi**: aggiungere o sostituire hardware richiede tempo.
- **Scalabilità limitata**: difficile adattarsi rapidamente a picchi di domanda.
- **Personale dedicato**: serve un team operativo 24/7 per monitorare l'infrastruttura.
- **Rischio disastri**: terremoti, incendi, blackout possono compromettere tutto.

La domanda chiave diventa: *possiamo esternalizzare tutto questo?* La risposta è il **cloud computing**.

---

### Cloud Computing

#### Cos'è il Cloud Computing

Il cloud computing è la fornitura **on-demand** di risorse IT (potenza di calcolo, storage, database, applicazioni) tramite Internet, con un modello di pagamento **pay-as-you-go** (paghi solo quello che usi).

In pratica:

- Puoi ottenere esattamente il tipo e la quantità di risorse di cui hai bisogno.
- Puoi accedere alle risorse quasi istantaneamente.
- Non devi comprare né gestire hardware: il provider (come AWS) possiede e mantiene l'infrastruttura, tu la usi tramite un'interfaccia web.

Esempi di servizi cloud che usi già ogni giorno: **Gmail** (email), **Dropbox** (storage), **Netflix** (video on demand, costruito su AWS).

#### Modelli di deployment

| Modello | Descrizione | Caratteristiche |
|---------|-------------|-----------------|
| **Private Cloud** | Servizi cloud usati da una singola organizzazione, non esposti al pubblico | Controllo completo, sicurezza elevata, adatto a esigenze specifiche |
| **Public Cloud** | Risorse di proprietà di un provider terzo, erogate via Internet (es. AWS, Azure, GCP) | Scalabilità, costi ridotti, nessuna gestione hardware |
| **Hybrid Cloud** | Parte dell'infrastruttura resta on-premise, parte viene estesa al cloud | Controllo sugli asset sensibili + flessibilità del cloud pubblico |

#### Le cinque caratteristiche del Cloud Computing

1. **Self-service on-demand**: gli utenti possono creare risorse senza intervento umano del provider.
2. **Accesso tramite rete**: le risorse sono disponibili via rete da qualsiasi piattaforma client.
3. **Multi-tenancy e resource pooling**: più clienti condividono la stessa infrastruttura fisica, mantenendo sicurezza e privacy.
4. **Elasticità e scalabilità rapida**: le risorse vengono acquisite e rilasciate automaticamente in base alla domanda.
5. **Servizio misurato**: l'utilizzo viene misurato e si paga esattamente per ciò che si è consumato.

#### I sei vantaggi del Cloud Computing

1. **Da CAPEX a OPEX**: si passa da spese in conto capitale (comprare hardware) a spese operative (pagamento on-demand). Riduzione del TCO (Total Cost of Ownership).
2. **Economie di scala**: AWS opera su larga scala, i costi per l'utente si riducono.
3. **Basta indovinare la capacità**: si scala in base all'utilizzo reale misurato.
4. **Maggiore velocità e agilità**: risorse disponibili in pochi minuti.
5. **Niente più data center da gestire**: nessun costo di manutenzione fisica.
6. **Presenza globale in pochi minuti**: grazie all'infrastruttura distribuita di AWS.

#### Problemi risolti dal Cloud

- **Flessibilità**: cambiare tipo di risorse quando serve.
- **Costo-efficacia**: paghi solo quello che usi.
- **Scalabilità**: gestire carichi maggiori potenziando l'hardware (scale-up) o aggiungendo nodi (scale-out).
- **Elasticità**: capacità di espandersi e contrarsi automaticamente.
- **Alta disponibilità e fault-tolerance**: distribuzione su più data center.
- **Agilità**: sviluppare, testare e rilasciare software rapidamente.

#### Tipi di Cloud Computing

| Tipo | Descrizione | Gestione | Esempi |
|------|-------------|----------|--------|
| **IaaS** (Infrastructure as a Service) | Blocchi base dell'IT cloud: rete, calcolo, storage. Massima flessibilità. | Tu gestisci: OS, middleware, runtime, dati, applicazioni | Amazon EC2, Azure, GCP, Digital Ocean |
| **PaaS** (Platform as a Service) | Il provider gestisce l'infrastruttura sottostante. Ti concentri su deployment e gestione delle applicazioni. | Tu gestisci: dati e applicazioni | AWS Elastic Beanstalk, Heroku, Google App Engine |
| **SaaS** (Software as a Service) | Prodotto completo, gestito interamente dal provider. | Tu usi il prodotto | Gmail, Dropbox, Zoom, AWS Rekognition |

#### Pricing AWS: concetti base

AWS si basa su tre pilastri di prezzo (pay-as-you-go):

| Voce | Cosa paghi |
|------|------------|
| **Compute** | Il tempo di calcolo utilizzato |
| **Storage** | I dati archiviati nel cloud |
| **Data transfer OUT** | I dati trasferiti *in uscita* dal cloud (il trasferimento *in ingresso* è gratuito) |

---

### Infrastruttura globale AWS

L'infrastruttura di AWS è organizzata su più livelli gerarchici distribuiti nel mondo.

#### Region

- AWS dispone di **Region** distribuite in tutto il mondo (es. `us-east-1`, `eu-west-3`).
- Ogni Region è un **cluster di data center**.
- La maggior parte dei servizi AWS è **region-scoped** (legata a una specifica regione).

**Come scegliere una Region?** Quattro criteri:

1. **Compliance**: i dati non lasciano mai una Region senza permesso esplicito. Scegli in base ai requisiti legali.
2. **Prossimità ai clienti**: una Region vicina ai tuoi utenti riduce la latenza.
3. **Servizi disponibili**: non tutti i servizi o le funzionalità sono presenti in tutte le Region.
4. **Prezzo**: il costo varia da Region a Region.

#### Availability Zones (AZ)

- Ogni Region ha tipicamente **3 Availability Zones** (minimo 3, massimo 6).
- Ogni AZ è composta da uno o più data center fisicamente separati, con alimentazione, rete e connettività ridondanti.
- Le AZ sono **isolate tra loro** per proteggere dai disastri locali.
- Sono collegate tra loro tramite rete ad **alta banda e bassissima latenza**.

Esempio: la Region Sydney (`ap-southeast-2`) ha tre AZ: `ap-southeast-2a`, `ap-southeast-2b`, `ap-southeast-2c`.

### Edge Locations (Points of Presence) (CDN)

- AWS ha oltre **400 Points of Presence** (Edge Locations e Regional Caches) in 90+ città in oltre 40 paesi.
- Servono per distribuire contenuti agli utenti finali con **latenza ridotta** (es. tramite Amazon CloudFront, la CDN di AWS).

#### Servizi globali vs servizi regionali

| Servizi globali | Servizi regionali (esempi) |
|-----------------|---------------------------|
| **IAM** (Identity and Access Management) | **Amazon EC2** (IaaS) |
| **Route 53** (DNS) | **Elastic Beanstalk** (PaaS) |
| **CloudFront** (CDN) | **Lambda** (FaaS - Function as a Service) |
| **WAF** (Web Application Firewall) | **Rekognition** (SaaS) |

#### Modello di responsabilità condivisa (Shared Responsibility Model)

AWS adotta un modello di responsabilità condivisa tra il provider e il cliente:

- **AWS è responsabile della sicurezza *del* cloud**: infrastruttura fisica, hardware, rete, data center.
- **Il cliente è responsabile della sicurezza *nel* cloud**: configurazione dei servizi, dati, gestione degli accessi, crittografia.

> Riferimento: [https://aws.amazon.com/compliance/shared-responsibility-model/](https://aws.amazon.com/compliance/shared-responsibility-model/)

#### Acceptable Use Policy

La policy di utilizzo accettabile di AWS vieta:

- Uso illegale, dannoso o offensivo.
- Violazioni della sicurezza.
- Abuso della rete.
- Spam o abuso di email/messaggi.

> Riferimento: [https://aws.amazon.com/aup/](https://aws.amazon.com/aup/)

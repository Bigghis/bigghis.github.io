---
title: "Agentic Memory"
description: "L'agentic memory si basa sull'idea che un agente possa memorizzare informazioni e utilizzarle nel proprio processo decisionale."
date: 2026-01-08 12:00:00 +0530
categories: [Agentic Memory]
tags: [Agentic Memory, Memory Manager, Memory Engineering, Context Engineering, Vector Store, PostgreSQL, pgvector, psycopg, LangChain, Embedding, stateless, stateful]
comments: false
protected: false
mermaid: true
---

Un LLM, di per sé, è **stateless**: ogni volta che lo interroghiamo riparte da zero e ricorda soltanto ciò che gli riscriviamo nella finestra di contesto.  
Un agente, invece, deve portare avanti attività che durano ore o giorni, ricordare cosa ha già provato, imparare dagli errori e riutilizzare le conoscenze acquisite; si presuppone che operi, cioè, in contesto **stateful**.  
La differenza tra le due situazioni sta nell'utilizzo di **memoria**: l'infrastruttura persistente che permette a un agente di leggere, scrivere e recuperare informazioni durante la propria esecuzione.



In questo articolo costruiamo lo strato di memoria di un agente: vedremo come progettare i **memory store** per i diversi tipi di memoria, come modellare i dati per un recupero efficiente e come implementare un **Memory Manager** che orchestri le operazioni di lettura e scrittura.

### I tipi di memoria

Prendendo a prestito concetti in uso nelle **scienze cognitive** applicati ai sistemi di AI, possiamo individuare **quattro tipi** principali di memoria:

| Tipo | Persistenza | Cosa conserva | Caratteristica distintiva | Implementazione tipica |
|:---|:---|:---|:---|:---|
| **Short-term / working** | **breve termine** | contesto attivo di una singola inferenza | veloce e immediata, ma **volatile** | finestra di contesto (prompt, history, output dei tool, documenti recuperati) |
| **Episodic** | **lungo termine** | eventi passati specifici, interazioni, esiti | ricorda *che cosa è successo* e *quando* | record con timestamp in un vector store; retrieval semantico o ibrido |
| **Semantic** | **lungo termine** | fatti strutturati, preferenze, relazioni tra entità | ricorda *cosa è vero* nel dominio dell'agente | profili di entità, storage relazionale + vettoriale |
| **Procedural** | **lungo termine** | workflow, regole decisionali, pattern di comportamento | ricorda *come si fa* | system prompt, few-shot, regole gestite dall'agente |

**Short-term o working memory** è la finestra di contesto: tutto ciò su cui il modello può ragionare attivamente in una singola chiamata di inferenza. Include system prompt, cronologia della conversazione, output dei tool e documenti recuperati. Va pensata come la **RAM**: è veloce e immediata, ma si azzera quando la sessione termina. Di solito è implementata come buffer a scorrimento o come array dello storico conversazionale. Basta per task semplici e mono-sessione; **non sopravvive** tra una sessione e l'altra.

**Episodic memory** registra eventi passati specifici: interazioni, azioni, esiti. Quando un agente ricorda che il deploy di un utente è fallito martedì scorso per una variabile d'ambiente mancante, sta usando memoria episodica. È particolarmente efficace per il **case-based reasoning**: usare eventi, azioni e risultati passati per migliorare le decisioni future. Di solito si conserva come record con timestamp in un database vettoriale e si recupera via ricerca semantica o ibrida al momento della query.

**Semantic memory** è ciò che l'agente **sa**, indipendentemente da quando l'ha imparato. Contiene fatti stabili: le preferenze dell'utente, le nozioni del dominio in cui l'agente lavora e le relazioni tra le entità che incontra (persone, aziende, prodotti, sistemi).

La differenza con la memoria episodica sta tutta qui: l'episodica ricorda *un evento datato* ("il 12 marzo l'utente ha chiesto di accorciare le risposte"), la semantica ricorda *il fatto che ne deriva* ("questo utente preferisce risposte concise"). L'evento ha una data, il fatto no.

Un esempio: un agente di customer service che sa che il proprio interlocutore lavora nel settore legale e vuole risposte sintetiche sta usando memoria semantica. Non gli serve ricordare in quale conversazione l'ha scoperto, gli basta sapere che è vero.

**Procedural memory** codifica *come si fanno le cose*: workflow, regole decisionali e pattern comportamentali appresi. In pratica compare come istruzioni nel system prompt, esempi few-shot o set di regole gestiti dall'agente e affinati con l'esperienza. Un coding assistant che ha imparato a controllare sempre i conflitti di dipendenze prima di suggerire l'upgrade di una libreria sta esprimendo memoria procedurale.

La differenza tra i quattro tipi si riassume in una domanda diversa per ciascuno:

- short-term → *cosa ho sotto gli occhi adesso?*
- episodica → *cosa è successo in passato?*
- semantica → *cosa so (fatti, preferenze, relazioni)?*
- procedurale → *come procedo in situazioni come questa?*

> Il passaggio da breve a lungo termine non è automatico: qualcosa deve decidere **cosa promuovere** dalla working memory alla memoria persistente, e **cosa scartare**. È esattamente il compito del Memory Manager, ed è il motivo per cui più avanti distingueremo le operazioni deterministiche da quelle agent-triggered.
{: .prompt-tip }

Questi tipi **non operano in isolamento**. Gli agenti di produzione capaci hanno tipicamente bisogno di **tutti e quattro gli strati** che lavorano insieme: la working memory ospita il ragionamento corrente, mentre episodi, fatti e procedure persistenti alimentano il contesto al momento giusto.

Più avanti, quando creeremo le tabelle, vedremo nel dettaglio come queste categorie si concretizzano nel database: quale store implementa quale tipo di memoria, e perché la working memory è l'unica a non avere una tabella dedicata.

### L'agent stack

L'insieme di strumenti e tecnologie che permettono a un agente AI di funzionare in modo affidabile ed efficiente in produzione è l'**agent stack**.  
Lo stack completo è composto da parecchi livelli, ma per i nostri scopi possiamo comprimerlo in tre soli strati: **Application Layer**, **Data Layer** e **Infrastructure Layer**.

Poiché però stiamo ragionando in ottica agentica, il Data Layer si trasforma in un **Memory Layer**.  
Un Memory Layer conserva dati organizzati in modo che l'agente possa recuperarli e usarli per decidere e rappresenta l'**esperienza** accumulata dall'agente nel tempo.  

```mermaid
flowchart TB
  A["Application Layer<br/>(agente, tool, prompt, orchestrazione)"]
  M["Memory Layer<br/>(Memory Core + Memory Manager)"]
  I["Infrastructure Layer<br/>(database, modelli, compute)"]

  A --> M
  M --> I
```

### Il Memory Layer: Memory Core e Memory Manager

Il Memory Layer è composto da due elementi che lavorano in coppia:

- il **Memory Core**, cioè l'insieme dei tipi di memoria dell'agente (conversazionale, knowledge base, workflow, summary, toolbox, entity) e dei relativi store;
- il **Memory Manager**, cioè la logica che governa come si legge e si scrive su quegli store.

Insieme danno vita ad **agenti memory-augmented**, capaci di gestire attività continue, di operare su **task long-horizon** (compiti che si estendono su orizzonti temporali lunghi) e di adattarsi a nuove informazioni.

> Il Memory Core risponde alla domanda *"che cosa ricorda l'agente?"*, il Memory Manager alla domanda *"come ci accede?"*.
{: .prompt-info }

### Il Memory Manager

Il Memory Manager si può descrivere come un'**astrazione costruita sopra il database**: al suo interno vivono i flussi e la logica di controllo che regolano lettura e scrittura verso i memory store.

In pratica è una classe che espone metodi **CRUD** (create, read, update, delete) sulle tabelle che rappresentano la memoria dell'agente.  
L'agente non scrive mai SQL: chiama metodi come `write_knowledge_base()` o `read_conversation()` e il Memory Manager si occupa di tradurli in operazioni sul database.

Il vantaggio è che tutta la complessità (connessioni, embedding, strategie di retrieval, indici) resta confinata in un unico punto, invece di sparpagliarsi nel codice dell'agente.

#### I memory store e i loro requisiti di storage

Ogni tipo di memoria ha la propria tabella dedicata all'interno del database, e ogni tipo ha esigenze di storage diverse.  
La memoria conversazionale è essenzialmente cronologica e testuale: una normale tabella relazionale SQL è più che sufficiente.  
Gli altri tipi di memoria sono anch'essi in forma relazionale, ma richiedono in più una colonna di tipo **`vector`** — fornita dall'estensione **pgvector** — capace di contenere gli embedding, perché su di essi vogliamo poter fare ricerca semantica.

| Tipo di memoria | Analogia umana | A cosa serve | Storage | Strategia di retrieval |
|:---|:---|:---|:---|:---|
| **Conversational** | memoria a breve termine | cronologia della chat per thread | tabella SQL | match esatto su `thread_id` |
| **Knowledge Base** | memoria semantica a lungo termine | fatti, documenti, risultati di ricerca | vector store (colonna `vector`) | similarità semantica |
| **Workflow** | memoria procedurale | pattern di azioni appresi | vector store (colonna `vector`) | similarità semantica + filtro sui metadati |
| **Toolbox** | memoria delle competenze | tool e capacità disponibili | vector store (colonna `vector`) | similarità semantica |
| **Entity** | memoria episodica | persone, luoghi, sistemi citati | vector store (colonna `vector`) | similarità semantica |
| **Summary** | memoria compressa | contesto condensato per conversazioni lunghe | vector store (colonna `vector`) | similarità semantica (con filtro opzionale per ID) |
| **Tool Log** | traccia di audit | input/output grezzi dei tool e stato di esecuzione | tabella SQL | match esatto su `thread_id` + ordinamento temporale |

Per ognuno di questi store il Memory Manager espone almeno le operazioni di **lettura** e **scrittura**; nulla vieta di implementare anche `update`, `delete` e `create` a seconda delle necessità.

### Operazioni deterministiche e operazioni agent-triggered

Le operazioni di memoria non sono tutte uguali: si distinguono in base a **chi decide** di invocarle.

**Operazioni deterministiche.** Vengono eseguite in modo programmatico, secondo una pianificazione fissa o condizioni predefinite. Avvengono a prescindere dalla situazione: per esempio, salvare ogni messaggio della conversazione nella tabella conversazionale è un'operazione deterministica, perché succede sempre, per ogni turno, senza che nessuno debba deciderlo.

**Operazioni agent-triggered.** Le operazioni di memoria vengono fornite all'agente sotto forma di **tool**, e sarà l'agente stesso a decidere quando e dove usarle, sulla base dell'intento e della situazione. Il momento in cui la memoria viene scritta o interrogata è lasciato alla discrezione dell'agente.

Decidere quali operazioni collocare nell'una o nell'altra categoria è una delle scelte di progettazione più importanti della memory engineering. 

#### Perché conviene il recupero deterministico

Le operazioni deterministiche vengono eseguite **a ogni turno** oppure sotto **condizioni fisse ed esplicite** (ad esempio "sempre all'inizio del loop dell'agente", "sempre dopo l'esecuzione di un tool").

Il recupero della memoria viene comunemente eseguito all'inizio di ogni ciclo dell'agente per tre motivi.  
Il primo è che il **bootstrap del contesto non è negoziabile**: senza il contesto pregresso l'agente si comporta come se fosse stateless e ricomincia da capo ogni volta.  
Il secondo è più sottile: **l'agente non può decidere di cercare ciò di cui ignora l'esistenza**. Se fosse lui a dover decidere se consultare la memoria, dovrebbe indovinare che cosa c'è dentro, e si creerebbe un problema circolare — *serve la memoria per sapere di quale memoria si ha bisogno*.  
Il terzo è la **prevedibilità**: caricare sempre la memoria produce un comportamento coerente e rende il sistema più facile da valutare e da debuggare.

Lo stesso vale per la scrittura. Persistere conversazioni, workflow ed entità è spesso deterministico per **affidabilità** (non vogliamo che l'agente "si dimentichi di salvare"), per **completezza** (ogni interazione va registrata, perché i salvataggi selettivi creano buchi di contesto che più avanti rompono i task long-horizon) e per **ridurre il carico cognitivo** del modello, che deve concentrarsi sull'esecuzione del compito e non sulla contabilità della memoria.

#### Perché servono le operazioni agent-triggered

Le operazioni lasciate all'agente sono quelle che richiedono **giudizio**: "questa informazione merita di diventare una preferenza durevole?", "è il momento di consolidare o riassumere?", "mi serve un recupero più profondo rispetto al precaricamento di base?", "questa memoria va rafforzata, aggiornata, fusa o lasciata decadere?".

I vantaggi sono tre.  
La **rilevanza**: non tutto merita di essere conservato a lungo termine, e l'agente sa distinguere il segnale (preferenze, decisioni, vincoli) dal rumore.  
Il **controllo di costi e latenza**: retrieval profondo, reranking, summarization e consolidamento costano token e tempo, quindi attivarli solo quando servono riduce l'overhead.  
La **qualità della gestione della memoria**: decidere *cosa* archiviare e *come* comprimerlo richiede una comprensione semantica dell'intento, ed è proprio ciò in cui il modello è bravo.

> Anche le chiamate a tool esterni (ricerca web, lookup su database esterni, job di summarization costosi) sono tipicamente agent-triggered: solo l'agente può giudicare se servano informazioni aggiuntive, i tool introducono latenza e costi di API, e scegliere *cosa* cercare richiede di aver capito l'obiettivo dell'utente.
{: .prompt-tip }

> Le due modalità non sono alternative, ma complementari: il deterministico garantisce continuità e prevedibilità, l'agent-triggered garantisce un rapporto segnale/rumore alto e un uso selettivo delle risorse.
{: .prompt-info }

### La Memory Unit

Una **Memory Unit** è la più piccola unità atomica di rappresentazione dell'informazione conservata in un database e utilizzata all'interno di un sistema agentico.  
In termini pratici corrisponde quasi sempre a una **riga di una tabella**.

Una **Conversational Memory Unit** contiene il timestamp, il ruolo dell'entità che sta conversando e il contenuto della conversazione stessa:

| `timestamp` | `role` | `content` |
|:---|:---|:---|
| `2026-01-08 10:14:02` | `user` | "Trova i paper recenti sull'esplorazione spaziale" |
| `2026-01-08 10:14:09` | `assistant` | "Ho trovato tre paper pertinenti..." |

Una **Workflow Memory Unit** è più ricca: contiene il contenuto del workflow, il suo tipo, il timestamp e una **rappresentazione vettoriale** di parte del contenuto.  
Il contenuto di una workflow memory unit è tipicamente costituito dai **passi eseguiti e dal loro esito**, cioè la traccia di come l'agente ha portato a termine (o mancato) un obiettivo.

| `content` | `workflow_type` | `timestamp` | `embedding` |
|:---|:---|:---|:---|
| passi eseguiti + esito | tipo di workflow | data e ora | vettore del contenuto |

### Context Engineering

Il **context engineering** è la pratica di **curare in modo selettivo il contenuto che passiamo nella finestra di contesto**.

Abbiamo a disposizione molte sorgenti dati, ciascuna capace di fornire una gran quantità di informazioni. La tentazione è di riversarle tutte nel contesto, ma è esattamente l'errore da evitare: dobbiamo invece ragionare con attenzione su *quale* contesto passare.

L'obiettivo è **massimizzare il valore di ogni singolo token** presente nella finestra di contesto.  
Idealmente vogliamo un **rapporto segnale/rumore alto** per ogni token: è così che si ottiene l'output e il risultato desiderati.

> Riempire il contesto con tutto ciò che abbiamo non rende l'agente più informato, lo rende più confuso.  
> Ogni token irrilevante è un token che diluisce l'attenzione del modello e costa denaro.
{: .prompt-warning }

### Memory Engineering

La **memory engineering** è la disciplina che si occupa di **costruire e mantenere i sistemi di memoria** di un agente AI, in modo che questo possa adattarsi e imparare davvero.  
Chi fa memory engineering è responsabile di tutti i processi e le operazioni che avvengono lungo il **memory lifecycle**.

#### Il memory lifecycle

Il ciclo di vita della memoria attraversa diverse fasi.  
Si parte da una **sorgente di dati grezzi**, che passa attraverso una **pipeline di ingestion** e viene poi **arricchita**, ad esempio con un modello di embedding oppure con un LLM che ne aumenta il contenuto informativo.  
Il risultato viene **archiviato** nel database, in tabelle distinte che rappresentano memoria a breve o a lungo termine.

Segue l'**organizzazione** dell'informazione, che comprende l'indicizzazione e la mappatura delle relazioni tra le informazioni.  
Si arriva quindi al **recupero**, che può avvenire con diverse strategie di retrieval:

- **testuale o lessicale**, basata sulla corrispondenza delle parole;
- **vettoriale**, basata sulla similarità semantica degli embedding;
- **graph traversal**, che naviga le relazioni tra le entità;
- **ibrida**, cioè la combinazione di due o più strategie.

L'informazione recuperata — che è memoria — viene passata all'LLM. E qui sta il punto interessante: **anche l'output dell'LLM può diventare memoria**. Attraversa fasi di **serializzazione e augmentation**, in cui viene arricchito per essere archiviato nel database, e rientra così nel ciclo di storage, organizzazione e retrieval, per tornare nuovamente all'LLM.

```mermaid
flowchart LR
  R["Raw data source"] --> IN["Ingestion"]
  IN --> EN["Enrichment<br/>(embedding / LLM)"]
  EN --> ST["Storage<br/>(short & long term)"]
  ST --> OR["Organization<br/>(indexing, relazioni)"]
  OR --> RE["Retrieval<br/>(lexical, vector, graph, hybrid)"]
  RE --> LLM["LLM"]
  LLM --> SE["Serialization<br/>& augmentation"]
  SE --> ST
```

È proprio questo ciclo a rendere possibile l'**apprendimento continuo** di cui un agente ha bisogno per affrontare i task long-horizon.

#### Le discipline che compongono la memory engineering

"Memory engineering" può sembrare un termine nuovo, ma non lo è affatto: è la **combinazione di discipline già esistenti**, di cui riprende pratiche e principi per implementare in modo efficiente le operazioni di memoria negli agenti AI.

| Disciplina | Cosa apporta alla memory engineering |
|:---|:---|
| **Database engineering** | transazioni ACID, storage persistente, comprensione delle architetture di storage |
| **Agent engineering** | come progettare l'agente e dove collocare le operazioni di memoria |
| **Machine learning engineering** | fine-tuning dei modelli di embedding o di small language model, versionamento dei modelli, pipeline di reranking, continual learning |
| **Information retrieval** | implementazione e ottimizzazione delle strategie di retrieval, indici vettoriali e altre strategie di indicizzazione |

> Non c'è nulla di realmente nuovo nella memory engineering: è l'intersezione di discipline che conosciamo già, applicata a un problema nuovo.
{: .prompt-info }

### Da agenti memory-augmented ad agenti memory-aware

Questo è il passaggio concettuale più importante, e conviene capire bene la differenza.

Si parte da un'implementazione **naive** di agente memory-augmented, dotato della sola memoria conversazionale: in pratica ha soltanto lo storico delle interazioni.  
Introducendo un'**allocazione esplicita dei tipi di memoria** si arriva a un agente **pienamente memory-augmented**, capace di recuperare informazioni da store diversi — conversazionale, workflow, toolbox e le altre forme di memoria presenti nel database.

Ma possiamo spingerci oltre e rendere l'agente **memory-aware**, cioè *consapevole* della propria memoria. Servono quattro passi:

1. **Dare all'agente consapevolezza dei memory store tramite il system prompt**, così che sappia quali memorie possiede e a cosa servono.
2. **Fornire le operazioni di memoria come tool**, in modo che l'agente possa archiviare, recuperare, leggere e *dimenticare* a propria discrezione.
3. **Dare all'agente la capacità di ragionare lungo il memory lifecycle**, non solo di eseguirne i passaggi.
4. **Segmentare la finestra di contesto** in porzioni allocate a tipi di memoria specifici.

```mermaid
flowchart LR
  N["Agente naive<br/>(solo memoria conversazionale)"] --> A["Memory-augmented<br/>(tipi di memoria espliciti)"]
  A --> W["Memory-aware<br/>(prompt + tool + reasoning + contesto segmentato)"]
```

La differenza è sostanziale: un agente memory-augmented **ha** una memoria, un agente memory-aware **sa di averla** e sa come usarla.

### L'implementazione

Vediamo ora come questi concetti prendono forma nel codice.  
Useremo **PostgreSQL** con l'estensione **pgvector** come storage, un modello di embedding di Hugging Face e l'integrazione ufficiale **`langchain-postgres`** (`PGEngine` + `PGVectorStore`) per i vector store.

Il caso d'uso che ci accompagna è un **assistente di ricerca agentico** (lo chiameremo *ArxivScout*) che aiuta l'utente a investigare argomenti complessi lungo più sessioni. L'assistente deve ricordare le scoperte precedenti, l'affidabilità delle fonti e le preferenze dell'utente, così da fornire risposte coerenti e contestualizzate senza dover rifare ogni volta lo stesso lavoro di ricerca.

#### Setup dell'ambiente

Per lo sviluppo locale possiamo avviare PostgreSQL già con pgvector abilitato tramite l'immagine Docker ufficiale:

```bash
docker run --name pgvector-container \
  -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=agentic_memory -p 6024:5432 \
  -d pgvector/pgvector:pg16
```

Le dipendenze Python necessarie sono:

```text
langchain-postgres
langchain-huggingface
sentence-transformers
psycopg[binary]
datasets
```

#### 1. Connessione al database

A differenza di un'integrazione in cui una sola connessione serve sia le tabelle SQL sia i vector store, con `langchain-postgres` servono **due oggetti distinti**:

- una **connessione raw** (`psycopg`) per le tabelle SQL (conversazionale e tool log);
- un **`PGEngine`**, cioè un pool di connessioni SQLAlchemy, usato da `PGVectorStore`.

Il setup una tantum si riduce ad abilitare l'estensione pgvector:

```python
import psycopg
from langchain_postgres import PGEngine

CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@localhost:6024/agentic_memory"

# Connessione raw per le tabelle SQL (conversazionale e tool log)
database_connection = psycopg.connect(
    "postgresql://langchain:langchain@localhost:6024/agentic_memory"
)
with database_connection.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
database_connection.commit()

# Pool di connessioni per i vector store
pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

print("Using user:", database_connection.info.user)
```

Stampare l'utente collegato è il modo più rapido per confermare che la connessione sia realmente attiva. Con `psycopg` l'attributo corretto è `database_connection.info.user`.

```mermaid
flowchart TB
  PC["database_connection<br/>(psycopg)"]
  PE["pg_engine<br/>(PGEngine, pool SQLAlchemy)"]
  PC --> PSQL["Tabelle SQL<br/>conversational, tool_log"]
  PE --> PINIT["init_vectorstore_table()"]
  PINIT --> PVS["PGVectorStore.create_sync()"]
```

#### 2. Il modello di embedding

Il secondo componente chiave è il modello di embedding, che useremo per trasformare il testo in vettori.  
Lo preleviamo da Hugging Face attraverso l'integrazione LangChain, usando la libreria `sentence-transformers` e il modello `paraphrase-mpnet-base-v2`.

Con `PGVectorStore` la dimensione del vettore va dichiarata **prima** di creare le tabelle: `init_vectorstore_table()` tipizza la colonna come `vector(768)`, e non può dedurla al primo inserimento.

```python
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_SIZE = 768  # dimensione di paraphrase-mpnet-base-v2

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2"
)
```

Al termine dell'esecuzione il modello risiede sulla macchina locale ed è pronto a produrre embedding.

#### 3. Le tabelle dei memory store

Definiamo i nomi delle tabelle che rappresentano le diverse forme di memoria dell'agente. Li raccogliamo in una lista, così da poterli iterare comodamente.

In PostgreSQL gli identificatori non quotati vengono **normalizzati a minuscolo**: usiamo quindi nomi in minuscolo fin da subito.

```python
# Nomi delle tabelle per ciascun tipo di memoria
CONVERSATIONAL_TABLE   = "conversational_memory"  # Episodic memory
KNOWLEDGE_BASE_TABLE   = "semantic_memory"        # Semantic memory
WORKFLOW_TABLE         = "workflow_memory"        # Procedural memory
TOOLBOX_TABLE          = "toolbox_memory"         # Procedural memory
ENTITY_TABLE           = "entity_memory"          # Semantic memory
SUMMARY_TABLE          = "summary_memory"         # Semantic memory
TOOL_LOG_TABLE         = "tool_log_memory"        # Tool execution logs

ALL_TABLES = [
    CONVERSATIONAL_TABLE,
    KNOWLEDGE_BASE_TABLE,
    WORKFLOW_TABLE,
    TOOLBOX_TABLE,
    ENTITY_TABLE,
    SUMMARY_TABLE,
    TOOL_LOG_TABLE,
]

# Eliminiamo le tabelle esistenti per ripartire da zero
for table in ALL_TABLES:
    with database_connection.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        print(f"  - {table} (dropped if existed)")

database_connection.commit()
```

`DROP TABLE IF EXISTS ... CASCADE` rende superflua la gestione delle eccezioni: se la tabella non c'è, il comando è semplicemente un no-op. `CASCADE` elimina anche indici e dipendenze collegate.

##### Dalle categorie cognitive alle tabelle

I commenti a fianco dei nomi non sono decorativi: indicano la **categoria cognitiva** che ogni store implementa. È qui che i quattro tipi di memoria visti all'inizio dell'articolo diventano oggetti concreti nel database.

| Categoria cognitiva | Tabelle | Perché sta lì |
|:---|:---|:---|
| **Short-term / working** | *nessuna tabella* | vive nella finestra di contesto, non viene persistita |
| **Episodica** | `conversational_memory` | eventi datati: chi ha detto cosa e quando, turno per turno |
| **Semantica** | `semantic_memory`, `entity_memory`, `summary_memory` | fatti, entità e conoscenza condensata, indipendenti dal momento in cui sono stati appresi |
| **Procedurale** | `workflow_memory`, `toolbox_memory` | come si fanno le cose: sequenze di azioni apprese e strumenti disponibili |
| *(nessuna: audit)* | `tool_log_memory` | traccia tecnica delle esecuzioni, serve a debug e osservabilità, non al ragionamento |

Tre osservazioni utili per orientarsi.

**La working memory non ha una tabella.** È l'unico tipo a breve termine: vive nella finestra di contesto e sparisce a fine sessione. Ciò che persistiamo è la sua *traccia*, cioè la cronologia della conversazione, che una volta scritta su disco diventa a tutti gli effetti memoria **episodica** — ed è per questo che `conversational_memory` è commentata come *Episodic memory* pur contenendo i messaggi della chat.

**Il nome del tipo di memoria e il nome della tabella non coincidono sempre.** La knowledge base è memorizzata nella tabella `semantic_memory`, non in una ipotetica `knowledge_base_memory`. È il caso più insidioso quando si leggono le query, quindi conviene fissarlo subito: `KNOWLEDGE_BASE_TABLE = "semantic_memory"`.

**Una stessa categoria può avere più tabelle.** La memoria semantica è divisa in tre store perché granularità e ciclo di vita sono diversi: `semantic_memory` contiene documenti di dominio ingeriti dall'esterno, `entity_memory` profili di entità aggiornati incrementalmente, `summary_memory` sintesi generate dall'agente per comprimere conversazioni lunghe. Stessa natura cognitiva, ma pattern di scrittura e retrieval differenti — e quindi tabelle separate.

> Il `tool_log_memory` è l'unico store che non corrisponde ad alcuna categoria cognitiva: non è memoria di cui l'agente si serve per ragionare, ma una traccia di audit per noi che sviluppiamo il sistema.
{: .prompt-info }

#### 4. La tabella della memoria conversazionale

Creiamo ora la tabella dello storico conversazionale. La funzione riceve la connessione e il nome della tabella, elimina un'eventuale tabella preesistente e crea la nuova struttura.

A differenza dei vector store, la memoria conversazionale usa una tabella tradizionale perché qui ci serve un **recupero esatto per thread**, non una ricerca per similarità.

Per una conversational memory unit vogliamo catturare il **contenuto**, il **ruolo** e il **timestamp**, come visto in precedenza. Possiamo però aggiungere metadati ulteriori: il campo `metadata` associato alla memory unit (qui in **JSONB**, interrogabile e indicizzabile), il campo `created_at` (che è cosa diversa dal timestamp in cui la memory unit viene catturata) e un `summary_id`, che useremo più avanti per collegare la conversazione ai suoi riassunti.

```python
def create_conversational_history_table(conn, table_name: str = "conversational_memory"):
    """
    Create a table to store conversational history.

    Args:
        conn: PostgreSQL database connection (psycopg)
        table_name: Name of the table to create
    """
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                thread_id VARCHAR(100) NOT NULL,
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                summary_id UUID DEFAULT NULL
            )
        """)

        # Create index on thread_id for faster lookups
        cur.execute(f"""
            CREATE INDEX idx_{table_name}_thread_id ON {table_name}(thread_id)
        """)

        # Create index on timestamp for ordering
        cur.execute(f"""
            CREATE INDEX idx_{table_name}_timestamp ON {table_name}(timestamp)
        """)

    conn.commit()
    print(f"Table {table_name} created successfully with indexes")
    return table_name
```

Gli indici non sono un dettaglio: quello su `thread_id` rende veloce il recupero di una conversazione, quello su `timestamp` rende efficiente l'ordinamento cronologico. Insieme garantiscono che la scansione delle righe resti rapida anche quando lo storico cresce.

Per chi arriva da dialetti SQL più "enterprise", ecco il mapping dei tipi usato nel DDL:

| Concetto | Tipico in dialetti Oracle-like | PostgreSQL |
|:---|:---|:---|
| stringa a lunghezza fissa | `VARCHAR2(n)` | `VARCHAR(n)` |
| testo lungo | `CLOB` | `TEXT` |
| identificatore generato | `DEFAULT SYS_GUID()` | `UUID DEFAULT gen_random_uuid()` |
| metadati flessibili | `CLOB` / JSON grezzo | `JSONB` (interrogabile e indicizzabile) |
| timestamp con timezone | `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` | `TIMESTAMPTZ DEFAULT NOW()` |
| drop sicuro | `DROP TABLE ... PURGE` + gestione errori | `DROP TABLE IF EXISTS ... CASCADE` |

Con lo stesso approccio creiamo la **tool log table**, che registra input, output e stato di esecuzione dei tool. La invochiamo insieme alla precedente:

```python
from helper import create_tool_log_table

# Creiamo le tabelle SQL della memoria
CONVERSATION_HISTORY_TABLE = create_conversational_history_table(database_connection, CONVERSATIONAL_TABLE)
TOOL_LOG_HISTORY_TABLE = create_tool_log_table(database_connection, TOOL_LOG_TABLE)
```

> **Attenzione a `helper.py`.** Se `create_tool_log_table` e `MemoryManager` sono stati scritti per Oracle, vanno adattati a PostgreSQL: i bind parameter passano da `:1` / `:nome` a `%s` / `%(nome)s`, e `FETCH FIRST n ROWS ONLY` diventa `LIMIT n`. Il DDL della tool log table segue gli stessi tipi visti sopra (`TEXT`, `JSONB`, `TIMESTAMPTZ`, `UUID`).
{: .prompt-warning }

#### 5. I vector store

Le tabelle SQL per memoria conversazionale e tool log sono pronte. Ora servono le tabelle capaci di gestire **dati vettoriali**.

Creiamo cinque vector store distinti, uno per ogni tipo di memoria. Ciascuno è appoggiato alla propria tabella PostgreSQL (con colonna `vector`) e usa lo **stesso modello di embedding**, per garantire la coerenza tra gli spazi vettoriali.

Usiamo l'integrazione ufficiale `langchain-postgres`: `PGEngine` gestisce il pool, `init_vectorstore_table()` crea lo schema con la colonna `vector(N)`, e `PGVectorStore.create_sync()` istanzia lo store. Importiamo anche la `DistanceStrategy` (qui `COSINE_DISTANCE`) e la `HybridSearchConfig` per l'**hybrid search**.

> Differenza importante rispetto ad alcune integrazioni più "magiche": la tabella vettoriale **non** viene creata nel costruttore dello store. Va creata esplicitamente con `init_vectorstore_table()` *prima* di chiamare `PGVectorStore.create_sync()`.
{: .prompt-info }

Per tenere ordinata la creazione dei vector store definiamo una classe che ne astrae metodi e istanze: la chiamiamo **`StoreManager`**.

```python
from langchain_postgres import PGVectorStore, Column
from langchain_postgres.v2.indexes import DistanceStrategy
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    weighted_sum_ranking,
)


class StoreManager:
    """Manages all stores (vector stores and SQL tables) with getter methods for easy access."""

    def __init__(
        self,
        engine,
        embedding_function,
        table_names,
        distance_strategy,
        conversational_table,
        vector_size: int = 768,
        tool_log_table: str | None = None,
    ):
        """
        Initialize all stores.

        Args:
            engine: PGEngine connection pool
            embedding_function: Embedding model to use
            table_names: Dict with keys: knowledge_base, workflow, toolbox, entity, summary
            distance_strategy: Distance strategy for vector search
            conversational_table: Name of the conversational history SQL table
            vector_size: Embedding dimension (must match the model)
            tool_log_table: Name of the SQL tool log table
        """
        self.engine = engine
        self.embedding_function = embedding_function
        self.distance_strategy = distance_strategy
        self.vector_size = vector_size
        self._conversational_table = conversational_table
        self._tool_log_table = tool_log_table

        # Hybrid search richiede colonna tsvector + indice GIN già in fase di creazione tabella
        hybrid_config = HybridSearchConfig(
            tsv_column="content_tsv",
            tsv_lang="pg_catalog.english",
            fusion_function=weighted_sum_ranking,
            primary_top_k=5,
            secondary_top_k=5,
            index_name="kb_tsv_index",
            index_type="GIN",
        )

        # Metadati tipizzati e filtrabili (solo knowledge base)
        kb_metadata_columns = [
            Column("arxiv_id", "TEXT"),
            Column("subjects", "TEXT"),
            Column("submission_date", "TEXT"),
        ]

        self._knowledge_base_vs = self._init_store(
            table_names["knowledge_base"],
            metadata_columns=kb_metadata_columns,
            hybrid_search_config=hybrid_config,
        )
        self._workflow_vs = self._init_store(table_names["workflow"])
        self._toolbox_vs = self._init_store(table_names["toolbox"])
        self._entity_vs = self._init_store(table_names["entity"])
        self._summary_vs = self._init_store(table_names["summary"])

    def _init_store(self, table_name, metadata_columns=None, hybrid_search_config=None):
        # 1. crea la tabella con la colonna vector(768)
        self.engine.init_vectorstore_table(
            table_name=table_name,
            vector_size=self.vector_size,
            metadata_columns=metadata_columns or [],
            overwrite_existing=True,
            hybrid_search_config=hybrid_search_config,
        )
        # 2. istanzia lo store sulla tabella appena creata
        return PGVectorStore.create_sync(
            engine=self.engine,
            embedding_service=self.embedding_function,
            table_name=table_name,
            metadata_columns=[c.name for c in (metadata_columns or [])],
            distance_strategy=self.distance_strategy,
            hybrid_search_config=hybrid_search_config,
        )

    def get_knowledge_base_store(self):
        """Return the knowledge base vector store."""
        return self._knowledge_base_vs

    def get_workflow_store(self):
        """Return the workflow vector store."""
        return self._workflow_vs

    def get_toolbox_store(self):
        """Return the toolbox vector store."""
        return self._toolbox_vs

    def get_entity_store(self):
        """Return the entity vector store."""
        return self._entity_vs

    def get_summary_store(self):
        """Return the summary vector store."""
        return self._summary_vs

    def get_conversational_table(self):
        """Return the conversational history table name."""
        return self._conversational_table

    def get_tool_log_table(self):
        """Return the tool log table name."""
        return self._tool_log_table
```

Ogni store nasce in **due passi**: prima `init_vectorstore_table()` crea la tabella tipizzata (`vector(768)`), poi `PGVectorStore.create_sync()` collega lo store a quella tabella. I parametri chiave sono: l'**engine** (`PGEngine`); l'**embedding service**, cioè il modello inizializzato prima; il **nome della tabella**; la **distance strategy**, qui `DistanceStrategy.COSINE_DISTANCE` (operatore pgvector `<=>`).

Ripetiamo il procedimento per ogni forma di memoria: knowledge base (memoria semantica), workflow, toolbox, entity e summary.  
Sulla sola knowledge base dichiariamo colonne di metadati **tipizzate e filtrabili** (`arxiv_id`, `subjects`, `submission_date`) e passiamo la `HybridSearchConfig` già in fase di creazione.

> L'hybrid search in Postgres non si "accende" a posteriori su uno store già esistente: richiede una colonna `tsvector` e un indice **GIN** che devono esistere al momento della creazione della tabella. Per questo la configurazione viene passata a `init_vectorstore_table()` e a `create_sync()`, invece di un metodo `setup_hybrid_search()` chiamato dopo.
{: .prompt-tip }

Lo `StoreManager` non gestisce solo i vector store: conserva anche i nomi delle tabelle SQL (conversazionale e tool log), così da diventare l'unico punto di accesso a tutti gli store del sistema.

Creiamo l'istanza:

```python
# Create StoreManager instance
store_manager = StoreManager(
    engine=pg_engine,
    embedding_function=embedding_model,
    table_names={
        "knowledge_base": KNOWLEDGE_BASE_TABLE,
        "workflow": WORKFLOW_TABLE,
        "toolbox": TOOLBOX_TABLE,
        "entity": ENTITY_TABLE,
        "summary": SUMMARY_TABLE,
    },
    distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    conversational_table=CONVERSATION_HISTORY_TABLE,
    vector_size=VECTOR_SIZE,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)
```

E recuperiamo tutti gli store attraverso i getter del manager:

```python
# Get all stores via the manager
conversation_table = store_manager.get_conversational_table()
knowledge_base_vs = store_manager.get_knowledge_base_store()
workflow_vs = store_manager.get_workflow_store()
toolbox_vs = store_manager.get_toolbox_store()
entity_vs = store_manager.get_entity_store()
summary_vs = store_manager.get_summary_store()
tool_log_table = store_manager.get_tool_log_table()
```

#### 6. Gli indici vettoriali

Per garantire un recupero efficiente delle informazioni bisogna **sempre creare un indice**.  
Un indice è una struttura dati che consente di recuperare informazioni da un database **senza doverne scandire tutti gli elementi**.

Con `PGVectorStore` gli indici si creano direttamente sullo store, senza helper esterni:

```python
from langchain_postgres.v2.indexes import HNSWIndex

print("Creating vector indexes...")
knowledge_base_vs.apply_vector_index(
    HNSWIndex(name="knowledge_base_hnsw", m=16, ef_construction=64)
)
workflow_vs.apply_vector_index(HNSWIndex(name="workflow_hnsw"))
toolbox_vs.apply_vector_index(HNSWIndex(name="toolbox_hnsw"))
entity_vs.apply_vector_index(HNSWIndex(name="entity_hnsw"))
summary_vs.apply_vector_index(HNSWIndex(name="summary_hnsw"))
print("All indexes created!")
```

Nel caso degli indici vettoriali di pgvector le strutture tipiche sono due: **IVFFlat** (*Inverted File Flat*), che partiziona lo spazio vettoriale in cluster e cerca solo nei più promettenti, e **HNSW** (*Hierarchical Navigable Small World*), che accelera la ricerca per similarità con una traversata a grafo dei vicini più prossimi.

Con pgvector l'indice funziona solo se **operatore**, **operator class** e `distance_strategy` coincidono:

| Distance strategy | Operatore | Operator class |
|:---|:---|:---|
| `COSINE_DISTANCE` | `<=>` | `vector_cosine_ops` |
| `EUCLIDEAN` | `<->` | `vector_l2_ops` |
| `INNER_PRODUCT` | `<#>` | `vector_ip_ops` |

Se l'indice è costruito con `vector_cosine_ops` ma le query usano un'altra metrica, Postgres farà una scansione sequenziale e l'indice resterà inutilizzato.

A query time si può sintonizzare il trade-off recall/latenza con `index_query_options`, ad esempio `HNSWQueryOptions(ef_search=40)` oppure `IVFFlatQueryOptions(probes=10)`, passati a `PGVectorStore.create_sync(...)`.

> **Attenzione a IVFFlat su tabella vuota.** I centroidi di IVFFlat si calcolano sui dati presenti al momento della creazione dell'indice: costruirlo prima dell'ingestion (come nel flusso di questo post) lo rende inutile. Per questo qui scegliamo **HNSW**, che si aggiorna man mano che arrivano i vettori. Se preferite IVFFlat, spostate `apply_vector_index` *dopo* la scrittura dei paper e poi, se necessario, chiamate `reindex()`.
{: .prompt-warning }

#### 7. Il Memory Manager

Arrivati a questo punto possiamo istanziare il Memory Manager, che — ricordiamolo — astrae tutte le operazioni con cui leggiamo e scriviamo informazioni nel database, nascondendo la complessità delle query SQL e delle operazioni sui vector store dietro un'interfaccia uniforme.

È una singola classe che gestisce sette tipi di memoria con lo stesso pattern read/write:

| Tipo di memoria | Storage | Metodo di scrittura | Metodo di lettura |
|:---|:---|:---|:---|
| **Conversational** | tabella SQL | `write_conversational_memory()` | `read_conversational_memory()` |
| **Knowledge Base** | vector store | `write_knowledge_base()` | `read_knowledge_base()` |
| **Workflow** | vector store | `write_workflow()` | `read_workflow()` |
| **Toolbox** | vector store | `write_toolbox()` | `read_toolbox()` |
| **Entity** | vector store | `write_entity()` | `read_entity()` |
| **Summary** | vector store | `write_summary()` | `read_summary_memory()`, `read_summary_context()` |
| **Tool Log** | tabella SQL | `write_tool_log()` | `read_tool_logs()` |

Riceve la connessione `psycopg` e deve conoscere tutti gli store a cui gli serve accedere: la tabella SQL per la memoria conversazionale, i `PGVectorStore` per le altre e la tabella dei tool log. La firma pubblica resta la stessa: cambia solo il tipo concreto di `conn`.

```python
from helper import MemoryManager

# Initialize the MemoryManager instance
# Note: Uses SQL table for conversational memory, vector stores for others
memory_manager = MemoryManager(
    conn=database_connection,
    conversation_table=CONVERSATION_HISTORY_TABLE,
    knowledge_base_vs=knowledge_base_vs,
    workflow_vs=workflow_vs,
    toolbox_vs=toolbox_vs,
    entity_vs=entity_vs,
    summary_vs=summary_vs,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)
```

> È proprio questo il vantaggio dell'astrazione: l'agente continua a chiamare `write_knowledge_base()` / `read_knowledge_base()` senza sapere se sotto c'è Oracle, Postgres o un altro backend. A cambiare è l'implementazione di `helper.py` e lo strato di store, non il contratto del Memory Manager.
{: .prompt-info }

#### 8. Scrittura nella knowledge base

Il modo migliore per capire il Memory Manager è usarlo, cioè scrivergli dentro dei dati e poi rileggerli: sono esattamente le operazioni di read e write di cui parlavamo.

Popoliamo la knowledge base di *ArxivScout* con dei **paper di arXiv** presi da Hugging Face. Usiamo il dataset `nick007x/arxiv-papers` in modalità **streaming**, così da non doverlo scaricare tutto per poi usarne solo una parte.

```python
from datasets import load_dataset
from itertools import islice

ds = load_dataset("nick007x/arxiv-papers", split="train", streaming=True)
```

Per ogni paper estraiamo i campi chiave (titolo, subjects, abstract, data di pubblicazione e arXiv ID), concateniamo titolo, subjects e abstract in un unico testo ricercabile e scriviamo il tutto nella knowledge base, cioè nella **memoria semantica** dell'agente. I campi estratti vengono conservati anche come **metadati**, utili per il filtraggio e per l'attribuzione della fonte.

```python
for paper in islice(ds, 100):
    # extract the key fields
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    subjects = (paper.get("subjects") or paper.get("primary_subject") or "").strip()
    submission_date = (paper.get("submission_date") or "").strip()

    # skip empty records
    if not (title or abstract or subjects):
        continue

    # concatenate the key fields containing context for semantic search
    text = "\n".join([part for part in (title, subjects, abstract) if part])

    memory_manager.write_knowledge_base(
        text=text,
        metadata={
            "arxiv_id": paper.get("arxiv_id"),
            "title": title,
            "subjects": subjects,
            "abstract": abstract,
            "submission_date": submission_date,
        },
    )
```

La scrittura nella knowledge base esegue due operazioni: crea la **rappresentazione vettoriale** del testo, che è ciò che rende possibile la ricerca semantica sulle righe della tabella, e conserva i **metadati** nella stessa riga.  
Ogni riga contiene quindi sia i metadati sia il vettore corrispondente al contenuto.

Con lo schema che abbiamo definito in `StoreManager`, `arxiv_id`, `subjects` e `submission_date` finiscono in **colonne tipizzate filtrabili**; il resto (ad esempio `title` e `abstract` come attributi di metadata) confluisce nella colonna JSONB `langchain_metadata` gestita da `PGVectorStore`.

> Si noti la scelta di cosa mandare nell'embedding: titolo, subjects e abstract finiscono nel testo vettorizzato perché sono ciò su cui vogliamo fare ricerca semantica, mentre `arxiv_id` e `submission_date` restano nei metadati (colonne tipizzate), perché servono a filtrare e citare, non a definire il significato del documento. È una decisione di **modellazione della memory unit**, e influenza direttamente la qualità del retrieval.
{: .prompt-tip }

#### 9. Lettura dalla knowledge base

Per completare il quadro, leggiamo dalla memoria knowledge base con `read_knowledge_base`, cercando le righe semanticamente simili alla nostra query.

```python
results = memory_manager.read_knowledge_base(query="space exploration")
print(results)
```

Poiché ogni riga della tabella di memoria semantica contiene la rappresentazione vettoriale del testo, ci aspettiamo che le righe restituite siano **semanticamente vicine** all'espressione "space exploration", anche quando non contengono letteralmente quelle parole.

L'output non è però un semplice elenco di passaggi. Contiene anche l'indicazione del **tipo di memoria** da cui stiamo leggendo, una **descrizione** di che cosa contiene quella memoria e di come vada interrogata, e le **istruzioni su come utilizzarne le informazioni**. La struttura è di questo tipo:

```text
MEMORY TYPE: Knowledge Base Memory (semantic memory)

DESCRIPTION: Contains domain knowledge ingested from external sources.
Query this memory with natural language questions about the domain.

USAGE: Use the retrieved passages as factual grounding. Cite the source
metadata when relevant. Do not invent information that is not present here.

RETRIEVED PASSAGES:
[1] <title> / <subjects> / <abstract>   (arxiv_id, submission_date)
[2] <title> / <subjects> / <abstract>   (arxiv_id, submission_date)
[3] <title> / <subjects> / <abstract>   (arxiv_id, submission_date)
```

Questi metadati descrittivi non sono per noi: sono **per l'LLM**.  
Stiamo costruendo agenti memory-aware, e questo significa che l'agente deve essere consapevole dei tipi di memoria che possiede e di come usarli. Descrizione e istruzioni d'uso restituite insieme ai dati sono proprio il meccanismo con cui gli forniamo questa consapevolezza.

> Questo è il punto in cui i concetti teorici si chiudono sul codice: la stessa lettura restituisce **il dato** (i passaggi recuperati) e **il contesto d'uso del dato** (che memoria è, come si interroga, come va usata). È context engineering applicato al retrieval.
{: .prompt-tip }

### Conclusioni

Abbiamo costruito l'infrastruttura di memoria che sta sotto a qualsiasi agente serio: memory store persistenti e differenziati per tipo di memoria, dati modellati per un recupero efficiente e un Memory Manager che orchestra lettura e scrittura durante l'esecuzione.

I concetti da portarsi dietro sono pochi ma decisivi:

- il **Memory Layer** sostituisce il Data Layer nell'agent stack e contiene Memory Core e Memory Manager;
- il **Memory Manager** è un'astrazione CRUD sopra il database, e isola l'agente dai dettagli di storage;
- le operazioni di memoria possono essere **deterministiche** o **agent-triggered**, e le due modalità vanno combinate;
- la **memory unit** è l'unità atomica di memoria, tipicamente una riga di tabella;
- il **context engineering** massimizza il valore di ogni token, privilegiando il rapporto segnale/rumore;
- la **memory engineering** governa l'intero memory lifecycle ed è l'intersezione di discipline consolidate;
- un agente **memory-aware** non si limita ad avere memoria: sa quali memorie possiede, sa interrogarle da solo e ragiona sul proprio ciclo di vita della memoria.

Resta infine il lavoro meno visibile ma decisivo: **valutare i compromessi** di progettazione. Ogni scelta — quali campi vettorizzare, quale distance strategy adottare, se usare **HNSW** o **IVFFlat**, quante operazioni rendere deterministiche — sposta l'equilibrio tra accuratezza del retrieval, latenza, costo e affidabilità dell'agente. Non esiste una configurazione ottimale in assoluto: esiste quella giusta per il caso d'uso che si sta costruendo.

---
title: "Goal–Plan–Action (GPA) di un sistema multi-agent"
description: "I fallimenti critici di un agente emergono alle intersezioni tra Goal, Plan e Action, non basta solo valutare groundness e context relevance (RAG Triad)"
date: 2026-08-18 12:00:00 +0200
categories: [LangGraph, MLflow, Evaluation]
tags: [Data Agent, GPA, TruLens, MLflow, Tracing, Plan Quality, Plan Adherence, Execution Efficiency, Logical Consistency]
comments: false
protected: false
mermaid: true
---

Nel [post precedente]({% post_url 2026-17-08-DATA-AGENT-EVAL %}) abbiamo misurato Context Relevance, Groundedness e Answer Relevance. Qui guardiamo **perché** il Data Agent ha scelto un certo percorso — non solo se la risposta era grounded.  

Nel caso specifico del sistema Data Agents, la RAG Triad valuta il lavoro di ricerca e sintesi prodotto dagli agenti, ma non dice se il piano è adatto all'obiettivo, se l'agente lo ha seguito, o se il percorso di esecuzione era efficiente.

Per quello serve l'allineamento **Goal–Plan–Action** ([GPA](https://arxiv.org/abs/2510.08847){:target="_blank"}): un framework di valutazione in cui i fallimenti critici di un agente emergono alle intersezioni tra obiettivo, piano e azioni eseguite. GPA non giudica solo la risposta finale: suddivide il comportamento in tre oggetti (goal, plan, act) e misura se sono allineati valutando se il piano è adeguato al goal, se le azioni sono fedeli al piano, se il percorso è efficiente e se il ragionamento è coerente.  

La richiesta dell'utente (`user_query`) diventa quindi l'obiettivo (**Goal**) del sistema multi-agent. Il Planner decompone l'obiettivo in step che compongono il piano (**Plan**) e l'Executor esegue questi step, chiamando i tool e il researcher per recuperare il contesto. Il risultato dell'esecuzione è l'azione (**Action**) dell'agente.  

In sintesi:

- **Goal**: ciò che l'utente vuole ottenere: la `user_query` es.: "cerca i 5 istituti bancari più grandi in Italia"
- **Plan**: come l'agente *intende* arrivarci: gli step del Planner nello stato (`{agent, action}` numerati)
- **Act**: ciò che l'agente *fa davvero*: la sequenza di nodi e tool nella trace (Executor, researcher, chart, synthesizer, eventuali replan)

![Goal Plan Action alignment](/assets/images/gpa.png)

Molto importanti sono le intersezioni tra Goal, Plan e Act, perché sono queste che ci permettono di valutare il comportamento dell'agente durante l'esecuzione del piano.


| Intersezione | Metrica | Domanda |
|:---|:---|:---|
| Goal ∩ Plan | **Plan Quality** | Il piano è adeguato all'obiettivo? |
| Plan ∩ Act | **Plan Adherence** | Le azioni seguono il piano dichiarato? |
| Goal ∩ Act | **Execution Efficiency** | Il percorso verso l'obiettivo è ottimale? |
| Goal ∩ Plan ∩ Act | **Logical Consistency** | Ragionamento, piano e azioni sono coerenti? |

MLflow espone già questi giudizi come [scorer TruLens](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/trulens/#agent-trace-metrics){:target="_blank"}

### Cosa misura GPA

La RAG Triad guarda *cosa* è stato recuperato e *cosa* è stato scritto. GPA guarda *come* l'agente ha deciso di arrivarci. Per esempio: un Answer Relevance alto con un piano sbagliato è possibile: la risposta è pertinente, ma il percorso è un caso. GPA rende visibile quel caso.


> Come la RAG Triad, anche GPA è valutato con un **LLM-as-judge**. La differenza è l'input: questi scorer **richiedono un trace** completo, perché ispezionano l'albero degli span (piano dichiarato, tool chiamati, ordine dei nodi). Non bastano `inputs` / `outputs` / `context`.
{: .prompt-info }

### Le quattro intersezioni

Prima di applicarle al Data Agent, conviene capire cosa cerca il judge su un esempio minimale. L'esempio sotto — *«Quali lead di vendita dovremmo priorizzare questa settimana, e quali azioni specifiche dobbiamo fare per ciascun lead?»* — è **solo illustrativo**: non fa parte del mini-dataset di tre query del grafo.

#### Plan Quality (Goal ∩ Plan)

Il piano è *teoricamente* adatto all'obiettivo, ancora prima di eseguirlo?

| Cosa cerca il judge | Domanda |
|:---|:---|
| Allineamento al goal | Gli step rispondono davvero alla richiesta? |
| Criteri misurabili | Filtri, soglie, vincoli di urgenza espliciti? |
| Ordine e ownership | Sequenza logica; owner quando servono? |
| Output concreto | Schema/colonne o formato di successo definito? |
| Agenti/tool giusti | Ogni step usa lo specialista corretto? |

| Inadeguato | Adeguato |
|:---|:---|
| «Recupera tutti i lead di vendita degli ultimi 12 mesi» | Solo lead con prossima azione ≤ 14 giorni (o senza prossima azione) |
| «I 20 più grandi» senza giustificazione | Filtro valore > 10k € **o** lead score alto; ordinamento per urgenza di stage |
| «Presenta le raccomandazioni in una sola tabella» senza colonne | Tabella con Nome, Valore, Stage, Urgenza, Prossima azione, Scadenza, Owner |
| Nessun next step / owner | Next step specifici (demo, revisione proposta, escalation) con owner e scadenza |

Nel Data Agent: per “top 3 deal clienti e un grafico” un piano di qualità assegna Cortex al retrieval e il chart generator (più summarizer) alla visualizzazione. Un piano che manda solo il web researcher, o che salta il filtro “in sospeso”, è un piano inadeguato anche se poi qualcuno recupera qualcosa di utile.

#### Plan Adherence (Plan ∩ Act)

Le azioni seguono il piano dichiarato? L'Executor può ignorare uno step, chiamare l'agente sbagliato, o ripianificare senza necessità.

| Cosa cerca il judge | Domanda |
|:---|:---|
| Copertura degli step | Ogni requisito del piano appare nella trace? |
| Filtri completi | Nessuna omissione o filtro “parziale”? |
| Ordine | Passi eseguiti nell'ordine previsto? |
| Output | Stesso schema richiesto dal piano? |
| Replan | Se c'è un cambio di piano, è giustificato e poi seguito? |

| Fuori piano | Conforme |
|:---|:---|
| Recupero di *tutte* le opportunità aperte, senza filtro data | Filtro prossima azione ≤ 14 giorni applicato |
| Solo valore del deal; lead score saltato | Valore > 10k € **o** lead score alto |
| Tabella senza Urgenza / Scadenza / Owner | Colonne come da piano |
| Prossima azione del CRM copiata senza revisione | Prossime azioni aggiornate dal contesto |

Adherence bassa con Quality alta: il piano è corretto ma l'esecuzione non l'ha seguito correttamente.  

#### Execution Efficiency (Goal ∩ Act)

Il percorso verso l'obiettivo è ragionevole, *indipendentemente* dal piano scritto? Anche con piano buono e azioni “logiche” si può essere inefficienti: retrieval duplicati, filtri riapplicati, export non richiesti, retry difensivi.

| Cosa cerca il judge | Domanda |
|:---|:---|
| Lavoro ridondante | Stesso filtro / stesso retrieval ripetuto? |
| Output extra | Formati o artefatti non richiesti? |
| Retry proporzionati | Error handling eccessivo rispetto al fallimento? |

| Inefficiente | Efficiente |
|:---|:---|
| Note recuperate da CRM **e** da cache «per ricontrollare» | Una sola fonte sufficiente |
| Filtro sul valore applicato due volte «per conferma» | Filtri combinati in un solo passaggio |
| Export XLSX **e** CSV quando ne bastava uno | Solo il formato richiesto |

Nel Data Agent: researcher chiamati due volte sulla stessa sotto-query, replan a catena, tool inutili. L'agente può comunque arrivare al goal, ma sprecando passi.

#### Logical Consistency (Goal ∩ Plan ∩ Act)

Ragionamento, piano e azioni restano coerenti per tutta l'esecuzione? Il judge cerca contraddizioni, assunzioni non giustificate e stati impossibili.

| Cosa cerca il judge | Domanda |
|:---|:---|
| Sanity numerica | I conteggi dopo un filtro diminuiscono (o restano), non crescono? |
| Stato vs azione | Le azioni sono compatibili con i fatti già osservati? |
| Grounding del ragionamento | Le assunzioni sono tracciabili al retrieval, non solo parametriche? |
| Replan coerente | Un cambio di piano contraddice lo step precedente senza giustificazione? |

| Incoerente | Coerente |
|:---|:---|
| 96 lead → **113** dopo un filtro più stretto | 96 → 54 dopo valore > 10k € **o** lead score alto |
| Decision maker «da definire» ma next step attivi | Next step solo dove il contesto lo consente |
| Ranking per «engagement minimo» senza giustificare | Criterio di ranking esplicitato e stabile |

Nel Data Agent: un replan che contraddice lo step precedente, un synthesizer che afferma il contrario di ciò che i sub agenti hanno trovato, un Executor che giustifica `goto` in un modo e poi ne fa un altro.

Due scorer aggiuntivi, sempre su trace, coprono i tool:

| Scorer | Cosa valuta |
|:---|:---|
| [`ToolSelection`](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/trulens/){:target="_blank"} | L'agente sceglie lo strumento giusto a ogni passo? |
| [`ToolCalling`](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/trulens/){:target="_blank"} | Lo invoca con i parametri corretti? |

Nel nostro grafo “tool” è anche la scelta del sub-agent: Testuale vs Web vs Chart. ToolSelection bassa sulla query dei deal interni è lo stesso failure mode del researcher sbagliato.

### Scorer TruLens via MLflow

Le evaluation GPA usano lo stesso `mlflow.langchain.autolog()` del [post precedente]({% post_url 2026-17-08-DATA-AGENT-EVAL %}): experiment già aperto, judge `openai:/gpt-4o`. Gli scorer si applicano ai **trace**, non a triple input,output e contesto. Non serve nuova instrumentation `SpanType.RETRIEVER`: GPA legge l'albero degli span (piano del Planner, sequenza Executor/tool).

Lo score è tra **0 e 1**. Le trace GPA sono più lunghe e intricate delle triple input/output/contesto della RAG Triad: conviene un judge più performante, per esempio `openai:/gpt-4o` invece di un mini.

```python
import mlflow
from mlflow.genai.scorers.trulens import (
    PlanQuality,
    PlanAdherence,
    ExecutionEfficiency,
    LogicalConsistency,
    ToolSelection,
    ToolCalling,
)

judge = "openai:/gpt-4o"

traces = mlflow.search_traces(experiment_ids=["<experiment_id>"])

results = mlflow.genai.evaluate(
    data=traces,
    scorers=[
        PlanQuality(model=judge),
        PlanAdherence(model=judge),
        ExecutionEfficiency(model=judge),
        LogicalConsistency(model=judge),
        ToolSelection(model=judge),
        ToolCalling(model=judge),
    ],
)
```

Il blocco sopra valuta **trace già registrate**.  
Se l'agente non è ancora stato eseguito, si passa in argomento anche `predict_fn`: MLflow invoca il grafo, `autolog` scrive la trace, gli scorer GPA la leggono, come nel caso della RAG Triad.

```python
from langchain.schema import HumanMessage

ENABLED = [
    "cortex_researcher", "web_researcher",
    "chart_generator", "chart_summarizer", "synthesizer",
]

def run_data_agent(inputs: dict) -> str:
    """Wrapper per `mlflow.genai.evaluate(..., predict_fn=...)`.

    MLflow la chiama una volta per ogni riga del dataset (`inputs["query"]`):
    costruisce lo stato iniziale, esegue il grafo, restituisce l'ultimo
    messaggio. `autolog` registra la trace di quell'invoke; gli scorer GPA
    la usano come unico input.
    """
    query = inputs["query"]
    state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "enabled_agents": ENABLED,
    }
    result = graph.invoke(state)
    return result["messages"][-1].content


results = mlflow.genai.evaluate(
    data=[{"inputs": {"query": q}} for q in QUERIES],
    predict_fn=run_data_agent, # <-------
    scorers=[
        PlanQuality(model=judge),
        PlanAdherence(model=judge),
        ExecutionEfficiency(model=judge),
        LogicalConsistency(model=judge),
    ],
)
```

`QUERIES` nell'esempio è lo stesso [mini-dataset]({% post_url 2026-17-08-DATA-AGENT-EVAL %}#dataset-di-eval) del post sulla RAG Triad. 
La differenza è *cosa* misuriamo sulla stessa esecuzione.

> Gli scorer RAG (`Groundedness`, `ContextRelevance`, `AnswerRelevance`) accettano dati espliciti **oppure** un trace. Gli scorer GPA accettano **solo** il trace: senza albero degli span non c'è piano da giudicare né sequenza di azioni.
{: .prompt-warning }

### Come leggere i fallimenti con GPA

GPA evidenzia failure mode che la RAG Triad vede solo di sfuggita. Le due famiglie di metriche **non sostituiscono** l'un l'altra ma si **complementano**.

Vediamo alcuni esempi tipici che possono verificarsi nel caso del Data Agent:

**Query 1 — top 3 deal + grafico.**  
Se abbiamo un grafico senza riassunto, Answer Relevance è a zero. GPA aggiunge *dove* si è rotto il contratto col piano: Plan Adherence bassa se lo step “riassumi il grafico” c'era e non è stato eseguito; Plan Quality bassa se il Planner non aveva proprio previsto di fare il riassunto. Quality vs Adherence discrimina **Planner** vs **Executor**.

**Query 2 — deal in sospeso + regolamentazione + proposta di valore.**  
Path lungo: planner → executor → cortex → replan → web → cortex → synthesizer. Qui il pattern evidenzia un **GPA alto** (piano adeguato, seguito, efficiente, coerente) mentre **Context Relevance / Groundedness restano bassi**. La proposta non è supportata dal retrieval: il *comportamento* va bene, il *contenuto* no. Solo la RAG Triad, in questo caso, evidenzia questo problema.

**Query 3 — deal più grande + note riunioni + news.**  
Se la sorgente dati non risponde, Context Relevance va a zero. Sul lato GPA, un pattern frequente è **Quality e Consistency accettabili con Adherence ed Efficiency basse**: il piano c'era, dopo lo step 1 è stato di fatto ignorato (capability limitation, tool sbagliato, niente ripianificazione utile). È un failure mode dell'**Executor**, non del Planner. ToolSelection / ToolCalling dicono se l'agente ha insistito sullo strumento sbagliato o con parametri errati.

| Pattern GPA | Failure mode probabile |
|:---|:---|
| Plan Quality ↓ | Decomposizione sbagliata (agente o step mancante) — **Planner** |
| Plan Adherence ↓ | Executor o sub-agent fuori piano — **Executor** |
| Execution Efficiency ↓ | Replan, loop, retrieval ridondanti |
| Logical Consistency ↓ | Contraddizione tra piano, azioni e output |
| ToolSelection ↓ | Researcher o tool sbagliato per lo step |
| GPA ↑, Groundedness / Context Relevance ↓ | Comportamento ok, retrieval/sintesi deboli — serve la RAG Triad |

### Conclusioni

GPA non sostituisce la RAG Triad: la completa. La prima giudica il *contenuto* (retrieval e sintesi), la seconda il *comportamento* (piano, aderenza, efficienza, coerenza). Insieme isolano se il problema è nel Planner, nell'Executor o nei dati recuperati.

In produzione gli scorer indicano *dove* Goal, Plan e Act si disallineano.


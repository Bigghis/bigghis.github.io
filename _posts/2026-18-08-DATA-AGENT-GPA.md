---
title: "Goal–Plan–Action (GPA) di un sistema multi-agent"
description: "I fallimenti critici di un agente emergono alle intersezioni tra Goal, Plan e Action, non basta solo valutare groundness e context relevance (RAG Triad)"
date: 2026-08-18 12:00:00 +0200
categories: [LangGraph, MLflow, Evaluation]
tags: [Data Agent, GPA, TruLens, MLflow, Tracing, Plan Quality, Plan Adherence, Execution Efficiency, Logical Consistency, Inline Evaluation, Prompt Engineering]
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

### Migliorare il GPA dell'agente

Un failure mode ricorrente che è stato rilevato è **Plan Adherence bassa**.In sostanza l'Executor non segue il piano. 

In genere per migliorare il GPA si possono applicare le seguenti leve:

- **Prompt**: arricchire il planning prompt con sub-goal espliciti, pre-condizioni e post-condizioni
- **Inline evaluation**: feedback in tempo reale nello stato, così l'agente valuta uno step prima di passare al successivo
- Tuning del retriever o cambio modello
- Validare sempre le modifiche con evaluation **offline** sulle stesse query

Qui applichiamo due modifiche di prova: inline evals sui researcher e planning prompt arricchito.

>Prima di modificare qualsiasi aspetto del grafo, conviene ragionare per **versioni**: ogni variante del grafo viene eseguita sulle stesse query e registrata come run separata. Ciò agevola il confronto tra run differenti e permette di capire subito se un cambiamento ha migliorato o peggiorato l'agente.
{: .prompt-info }

#### Inline evaluation sui researcher

Dopo ogni retrieval proviamo a valutare subito il contesto recuperato e scrivere score + spiegazione nei `messages` dello stato. L'Executor legge quel feedback e decide se fare ricerca aggiuntiva o ripianificare.

> TruLens espone un decorator `inline_evaluation` specifico per LangGraph. Qui riusiamo lo stesso scorer `ContextRelevance` della [RAG Triad]({% post_url 2026-17-08-DATA-AGENT-EVAL %}): non lo passiamo a `mlflow.genai.evaluate`, ma lo invochiamo direttamente dentro il nodo researcher e appendiamo score + spiegazione allo stato.
{: .prompt-info }

Partiamo dai researcher instrumentati del [post sulla RAG Triad]({% post_url 2026-17-08-DATA-AGENT-EVAL %}):

```python
from typing import Literal

import mlflow
from mlflow.entities import SpanType, Document
from mlflow.genai.scorers.trulens import ContextRelevance
from langchain.schema import HumanMessage
from langgraph.types import Command

from helper import State, cortex_agent, web_search_agent

inline_judge = ContextRelevance(model="openai:/gpt-4o-mini")


@mlflow.trace(span_type=SpanType.RETRIEVER, name="cortex_researcher")
def cortex_agents_research_node(
    state: State,
) -> Command[Literal["executor"]]:
    query = state.get("agent_query") or state.get("user_query", "")
    agent_response = cortex_agent.invoke({"messages": query})
    content = agent_response["messages"][-1].content

    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_inputs({"query": query})
        span.set_outputs([Document(page_content=content)])

    feedback = inline_judge(
        inputs={"query": query},
        expectations={"context": content},
    )
    score = feedback.metadata["score"]
    rationale = getattr(feedback, "rationale", "") or ""

    new_message = HumanMessage(content=content, name="cortex_researcher")
    eval_msg = HumanMessage(
        content=f"[inline eval] context_relevance={score}: {rationale}",
        name="inline_evaluator",
    )
    return Command(
        update={"messages": [new_message, eval_msg]},
        goto="executor",
    )


@mlflow.trace(span_type=SpanType.RETRIEVER, name="web_researcher")
def web_research_node(
    state: State,
) -> Command[Literal["executor"]]:
    agent_query = state.get("agent_query")
    result = web_search_agent.invoke({"messages": agent_query})
    content = result["messages"][-1].content

    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_inputs({"query": agent_query})
        span.set_outputs([Document(page_content=content)])

    feedback = inline_judge(
        inputs={"query": agent_query},
        expectations={"context": content},
    )
    score = feedback.metadata["score"]
    rationale = getattr(feedback, "rationale", "") or ""

    result["messages"][-1] = HumanMessage(
        content=content, name="web_researcher"
    )
    eval_msg = HumanMessage(
        content=f"[inline eval] context_relevance={score}: {rationale}",
        name="inline_evaluator",
    )
    return Command(
        update={"messages": result["messages"] + [eval_msg]},
        goto="executor",
    )
```

Dopo ogni ricerca, lo stato contiene anche il giudizio inline. L'Executor lo usa per capire se mancano dettagli chiave prima di andare avanti.

#### Sub-goal espliciti nel planning prompt

Aggiungiamo agli step delle precondizioni e postcondizioni e un obiettivo esplicito.  
Ogni step del piano riceve `pre_conditions`, `post_conditions` e `goal`. L'Executor capisce meglio l'obiettivo di ciascun passo: migliorano tool calling e decisioni di routing.

```python
import helper
import prompts
from langchain.schema import HumanMessage


def patched_plan_prompt(state):
    base = prompts.plan_prompt(state).content
    insertion = (
        '"action": "string",\n'
        '            "pre_conditions": ["string", ...],\n'
        '            "post_conditions": ["string", ...],\n'
        '            "goal": "string",'
    )
    return HumanMessage(content=base.replace('"action": "string",', insertion))


helper.plan_prompt = patched_plan_prompt
```

Il template di output del Planner passa da `{agent, action}` a `{agent, action, pre_conditions, post_conditions, goal}`. A runtime il Planning LLM lo popola per ogni step.

#### Ricostruire il grafo e versionare l'agente

Ricostruiamo il grafo con i researcher modificati (Planner, Executor e gli altri nodi restano quelli del [post multi-agent]({% post_url 2026-16-08-DATA-AGENT-MULTI-AGENT %})):

```python
from langgraph.graph import START, StateGraph
from helper import (
    State, planner_node, executor_node,
    chart_node, chart_summary_node, synthesizer_node,
)

workflow = StateGraph(State)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("web_researcher", web_research_node)
workflow.add_node("cortex_researcher", cortex_agents_research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("chart_summarizer", chart_summary_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "planner")
graph = workflow.compile()
```
Per capire che impatto hanno avuto le modifiche, bisogna eseguire le due versioni sulle stesse query e confrontare i risultati.


```python
from mlflow.genai.scorers.trulens import (
    PlanQuality,
    PlanAdherence,
    ExecutionEfficiency,
    LogicalConsistency,
    Groundedness,
    AnswerRelevance,
    ContextRelevance,
)

judge = "openai:/gpt-4o"

with mlflow.start_run(run_name="v2: inline evals + sub-goals nel piano"):
    results_v2 = mlflow.genai.evaluate(
        data=[{"inputs": {"query": q}} for q in QUERIES],
        predict_fn=run_data_agent,
        scorers=[
            PlanQuality(model=judge),
            PlanAdherence(model=judge),
            ExecutionEfficiency(model=judge),
            LogicalConsistency(model=judge),
            Groundedness(model=judge),
            AnswerRelevance(model=judge),
            ContextRelevance(model=judge),
        ],
    )
```

`QUERIES` e `run_data_agent` sono gli stessi del [post sulla RAG Triad]({% post_url 2026-17-08-DATA-AGENT-EVAL %}#dataset-di-eval) e della sezione GPA sopra. Cambia solo il grafo wrappato.

#### Leggere i risultati del confronto

Confrontando la versione base con la `v2` abbiamo i seguenti risultati:

| Metrica | Direzione | Perché |
|:---|:---|:---|
| Plan Adherence | ↑ netto | Ogni step del piano (e dei replan) viene eseguito; le deviazioni sono giustificate |
| Groundedness / Answer Relevance | ↑ | Ricerche aggiuntive colmano i gap segnalati dall'inline eval |
| Context Relevance | ≈ | Il retrieval base non cambia; migliora la *copertura* del piano |
| Execution Efficiency | ↓ lieve | Trace più lunghe: research extra innescato dall'inline eval |
| Logical Consistency | ↓ lieve | Più passi → più superficie per piccole incoerenze |

Nell'esempio, Plan Adherence passa da 0 a 1: nella base molti step del piano erano omessi; in `v2` nessun passo è saltato e ogni scostamento è motivato (es. limiti di accesso ai dati esterni).

È un **trade-off esplicito**: si sacrifica un po' di efficienza per completare il goal. Nelle trace di `v2` si vedono chiamate extra a web researcher proprio dove l'inline eval ha segnalato contesto insufficiente, e i sub-goal del piano arricchito guidano l'Executor.

### Conclusioni

GPA non sostituisce la RAG Triad: la completa. La prima giudica il *contenuto* (retrieval e sintesi), la seconda il *comportamento* (piano, aderenza, efficienza, coerenza). Insieme isolano se il problema è nel Planner, nell'Executor o nei dati recuperati.

Una volta individuato il failure mode, le modifiche da effettuare sono mirate e validate offline confrontando versioni sullo stesso dataset. 
E' possibile anche, per esempio, valutare i singoli agenti specializzati, sperimentare altre metriche inline, aggiornare i prompt etc.

In definitiva, un agente affidabile non nasce assolutamente dal primo prompt definito, ma da un ciclo continuo di tracing, valutazione e iterazione: si misura dove Goal, Plan e Act si disallineano, si interviene in modo mirato e si lascia che siano i numeri a dire se il sistema è davvero migliorato.



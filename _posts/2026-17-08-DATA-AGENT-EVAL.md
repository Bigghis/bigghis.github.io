---
title: "Osservare e misurare le performance del Data Agent"
description: "Tracing con MLflow e RAG Triad con gli scorer TruLens: Context Relevance, Groundedness e Answer Relevance sul sistema multi-agente LangGraph."
date: 2026-08-17 12:00:00 +0200
categories: [LangGraph, MLflow, Evaluation]
tags: [Data Agent, RAG Triad, TruLens, MLflow, Mlflow monitoring, Tracing, LLM-as-Judge, Groundedness, Context Relevance, Answer Relevance]
comments: false
protected: false
mermaid: true
---

Nel [post precedente]({% post_url 2026-16-08-DATA-AGENT-MULTI-AGENT %}) abbiamo costruito un Multi-Agent (Data Agent) con LangGraph: Planner, Executor, researcher e synthesizer collaborano su uno stato condiviso. Dobbiamo capire in che modo gli agenti portano a termine il loro lavoro e se lo fanno in modo corretto ed efficiente.

In questa parte aggiungiamo la parte del **tracing** e sistemi di **valutazione** del lavoro svolto dagli agenti. Registriamo ogni passo dell'esecuzione e misuriamo tre metriche con un LLM-as-judge: **Context Relevance**, **Groundedness** e **Answer Relevance** — la cosiddetta *RAG Triad*. Useremo [MLflow](https://mlflow.org/docs/latest/genai/tracing/){:target="_blank"} per i trace e gli [scorer TruLens](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/trulens/){:target="_blank"} esposti da MLflow per le metriche.

### Perché valutare un Data Agent

Un Data Agent, nella sostanza, fa **ricerca** e **sintesi**: recupera contesto da fonti interne o dal web e poi genera una risposta. La RAG Triad, nata per i sistemi Retrieval-Augmented Generation, si applica bene anche per questi sistemi:

| Fase del Data Agent | Metrica | Domanda che risponde |
|:---|:---|:---|
| Ricerca (researcher) | **Context Relevance** | Il contesto recuperato è pertinente alla sotto-query? |
| Sintesi (synthesizer / chart) | **Groundedness** | La risposta è supportata dal contesto recuperato? |
| Risposta end-to-end | **Answer Relevance** | La risposta è pertinente alla query dell'utente? |

Senza queste misure, un fallimento resta opaco: non sappiamo se il problema è nel retrieval, nell'allucinazione del modello, o in una risposta semplicemente fuori tema.

> Le metriche della RAG Triad sono calcolate da un **LLM-as-judge**: un modello separato riceve l'input, l'output e contesto e produce uno score con reasoning.
{: .prompt-info }

Queste metriche sono già definite in [TruLens](https://www.trulens.org/){:target="_blank"} come *feedback function* (Groundedness, Context Relevance, Answer Relevance). MLflow le espone come [scorer di terze parti](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/third-party/trulens/){:target="_blank"} in `mlflow.genai.scorers.trulens`, così possiamo usarle nello stesso flusso di tracing e `evaluate` senza passare dalla dashboard nativa di TruLens.

| Scorer MLflow (TruLens) | Input tipici | Cosa valuta |
|:---|:---|:---|
| [`ContextRelevance`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.trulens.ContextRelevance){:target="_blank"} | query + contesto recuperato | Qualità del retrieval |
| [`Groundedness`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.trulens.Groundedness){:target="_blank"} | output + contesto | Assenza di allucinazioni rispetto al contesto |
| [`AnswerRelevance`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html#mlflow.genai.scorers.trulens.AnswerRelevance){:target="_blank"} | input (query) + output | Pertinenza end-to-end |

### Tracing: OpenTelemetry, trace e span

Per calcolare Groundedness e Context Relevance serve conoscere, dunque, **quali passi** l'agente ha compiuto e **quale contesto** ha recuperato lungo il percorso. Queste informazioni fanno parte del **tracing**.

Una **trace** è il registro dell'intera esecuzione su una query: ogni nodo visitato, ogni tool chiamato, ogni output prodotto. In un Data Agent un percorso tipico può partire da un nodo di ricerca, passare da un chart, tornare a un altro researcher e chiudere con la sintesi. Dentro quella sequenza i passi di **retrieval** sono quelli che contengono i dati chiave per la RAG Triad.

Il tracing che useremo è costruito su [OpenTelemetry](https://opentelemetry.io/){:target="_blank"}: un sistema di tracing distribuito **indipendente dal linguaggio**. TruLens e MLflow condividono questo modello: catturano ogni passo che l'agente compie per raggiungere l'obiettivo, senza legarsi a un runtime specifico.

Quei passi si chiamano **span**, ovvero l'unità minima di lavoro all'interno della trace. In un Data Agent gli span coprono, tra gli altri:

| Tipo di span | Cosa rappresenta nel grafo |
|:---|:---|
| Planning | Il Planner decompone la query in step |
| Routing | L'Executor sceglie il prossimo sub-agent |
| Retrieval | Un researcher recupera contesto |
| Tool use | Chiamata a uno strumento (SQL, search, …) |
| Generation | Synthesizer o chart summarizer |

MlFlow mette a disposizione vari tipi di [span](https://mlflow.org/docs/latest/genai/concepts/span/#span-types){:target="_blank"}. Per questa valutazione prestiamo attenzione soprattutto a quelli di tipo **retrieval** (`RETRIEVER` in MLflow). Contengono la sotto-query e i documenti o/e i contesti recuperati che servono a calcolare Context Relevance e Groundedness.



MLflow offre [autolog per LangGraph](https://mlflow.org/docs/latest/genai/flavors/langchain/autologging/){:target="_blank"}: `mlflow.langchain.autolog()` che registra automaticamente i nodi del grafo, le chiamate LLM e i tool. 

### Setup MLflow e TruLens

Gli scorer TruLens sono disponibili in MLflow.  

Abilitiamo tracing e scegliamo un experiment:

```python
import mlflow

mlflow.langchain.autolog()
mlflow.set_experiment("Sales Data Agent")
# opzionale in locale:
# mlflow.set_tracking_uri("http://localhost:5000")
```

Il modello "judge" si specifica come URI LiteLLM, ad esempio `openai:/gpt-4o` oppure `openai:/gpt-4o-mini` etc.. La soglia **threshold** di default è `0.5`: lo score passa se è ≥ soglia. Per una valutazione più severa si può alzare a `0.7` etc.

### Gli scorer TruLens via MLflow

Gli scorer sono definiti in `mlflow.genai.scorers.trulens`. Si possono invocare direttamente su una singola tripla input/output/contesto:

```python
from mlflow.genai.scorers.trulens import Groundedness, AnswerRelevance, ContextRelevance

judge = "openai:/gpt-4o"

groundedness = Groundedness(model=judge, threshold=0.5)
answer_relevance = AnswerRelevance(model=judge)
context_relevance = ContextRelevance(model=judge)

feedback = groundedness(
    outputs="MLflow è una piattaforma open-source per il ciclo di vita ML.",
    expectations={
        "context": (
            "MLflow è una piattaforma open-source per experiment tracking, "
            "model registry e deployment."
        ),
    },
)

print(feedback.value)                 # "yes" / "no"
print(feedback.metadata["score"])     # 0.0 – 1.0
```

In batch, su un dataset di esempi già materializzati (query, risposta, contesto), si usa [`mlflow.genai.evaluate`](https://mlflow.org/docs/latest/genai/eval-monitor/){:target="_blank"}:

```python
import mlflow
from mlflow.genai.scorers.trulens import Groundedness, AnswerRelevance, ContextRelevance

eval_dataset = [
    {
        "inputs": {"query": "What is MLflow?"},
        "outputs": "MLflow is an open-source AI engineering platform.",
        "expectations": {
            "context": "MLflow is an ML platform for experiment tracking and model deployment."
        },
    },
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        Groundedness(model="openai:/gpt-4o"),
        AnswerRelevance(model="openai:/gpt-4o"),
        ContextRelevance(model="openai:/gpt-4o"),
    ],
)

print(results.tables["eval_results"])
print(results.metrics)  # es. Groundedness/mean, AnswerRelevance/mean, …
```


### Instrumentation dei nodi di ricerca

Autolog cattura il grafo in automatico, ma per la RAG Triad dobbiamo esporre esplicitamente **query** e **contesto recuperato** sugli span di retrieval.
Per questo motivo si usa annotare i nodi di ricerca con `@mlflow.trace` e [`SpanType.RETRIEVER`](https://mlflow.org/docs/latest/genai/concepts/span/){:target="_blank"}.  
Riprendiamo i due researcher del post precedente e li annotiamo:

```python
from typing import Literal

import mlflow
from mlflow.entities import SpanType, Document
from langchain.schema import HumanMessage
from langgraph.types import Command

from helper import State, cortex_agent, web_search_agent


@mlflow.trace(span_type=SpanType.RETRIEVER, name="cortex_researcher") # <-------
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

    new_message = HumanMessage(content=content, name="cortex_researcher")
    return Command(
        update={"messages": [new_message]},
        goto="executor",
    )


@mlflow.trace(span_type=SpanType.RETRIEVER, name="web_researcher") # <-------
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

    result["messages"][-1] = HumanMessage(
        content=content, name="web_researcher"
    )
    return Command(
        update={"messages": result["messages"]},
        goto="executor",
    )
```

Senza questa etichetta, MLflow vedrebbe comunque il nodo come generico chain/agent; gli scorer TruLens non troverebbero un retrieval strutturato da cui estrarre il contesto per Groundedness e Context Relevance.

Ricostruiamo il grafo con i researcher instrumentati (Planner, Executor e gli altri nodi restano quelli del [post precedente]({% post_url 2026-16-08-DATA-AGENT-MULTI-AGENT %})):

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

### Dataset di eval

Costruiamo un mini dataset di eval, sottoponiamo l'agente a tre query progressive e valutiamo la risposta. Non serve che la risposta sia perfetta: l'obiettivo è **produrre trace e score** da cui diagnosticare i failure mode.

1. *What are our top 3 client deals? Chart the deal value for each.*
2. *Identify our pending deals, research if they may be experiencing regulatory changes, and using the meeting notes for each customer, provide a new value proposition for each given the regulatory changes.*
3. *Identify our largest client deal, then find important topics in the meeting notes with that company, and find a news article related to the important topics discussed.*

```python
from langchain.schema import HumanMessage
from mlflow.genai.scorers.trulens import (
    Groundedness, AnswerRelevance, ContextRelevance,
)

ENABLED = [
    "cortex_researcher", "web_researcher",
    "chart_generator", "chart_summarizer", "synthesizer",
]

QUERIES = [
    "What are our top 3 client deals? Chart the deal value for each.",
    (
        "Identify our pending deals, research if they may be experiencing "
        "regulatory changes, and using the meeting notes for each customer, "
        "provide a new value proposition for each given the regulatory changes."
    ),
    (
        "Identify our largest client deal, then find important topics in the "
        "meeting notes with that company, and find a news article related to "
        "the important topics discussed."
    ),
]


def run_data_agent(inputs: dict) -> str:
    query = inputs["query"]
    state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "enabled_agents": ENABLED,
    }
    result = graph.invoke(state)
    # ultima risposta utile nello stato
    return result["messages"][-1].content


eval_data = [{"inputs": {"query": q}} for q in QUERIES]

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=run_data_agent,
    scorers=[
        Groundedness(model="openai:/gpt-4o"),
        AnswerRelevance(model="openai:/gpt-4o"),
        ContextRelevance(model="openai:/gpt-4o"),
    ],
)
```

In alternativa, dopo aver invocato il grafo a mano, si recuperano i trace già loggati e si valutano offline:

```python
traces = mlflow.search_traces(experiment_ids=["<experiment_id>"])

results = mlflow.genai.evaluate(
    data=traces,
    scorers=[
        Groundedness(model="openai:/gpt-4o"),
        AnswerRelevance(model="openai:/gpt-4o"),
        ContextRelevance(model="openai:/gpt-4o"),
    ],
)
```

> L'output di un LLM non è deterministico: ripetendo le stesse query puoi ottenere score diversi. Un fallimento (chart senza testo, deal “pending” non filtrati, sorgente dati non raggiungibile) **non** va ritentato a oltranza: è materiale di diagnosi.
{: .prompt-warning }

### Come leggere i fallimenti

Nella UI MLflow trovi i **trace** (albero degli span: planner → executor → researcher → …) e la tabella di `evaluate` con score e rationale del judge. Vediamo alcuni casi tipici che possono verificarsi nel caso del Data Agent:

**Query 1 — top 3 deals + chart.**  
Spesso compare il grafico ma manca il riassunto testuale. **Answer Relevance** crolla verso zero: la risposta non affronta la richiesta in forma utile all'utente. Se i researcher non hanno portato deal pertinenti, anche **Context Relevance** resta bassa.

**Query 2 — pending deals + regolamentazione + value proposition.**  
La risposta può sembrare pertinente (**Answer Relevance** alta) ma **Groundedness** bassa: il synthesizer inferisce value proposition non supportate dai contesti recuperati. Spesso il filtro “solo deal pending” non è stato applicato: nel trace vedi un path lungo (planner → executor → cortex → replan → web → cortex → synthesizer) e puoi ispezionare input/output di ogni nodo.

**Query 3 — largest deal + meeting notes + news.**  
Se il retrieval sulla sorgente dati fallisce, a monte non c'è contesto sul deal più grande: Context Relevance e Groundedness riflettono un problema di **accesso ai dati**, non solo di prompting.

Il punto della RAG Triad è proprio questo: tre score diversi isolano **tre failure mode** distinti.

| Pattern di score | Failure mode probabile |
|:---|:---|
| Answer Relevance ↓ | Risposta fuori tema o incompleta (es. solo chart, niente testo) |
| Groundedness ↓, Answer Relevance ↑ | Allucinazione / inferenza non supportata dal retrieval |
| Context Relevance ↓ | Retrieval sbagliato o sotto-query mal formulata |



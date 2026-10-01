# Agentic RAG Chatbot – Proof of Concept

[English](README.md) | **Magyar**

Agentic RAG (Retrieval-Augmented Generation) alapú chatbot prototípus Pythonban – [LangGraph](https://github.com/langchain-ai/langgraph) frameworkkel, helyben futó, nyílt forráskódú LLM-mel és [Streamlit](https://streamlit.io/) felülettel, Dockerrel teljesen konténerizálva.

> **Állapot:** 🚧 Fejlesztés alatt. A README már tartalmazza a projekt céljait, követelményeit és a dokumentáció szerkezetét; a *Kitöltendő* jelölésű részek a megvalósítás előrehaladtával egészülnek ki.

## Tartalom

1. [Áttekintés](#áttekintés)
2. [A feladat követelményei](#a-feladat-követelményei)
3. [Problémafelvetés és motiváció](#problémafelvetés-és-motiváció)
4. [Rendszerarchitektúra](#rendszerarchitektúra)
5. [Tervezési döntések](#tervezési-döntések)
6. [Értékelés](#értékelés)
7. [Telepítés és futtatás](#telepítés-és-futtatás)
8. [Projektstruktúra](#projektstruktúra)
9. [Licenc](#licenc)

## Áttekintés

A cél egy működőképes, jól dokumentált és reprodukálható prototípus, amely bemutatja:

- **az agentic workflow tervezését LangGraph-ban** – autonóm döntéshozatal feltételes elágazással (conditional routing), az összetett kérések részfeladatokra bontása és azok önálló végrehajtása, valamint a köztes eredmények explicit állapotkezelése;
- **egy moduláris RAG alrendszert** – dedikált LangGraph algráf (subgraph), amelyet a fő workflow hív meg;
- **a visszakeresésen túlmutató eszközhasználatot** – legalább egy olyan eszköz (tool), amely többre képes a tudásbázisban való keresésnél;
- **a teljesen helyi futtatást** – nyílt forráskódú LLM, fizetős API-k nélkül;
- **a mérhető minőséget és teljesítményt** – mini értékelő készlet, valamint terheléses teszt a szűk keresztmetszet azonosításával.

**Technológiák:** Python, LangGraph, Streamlit, Docker / Docker Compose és egy helyben futó, nyílt forráskódú LLM (lásd: [Tervezési döntések](#tervezési-döntések)).

A projekt egy Medior AI Engineer pozícióra kiírt technikai feladat megoldása; az eredeti feladatkiírás a `task/` mappában található.

## A feladat követelményei

A feladatkiírás egyes követelményeinek állapota:

**Probléma és adatforrás**

- [ ] Valós probléma (domain / use case) választása, írásos indoklással
- [ ] Szabadon választott szöveges adatforrás – a hangsúly a minőségi feldolgozáson és a skálázható adatintegráción van, nem a mennyiségen

**Agentic architektúra (LangGraph)**

- [ ] Legalább 5 node-ból álló agentic workflow
- [ ] Autonóm döntéshozatal (pl. conditional routing)
- [ ] Részfeladatokra bontás és önálló végrehajtás
- [ ] Állapotkezelés a köztes eredmények tárolására
- [ ] Legalább 2 eszköz (tool), amelyek közül legalább egy nem pusztán visszakeresési célú
- [ ] Dedikált, moduláris RAG algráf (subgraph), amely a fő workflow-ból hívható (nem számít bele a node-ok számába)

**Modell, UI és futtatási környezet**

- [ ] A helyi erőforrásokhoz illeszkedő, nyílt forráskódú LLM (fizetős API-k nélkül), a trade-offok indoklásával
- [ ] Streamlit prototípus UI, amely bemutatja az ágens működésének főbb lépéseit és a RAG folyamat eredményét
- [ ] Konténerizálás: `Dockerfile` (kötelező) és `docker-compose.yml` (több komponens esetén előny)

**Értékelés és teljesítmény**

- [ ] Funkcionális értékelés egy 10–20 kérdésből álló mini készleten (egyetlen node-ra vagy a teljes workflow-ra)
- [ ] Terheléses teszt 50–200 lekérdezéssel: alapvető latency metrikák, a fő szűk keresztmetszet azonosítása, 1–2 konkrét optimalizálási javaslat

**Dokumentáció**

- [ ] Ez a README: a probléma és a célkitűzés, az architektúra és a tervezési döntések indoklása, az értékelés és a terheléses teszt eredményei, telepítési és futtatási útmutató

## Problémafelvetés és motiváció

> 🚧 *Kitöltendő* – a választott domain / use case és a chatbot célja, három kérdés mentén:
>
> - **Miért releváns a probléma?**
> - **Milyen felhasználói igényt elégít ki?**
> - **Miért előnyös rá az agentic RAG megközelítés** – egy egyszerű „visszakeresés, majd generálás” lépéssel szemben?

## Rendszerarchitektúra

Magas szintű célarchitektúra:

```mermaid
flowchart LR
    user([Felhasználó]) <--> ui[Streamlit UI]
    subgraph app [LangGraph ágens]
        main["Fő agentic workflow<br/>≥ 5 node · conditional routing · közös állapot"]
        rag[["RAG algráf"]]
        main <--> rag
    end
    ui <--> main
    main <--> tools["Eszközök (tools)<br/>≥ 1 nem visszakeresési célú"]
    main <--> llm["Helyi, nyílt forráskódú LLM"]
    rag <--> index[("Dokumentumindex")]
    source["Szöveges adatforrás"] -. betöltés .-> index
```

| Komponens | Szerep |
|---|---|
| Streamlit UI | Chatfelület, amely megjeleníti az ágens lépéseit és a visszakeresett forrásokat |
| Fő agentic workflow | Megtervezi, irányítja és végrehajtja a részfeladatokat; a köztes eredményeket a gráf állapotában tárolja |
| RAG algráf | Moduláris visszakeresési folyamat a dokumentumindex felett, a fő workflow-ból hívva |
| Eszközök (tools) | A visszakeresésen túlmutató képességek – legalább egy nem visszakeresési célú eszköz |
| Helyi LLM | Helyben kiszolgált, nyílt forráskódú modell, fizetős API-k nélkül |
| Dokumentumindex | A választott szöveges adatforrás feldarabolt (chunking) és beágyazott (embedding) dokumentumai |

> 🚧 *Kitöltendő:* a fő workflow node-jai és routing logikája, a RAG algráf lépései, az eszközök, az állapot (state) sémája és az adatbetöltési (ingestion) folyamat.
>
> Tipp: a LangGraph a lefordított (compiled) gráfot Mermaid formátumban is exportálni tudja: `graph.get_graph(xray=True).draw_mermaid()` – az `xray=True` a RAG algráfot is kibontja.

## Tervezési döntések

| Terület | Fő szempontok (trade-offok) | Választás és indoklás |
|---|---|---|
| Domain és adatforrás | Relevancia, elérhetőség és licenc, előfeldolgozási igény | _Eldöntendő_ |
| LLM | Válaszminőség vs. válaszidő vs. memóriaigény (RAM/VRAM); eszközhívás (tool calling) támogatása; licenc | _Eldöntendő_ |
| LLM kiszolgálás | Beüzemelési igény, konténerizálhatóság, áteresztőképesség | _Eldöntendő_ |
| Embedding modell | Visszakeresési minőség vs. sebesség; nyelvi lefedettség | _Eldöntendő_ |
| Vektoradatbázis | Perzisztencia, metaadat-alapú szűrés, skálázhatóság | _Eldöntendő_ |
| Darabolás (chunking) | Chunkméret és átfedés vs. visszakeresési pontosság és kontextushossz | _Eldöntendő_ |

## Értékelés

### Funkcionális értékelés

**Megközelítés:** 10–20 domainspecifikus kérdésből álló mini értékelő készlet – mindegyikhez referenciaválasz és, ahol releváns, az elvárt forrásdokumentumok –, amellyel egy kiválasztott node vagy a teljes agentic workflow értékelhető.

**Lehetséges metrikák:** a válasz helyessége a referenciához képest, hűség a visszakeresett kontextushoz (faithfulness), visszakeresési találati arány (hit rate@k), valamint a routing és az eszközválasztás pontossága.

> 🚧 *Kitöltendő:* az értékelő készlet helye, a pontozás módja, az eredmények, a levont következtetések és a reprodukálásukhoz szükséges parancs.

### Terheléses teszt és a szűk keresztmetszet elemzése

**Forgatókönyv:** 50–200 lekérdezés a futó rendszeren, a párhuzamossági szint, a lekérdezések összetétele és a hardver dokumentálásával.

**Mért értékek:** válaszidő (átlag, p50, p95, p99, maximum), áteresztőképesség és hibaarány, valamint node-onkénti válaszidő-bontás a fő szűk keresztmetszet azonosításához – ezt 1–2 konkrét optimalizálási javaslat követi.

> 🚧 *Kitöltendő:* az eredmények, a szűk keresztmetszet elemzése, az optimalizálási javaslatok és a reprodukálásukhoz szükséges parancs.

## Telepítés és futtatás

### Előfeltételek

- Git
- Docker és Docker Compose v2
- Elegendő memória a választott LLM helyi futtatásához (_a pontos RAM/VRAM-igény még nincs meghatározva_)

### Futtatás Docker Compose-zal

> A `Dockerfile` és a `docker-compose.yml` még nincsenek a repositoryban – az alábbi parancsok a tervezett futtatási módot mutatják.

```bash
git clone https://github.com/Csabikahhh/agentic-rag-chatbot-poc.git
cd agentic-rag-chatbot-poc
docker compose up --build
```

Ezután a felület a <http://localhost:8501> címen érhető el.

> 🚧 *Kitöltendő:* környezeti változók, a modell letöltése, az adatbetöltés (ingestion), a Docker nélküli helyi futtatás, valamint az értékelés és a terheléses teszt futtatási parancsai.

## Projektstruktúra

```text
agentic-rag-chatbot-poc/
├── docs/          # Kiegészítő dokumentáció (diagramok, értékelési és teljesítményriportok)
├── task/          # Az eredeti feladatkiírás
├── LICENSE
├── README.md      # Dokumentáció (angol)
└── README.hu.md   # Dokumentáció (magyar)
```

A struktúra az alkalmazás kódjával, a konténeres környezettel, az értékelő készlettel és a terheléses teszt szkriptjeivel bővül.

## Licenc

A projekt [MIT licenc](LICENSE) alatt érhető el. © 2026 Csaba Ovari

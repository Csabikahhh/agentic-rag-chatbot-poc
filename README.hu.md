# Agentic RAG Chatbot – Proof of Concept

[English](README.md) | **Magyar**

Agentic RAG (Retrieval-Augmented Generation) alapú chatbot prototípus Pythonban – [LangGraph](https://github.com/langchain-ai/langgraph) frameworkkel, helyben futó, nyílt forráskódú LLM-mel és [Streamlit](https://streamlit.io/) felülettel, Dockerrel teljesen konténerizálva.

> **Állapot:** 🚧 Fejlesztés alatt. Az alapok elkészültek: az [1. fázis](docs/project-structure-plan.hu.md#8-felépítési-sorrend) váza (csomag, parancssori felület, konfiguráció, tesztek, lint), a közös infrastruktúra (beállítások, az LLM- és az embedding-factory offline fake változatokkal, lépésnyomkövetés, az állapotsémák, a Streamlit felület váza) és a konténeres környezet (`Dockerfile`, `compose.yaml`). Az ágens, a RAG algráf, az adatbetöltés, valamint az értékelés és a terheléses teszt futtatója egyelőre típusannotált váz: mindegyik jelzi, melyik fázisban készül el. A domain eldőlt (a [projektstruktúra-terv](docs/project-structure-plan.hu.md) 8–9. döntése): [frontend fejlesztői asszisztens](#problémafelvetés-és-motiváció) az MDN, a React, a Vue, a Next.js, a Nuxt és a TypeScript hivatalos dokumentációja felett, három nem visszakeresési eszközzel. Következik a 2. fázis (a korpusz letöltése, adatbetöltés és index). A *Kitöltendő* jelölésű részek a megvalósítás előrehaladtával egészülnek ki.

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

A projekt egy Medior AI Engineer pozícióra kiírt technikai feladat megoldása; az eredeti feladatkiírás a helyi, gitignore-olt `task/` mappában található.

## A feladat követelményei

A feladatkiírás egyes követelményeinek állapota:

**Probléma és adatforrás**

- [x] Valós probléma (domain / use case) választása, írásos indoklással
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

**Use case: frontend fejlesztői asszisztens.** A chatbot a hivatalos dokumentáció alapján válaszol a webes frontend fejlesztés kérdéseire: HTML, CSS, JavaScript és TypeScript, akadálymentesség, valamint a React, a Vue, a Next.js és a Nuxt keretrendszer. A konkrét tényeket determinisztikus eszközök ellenőrzik: a böngészőtámogatást, a színkontrasztot és a CSS specificitást. Az üzemeltetési bővítés (a Kubernetes és a Docker dokumentációja, manifest- és ütemezés-ellenőrzéssel) akkor következik, amikor a frontend rész már működik; ez forrásokat és eszközöket ad hozzá, nem új architektúrát.

- **Miért releváns a probléma?** A frontend tudás sok, gyorsan változó forrásban van szétszórva: a webplatform referenciájában (MDN), az egyes keretrendszerek és verzióik dokumentációjában, valamint az akadálymentességi irányelvekben. Az általános LLM-ek az ilyen kérdésekre gördülékenyen, de megbízhatatlanul válaszolnak. Kitalálnak API-kat, összekeverik a keretrendszer-verziókat (a Next.js Pages és App Routerét, a Nuxt 2-t és a mostani Nuxtot), valamint az azonos nevű API-kat (a React `useState` hookját és a Nuxt `useState` composable-jét), és nem tudják megmondani, hogy egy funkció működik-e azokban a böngészőkben, amelyeket a projektnek támogatnia kell.
- **Milyen felhasználói igényt elégít ki?** A fejlesztőknek rövid, helyes, forrásra hivatkozó válasz kell, és pontos eredmény azokra a kérdésekre, amelyekre van ilyen: *működik-e a `:has()` Safari 15-ben?*, *megfelel-e ez a szürke szöveg a WCAG AA szintjének?*, *miért nem érvényesül ez a szabály?* Az asszisztens a hivatalos dokumentáció rögzített verzióiból válaszol, minden állítását forrással támasztja alá, és jelzi, ha a dokumentáció nem fedi le a kérdést.
- **Miért előnyös rá az agentic RAG megközelítés?** A valódi kérdések magyarázatot és ellenőrzést kevernek, és gyakran több keretrendszert érintenek, ezért egyetlen „visszakeresés, majd generálás” lépés nem elég:
  - egy olyan kérés, mint a *Hogyan kérek le adatot szerveroldalon Next.js-ben és Nuxtban, és mi a különbség?*, keretrendszerenként egy-egy visszakeresésre bomlik, ezek párhuzamosan futnak, és egyetlen összehasonlításban állnak össze;
  - amit ki lehet számolni, azt egy eszköz számolja ki, nem a modell találgatja: a `#777777` színű szöveg kontrasztaránya fehér háttéren 4,48:1, éppen az AA szinthez szükséges 4,5:1 alatt, és ezt a különbséget egy modell könnyen eltéveszti;
  - az ellenőrző lépés a visszakeresett dokumentációhoz méri a választervezetet, és újratervez, ha a tervezet nincs alátámasztva; így a kitalált API-k még azelőtt kiszűrődnek, hogy a felhasználóhoz érnének.

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

A részletes céltervezés (a hét fő node és a routing, a RAG algráf négy lépése, az eszközök), az állapotsémák a kód jelenlegi állapota szerint, a modulokon átívelő szerződések (függőséginjektálás, végrehajtási modell, trace-események, az újratervezési ciklus, a hivatkozások számozása, hibák és kilépési kódok), valamint a konfigurációs referencia a [docs/architecture.md](docs/architecture.md) fájlban található (angolul).

> 🚧 *Kitöltendő:* a fő workflow node-jai és routing logikája, a RAG algráf lépései, az eszközök, az állapot (state) sémája és az adatbetöltési (ingestion) folyamat.
>
> Tipp: az `uv run agentic-rag export-graph` Mermaid formátumban kiírja a lefordított (compiled) gráfokat (a `graph.get_graph(xray=True).draw_mermaid()` segítségével), amint a 3–4. fázisban elkészülnek. A RAG algráf külön diagramot kap: a fő workflow a `run_rag_subtask` node-on belül, a `search_knowledge_base` eszközön keresztül hívja, ahol az `xray=True` nem bontja ki.

## Tervezési döntések

A [projektstruktúra-terv](docs/project-structure-plan.hu.md#3-a-váz-elkészítése-előtt-eldöntendő-kérdések) 1–7. döntése beépült a kódba. A 8–9. döntés, a domain és a nem visszakeresési eszközök, 2026. 10. 02-án született meg; a 2. és a 4. fázisban épülnek be. Az LLM és az embedding modell (4. és 5. döntés) ideiglenes alapértelmezés, amíg az értékelés és a terheléses teszt meg nem méri őket, a darabolást pedig a 2. fázisban hangoljuk.

| Terület | Fő szempontok (trade-offok) | Választás és indoklás |
|---|---|---|
| Domain és adatforrás | Relevancia, elérhetőség és licenc, előfeldolgozási igény | **Frontend fejlesztői asszisztens** a hivatalos dokumentáció felett (2. fázis): MDN Web Docs (válogatott rész a CSS-ről, a HTML-ről, az akadálymentességről és a JavaScriptről; a szöveg CC BY-SA 2.5, a kódpéldák CC0), React (CC BY 4.0), Vue (CC BY 4.0), Next.js (MIT), Nuxt (MIT) és a TypeScript Handbook (CC BY 4.0). Az `agentic-rag ingest --download` rögzített commitokról tölti le őket, a `data/sources.yaml` listája alapján: minden futás ugyanazokat a verziókat indexeli, és a repositoryba nem kerül share-alike licencű szöveg. A dokumentáció verziózott, strukturált, és azokat a kérdéseket fedi le, amelyeket a fejlesztők ténylegesen feltesznek. Az üzemeltetési bővítés (Kubernetes, CC BY 4.0; Docker, Apache 2.0) két újabb bejegyzés ugyanebben a listában |
| Nem visszakeresési eszközök | Illeszkedés a domainhez; determinisztikus, helyi és tesztelhető | **Három eszköz** (4. fázis): a *böngészőtámogatás* az MDN `browser-compat-data` adatbázisában (CC0, rögzített kiadás) keres meg egy funkciót, és a verzióit a célböngészőkhöz hasonlítja; a *színkontraszt* kiszámolja két szín WCAG 2.x szerinti kontrasztarányát, és hogy megfelel-e az AA és az AAA szintnek normál és nagy szövegméretnél; a *CSS specificitás* a Selectors Level 4 szabályai szerint kiszámolja a szelektorok specificitását, és megmondja, melyik érvényesül. Mindegyik számítás vagy rögzített adatban való keresés, ezért pontosan tesztelhető, és olyan tényeket ad a modellnek, amelyeket az különben találgatna |
| Csomagkezelés és Python-verzió | Reprodukálható build, az ML stack wheel-lefedettsége, beüzemelési igény | **uv** (`pyproject.toml` + `uv.lock`), **Python 3.12**, `src/` elrendezés: a lock fájl minden csomagot rögzít a helyi futtatáshoz és a képfájlhoz egyaránt, az uv maga telepíti a rögzített Pythont, és a 3.12-höz érhető el a legszélesebb wheel-lefedettség a torch és a chromadb számára |
| LLM | Válaszminőség vs. válaszidő vs. memóriaigény (RAM/VRAM); eszközhívás (tool calling) támogatása; licenc | **`qwen2.5:7b-instruct`**, ideiglenesen: többnyelvű, Apache 2.0 licencű 7B-s instruct modell, amelynek 4 bites változata (kb. 4,7 GB) elfér a fejlesztői gép 8 GB-os VRAM-jában; az értékelés és a terheléses teszt alapján véglegesítjük vagy cseréljük |
| LLM kiszolgálás | Beüzemelési igény, konténerizálhatóság, áteresztőképesség | **Ollama** (Compose szolgáltatásként vagy a gépen futtatva) és egy **szkriptelt fake provider**: az Ollama HTTP API-t és GPU-támogatást ad anélkül, hogy bármit a képfájlba kellene fordítani; a fake (`LLM_PROVIDER=fake`) a feladatkiírás szerinti dummy LLM, és modell nélkül tartja a teszteket |
| Eszközhívás módja | Megbízhatóság kis helyi modellekkel vs. a natív eszközhívás rugalmassága | **Strukturált kimenetű tervező + explicit eszköz-node-ok**: a tervező tipizált részfeladatokat ad vissza JSON-ként, amit a kis helyi modellek megbízhatóbban állítanak elő, mint a natív eszközhívást; az eszközök LangChain toolok maradnak, így a `bind_tools` később is lehetséges |
| Embedding modell | Visszakeresési minőség vs. sebesség; nyelvi lefedettség | **`intfloat/multilingual-e5-small`**, ideiglenesen, helyben, sentence-transformers-szel futtatva: többnyelvű (a magyart is lefedi) és kicsi (384 dimenzió), így CPU-n fut, a GPU pedig az LLM-é marad |
| Vektoradatbázis | Perzisztencia, metaadat-alapú szűrés, skálázhatóság | **Chroma** perzisztens klienssel a `data/chroma_db/` könyvtárban (Compose-ban nevesített volume): perzisztencia és metaadat-alapú szűrés pickle-deszerializálás nélkül |
| Darabolás (chunking) | Chunkméret és átfedés vs. visszakeresési pontosság és kontextushossz | *A 2. fázisban:* a Markdown szöveget a H2 és H3 fejléceinél vágjuk, a fejlécláncot `section` metaadatként megtartjuk, a kódblokkokat soha nem vágjuk ketté, és csak a túl hosszú szakaszokat daraboljuk tovább a kód 900 karakteres chunkjaival és 150 karakteres átfedésével; az értékelő készleten hangolva |

Megjegyzések az ideiglenes alapértelmezésekhez:

- **Modellek.** Mindkét modell alapértelmezés, nem végleges választás (a terv 4. és 5. döntése): az értékelés és a terheléses teszt mérései alapján véglegesítjük vagy cseréljük őket. A dokumentáció angol, a kérdések lehetnek magyarok vagy angolok, ezért az értékelő készlet magyar kérdéseket is tartalmaz az angol korpusz felett: ez a többnyelvű E5 modell nyelvek közötti visszakeresését és a 7B-s modell magyar válaszait egyaránt ellenőrzi.
- **A korpusz terjedelme.** A keretrendszerek dokumentációja verziókat és elavult részeket is kever. A 2. fázis a jelenlegi útmutatókat és API-referenciákat tartja meg, például a Next.js App Routerét, és kihagyja a Pages Routert, a Nuxt Bridge-et és a migrációs útmutatókat. Minden forrás saját könyvtárba kerül a `data/raw/` alatt (`mdn/`, `react/`, `vue/`, `nextjs/`, `nuxt/`, `typescript/`), így minden hivatkozásból látszik, melyik dokumentációból származik, és az azonos nevű API-k nem keverednek.
- **Embedding.** A fizetős API-k tilalma kizárja a hosztolt embedding API-kat, ezért az embedding helyben fut. Az alapértelmezett modell egyszer töltődik le (kb. 0,5 GB, a Hugging Face cache-be, `HF_HOME`), utána offline működik; néhány másodperc alatt töltődik be, az E5 modellek `query:` / `passage:` előtagjait a kód automatikusan hozzáadja, és CPU-s torch-ot visz a képfájlba (a 2,9 GB-os képfájlból kb. 0,8 GB-ot). Az `EMBEDDING_PROVIDER=fake` determinisztikus, hash-elt szózsák- (bag-of-words) vektorokkal helyettesíti: offline és azonnali, de tisztán lexikális, ezért csak tesztekhez és modell nélküli bemutatókhoz való.
- **Az index újraépítése.** A különböző modellek vektorai nem összehasonlíthatók: az `EMBEDDING_PROVIDER` vagy az `EMBEDDING_MODEL` módosítása után az indexet újra kell építeni az `agentic-rag ingest --rebuild` paranccsal.

## Értékelés

### Funkcionális értékelés

**Megközelítés:** 10–20 domainspecifikus kérdésből álló mini értékelő készlet – mindegyikhez referenciaválasz és, ahol releváns, az elvárt forrásdokumentumok –, amellyel egy kiválasztott node vagy a teljes agentic workflow értékelhető.

**Lehetséges metrikák:** a válasz helyessége a referenciához képest, hűség a visszakeresett kontextushoz (faithfulness), visszakeresési találati arány (hit rate@k), valamint a routing és az eszközválasztás pontossága.

**Már megvan:** az értékelő készlet formátuma (`data/eval/questions.jsonl`, kérdésenként egy JSON-objektum, amelyet az `agentic_rag.evaluation.dataset` ellenőriz) a [data/eval/README.md](data/eval/README.md) fájlban van leírva (angolul). Minden kérdés felsorolja az elvárt dokumentumait (`expected_documents`): a `DATA_DIR`-hez viszonyított, perjeles útvonalakat, ahogy az adatbetöltés azonosítja a dokumentumokat. A visszakeresési hit@k-t visszakeresési részfeladatonként számoljuk, a részfeladat chunkjainak rangsorolt dokumentumaiból (a riportban `retrieved_documents`, részfeladatonként egy lista), sosem a válasz deduplikált hivatkozásaiból. Az `eval --target node` azon node-ok egyikét értékeli, amelyeket egy kérdés önmagában meg tud hajtani (`NODE_TARGETS`: az `analyze_request` a routinghoz, a `run_rag_subtask` a visszakereséshez); más node-ot 2-es kilépési kóddal elutasít. A visszakeresési hit@k és a routing pontossága elkészült; az LLM által pontozott helyesség és faithfulness, a futtató és maguk a kérdések a 7. fázisban következnek. Az eredmények JSON-ként a `data/eval/results/` mappába kerülnek majd.

> 🚧 *Kitöltendő:* az értékelő készlet helye, a pontozás módja, az eredmények, a levont következtetések és a reprodukálásukhoz szükséges parancs.

### Terheléses teszt és a szűk keresztmetszet elemzése

**Forgatókönyv:** 50–200 lekérdezés a futó rendszeren, a párhuzamossági szint, a lekérdezések összetétele és a hardver dokumentálásával.

**Mért értékek:** válaszidő (átlag, p50, p95, p99, maximum), áteresztőképesség és hibaarány, valamint node-onkénti válaszidő-bontás a fő szűk keresztmetszet azonosításához – ezt 1–2 konkrét optimalizálási javaslat követi.

**Már megvan:** a válaszidő-statisztikák és a riport formátuma (`agentic_rag.loadtest.runner`). A percentiliseket a legközelebbi rangok közötti lineáris interpoláció adja (a `numpy.percentile` alapértelmezése), a bemelegítő kérések külön szerepelnek, és a node-onkénti részesedésnél a `run_rag_subtask` idejét nem szabad összeadni az általa futtatott RAG algráf node-okéval, mert azokat már tartalmazza. A futtató a 8. fázisban következik: egyszer építi fel a gráfot, és egy `ThreadPoolExecutor(max_workers=concurrency)` szálain hívja a `graph.invoke`-ot, ugyanazon a szinkron úton, amelyet a UI és az értékelés is használ.

> 🚧 *Kitöltendő:* az eredmények, a szűk keresztmetszet elemzése, az optimalizálási javaslatok és a reprodukálásukhoz szükséges parancs.

## Telepítés és futtatás

### Előfeltételek

- Git.
- Helyi fejlesztéshez [uv](https://docs.astral.sh/uv/getting-started/installation/); a Python 3.12-t is telepíti, ha hiányzik.
- A konténerekhez Docker és Docker Compose 2.24 vagy újabb (a `compose.yaml` az opcionális `env_file` szintaxist használja).
- Valódi válaszokhoz helyi LLM, amelyet az [Ollama](https://ollama.com/) szolgál ki: a Compose szolgáltatás vagy a gépre telepített Ollama. Az ideiglenes alapértelmezett modell, a `qwen2.5:7b-instruct` (4 bites, kb. 4,7 GB) elfér egy 8 GB-os GPU-n, és CPU-n is fut, lassabban (*a pontos RAM/VRAM-igény még nincs meghatározva*). A fake módhoz nem kell sem modell, sem GPU.
- Lemezterület a teljes stackhez: az alkalmazás képfájlja (kb. 2,9 GB; mérve 2,88 GB), az Ollama képfájlja és a chatmodell.

### Ami már most működik

Az alapok végponttól végpontig futnak, de kérdésekre még nem válaszolnak:

- a tesztek offline, a fake LLM-mel és a fake embeddinggel átmennek;
- a parancssori felület kilistázza a parancsait, a `config` kiírja az érvényes beállításokat; az `ingest`, az `eval`, a `loadtest` és az `export-graph` kiírja, melyik fázisra van tervezve (`… is planned for Phase N (see docs/project-structure-plan.md, section 8)`), és 1-es kilépési kóddal áll le;
- a Streamlit UI elindul, megjeleníti a konfigurációt, és minden kérdésre azzal az üzenettel felel, hogy az ágens a 4. fázisban készül el. A felhasználó által megállított futás a *Stopped before an answer was produced.* üzenetet kapja, az ágens az új kérdés mellett csak a korábbi megválaszolt kérdéseket kapja meg, a válaszok `$` jelei szövegként jelennek meg (LaTeX nélkül), érvénytelen beállítás vagy olvashatatlan `.env` esetén pedig a chat helyén *Invalid configuration* hiba áll;
- a képfájl felépül, és az `app` szolgáltatás fake módban egészséges (healthy) állapotban indul.

### Helyi fejlesztés uv-vel

Az uv telepítése (további lehetőségek a [telepítési útmutatóban](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # Linux és macOS
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

A repository klónozása és beállítása:

```bash
git clone https://github.com/Csabikahhh/agentic-rag-chatbot-poc.git
cd agentic-rag-chatbot-poc
uv sync --locked                                 # .venv Python 3.12-vel, a rögzített függőségekkel és a fejlesztői eszközökkel
uv run pytest                                    # offline tesztek; a végén: "... passed, 1 deselected"
uv run ruff check .                              # lint
uv run ruff format --check .                     # formázás
uv run python -m agentic_rag --help              # a parancsok (vagy: uv run agentic-rag --help)
uv run python -m agentic_rag config              # az érvényes beállítások
uv run streamlit run src/agentic_rag/ui/app.py   # a UI a http://localhost:8501 címen
```

**Fake módban** az alkalmazás Ollama és modell-letöltés nélkül fut: a szkriptelt fake LLM-mel (`LLM_PROVIDER=fake`) és az offline, hash-alapú embeddinggel (`EMBEDDING_PROVIDER=fake`). A két változó a shellben vagy a `.env` fájlban állítható be (lásd: [Konfiguráció](#konfiguráció)):

```bash
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake uv run streamlit run src/agentic_rag/ui/app.py
```

```powershell
$env:LLM_PROVIDER="fake"; $env:EMBEDDING_PROVIDER="fake"; uv run streamlit run src/agentic_rag/ui/app.py
```

Az `uv sync --locked` hibával leáll, ahelyett hogy átírná az `uv.lock` fájlt, ha a lock fájl nem egyezik a `pyproject.toml`-lal; a képfájl buildje ugyanezt az ellenőrzést használja.

A tesztek fake módban futnak, és figyelmen kívül hagyják a shell beállításait és a `.env` fájlt. Az egyetlen kivétel az élő Ollama-teszt (`ollama` marker): a sima `uv run pytest` kihagyja (deselect), ezért az összesítés `552 passed, 1 deselected` (2026. 10. 01-jén mérve; a sikeres tesztek száma a fázisokkal nő). Az `uv run pytest -m ollama` futtatja, ahogy lent látható.

**A gépen futó Ollama** a leggyorsabb fejlesztési kör valódi modellel. Az [Ollama](https://ollama.com/download) telepítése és elindítása (az asztali alkalmazással vagy az `ollama serve` paranccsal) után le kell tölteni a modellt; az alapértelmezett `OLLAMA_BASE_URL` (`http://localhost:11434`) eléri:

```bash
ollama pull qwen2.5:7b-instruct
uv run pytest -m ollama       # élő ellenőrzés a helyi szerverrel; kimarad, ha a szerver nem érhető el
```

Az élő teszt a shell `OLLAMA_BASE_URL` és `OLLAMA_MODEL` értékét követi, így másik szervert vagy modellt is ellenőrizhet.

### Futtatás Docker Compose-zal

A feladatkiírás `docker-compose.yml`-t kér; a repository ezt `compose.yaml` néven tartalmazza, ahogy a Compose dokumentációja ajánlja, és a `docker compose` automatikusan felismeri.

| Szolgáltatás | Képfájl | Szerep |
|---|---|---|
| `app` | A `Dockerfile` alapján épül | Streamlit UI a <http://localhost:8501> címen (csak a 127.0.0.1-en publikálva); nem root felhasználóként fut |
| `ollama` | `ollama/ollama:0.35.0` | Az LLM kiszolgálója; a stacken belül a `http://ollama:11434` címen érhető el, a gépre nincs publikálva |
| `ollama-pull` | `ollama/ollama:0.35.0` | Egyszeri futás: letölti az `OLLAMA_MODEL` modellt, ha az `ollama-data` volume-ban még nincs meg |

A korpusz (`./data/raw`) csak olvashatóan van csatolva. Az `ollama-data` (Ollama modellek), a `chroma-data` (vektorindex) és a `hf-cache` (Hugging Face modellek) nevesített volume-ok az újraépítések között is megőrzik a letöltéseket és az indexet.

**Teljes stack:**

```bash
docker compose up --build
```

Az első indítás felépíti a képfájlt (néhány perc), és letölti az Ollama képfájlját és a chatmodellt (több GB); a UI a modell letöltése után indul. Az embedding modell az első használatkor töltődik le a `hf-cache` volume-ba (a 2. fázistól). A későbbi indítások újrahasznosítják a volume-okat.

**Fake mód** (csak az `app` szolgáltatás, Ollama és modell-letöltés nélkül):

```bash
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake docker compose up --build --no-deps app
```

```powershell
$env:LLM_PROVIDER="fake"; $env:EMBEDDING_PROVIDER="fake"; docker compose up --build --no-deps app
```

A `--no-deps` kihagyja a két Ollama szolgáltatást. PowerShellben a változók a munkamenet végéig beállítva maradnak; a `.env` fájlban is megadhatók.

**NVIDIA GPU az Ollamához** (opcionális override fájl):

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

NVIDIA driver és Docker GPU-támogatás kell hozzá: Windowson Docker Desktop WSL 2 backenddel, Linuxon az NVIDIA Container Toolkit. Alapértelmezetté a `.env` fájlban megadott `COMPOSE_FILE=compose.yaml:compose.gpu.yaml` beállítással tehető (Windowson `;` az elválasztó). Az override nélkül az Ollama CPU-n fut.

**A gépen futó Ollama** az `ollama` szolgáltatás helyett:

```bash
docker compose run --rm --no-deps --service-ports -e OLLAMA_BASE_URL=http://host.docker.internal:11434 app
```

**Parancssori parancsok a konténerben:**

```bash
docker compose run --rm --no-deps app agentic-rag config
```

Az `eval` és a `loadtest` a gépen futtatandó (`uv run agentic-rag eval`, `uv run agentic-rag loadtest`); ez az ajánlott út. A stack csak a `data/raw` mappát csatolja, ezért a konténerben futtatásukhoz egy további `./data/eval:/app/data/eval` bind mount kell, és a riportok csak akkor íródnak ki, ha a konténer felhasználója írhatja a `data/eval/results` mappát.

**Linuxos gépek és a 10001-es UID.** Az `app` konténer 10001-es UID-dal és GID-dal fut. A bind mount megtartja a gépen lévő könyvtár tulajdonosát, ezért Linuxos Docker Engine-en az alkalmazás csak akkor írhat egy bind mountba, ha a könyvtár a 10001-es UID számára írható; a Windowsos és macOS-es Docker Desktop ezt elfedi, mert a bind mountokat mindenki számára írhatónak mutatja. Vagy írhatóvá kell tenni a könyvtárat a 10001-es UID számára, vagy a saját azonosítóinkkal kell felépíteni a képfájlt az `APP_UID` és `APP_GID` build argumentumokkal:

```bash
APP_UID=$(id -u) APP_GID=$(id -g) docker compose up --build
```

A nevesített volume-ok (`chroma-data`, `hf-cache`) csak addig veszik át a tulajdonosukat a képfájlból, amíg üresek. Az azonosítók módosítása után ezért vagy helyben kell átállítani a tulajdonosukat (a parancs a [`compose.yaml`](compose.yaml) `app` szolgáltatásánál, a megjegyzésben található), vagy újra kell létrehozni őket a `docker compose down -v` paranccsal, amely az indexet és a letöltött modelleket is törli.

**A képfájl önmagában** (a kötelező `Dockerfile`, Compose nélkül):

```bash
docker build -t agentic-rag-chatbot:dev .
docker run --rm -p 127.0.0.1:8501:8501 --mount type=bind,source=./data/raw,target=/app/data/raw,readonly -e LLM_PROVIDER=fake -e EMBEDDING_PROVIDER=fake agentic-rag-chatbot:dev
```

A `--mount` alak változatlanul jut el a Dockerhez Git Bashből, PowerShellből és POSIX shellekből is; a Git Bash a rövid `-v ./data/raw:/app/data/raw:ro` alakot Windows-útvonallá írná át, és a korpusz rossz helyre, írhatóan kerülne. Sima `docker build` esetén más azonosítókhoz a `--build-arg APP_UID=... --build-arg APP_GID=...` kapcsolók adhatók meg.

**A képfájl rétegei.** A `Dockerfile` két lépcsős. A `deps` lépcső csak az `uv.lock`-ban rögzített függőségeket telepíti (`uv sync --locked --no-dev --no-install-project`); a `--locked` leállítja a buildet, ha az `uv.lock` nem egyezik a `pyproject.toml`-lal. A futtató lépcső két, a kódtól független rétegben átmásolja ezt a virtuális környezetet (1,71 GB), és lefordítja a bytecode-ját (415 MB), majd hozzáadja az `src/` mappát (348 kB) és a projekt kis, szerkeszthető (editable) telepítését (115 kB). Az `src/` módosítása ezért csak a két kis réteget építi újra: mérve 7 s, szemben a szétválasztás előtti kb. 53 s-mal és egy új, 2,11 GB-os réteggel. A képfájl 2,88 GB (`python:3.12.14-slim-trixie`, csak CPU-s torch); nincs benne uv, buildfájl és fejlesztői függőség, a kód és a függőségek pedig root tulajdonúak, az alkalmazás felhasználója számára csak olvashatók.

**Leállítás és takarítás:**

```bash
docker compose down      # törli a konténereket és a hálózatot, a volume-ok megmaradnak
docker compose down -v   # a volume-okat is törli
```

> **Figyelem:** a `docker compose down -v` törli a letöltött modelleket (`ollama-data`, `hf-cache`) és a vektorindexet (`chroma-data`); a következő indítás újra letölti, illetve felépíti őket.

További lehetőségek, például az Ollama API publikálása a gépre egy helyi `compose.override.yaml` fájllal, a [`compose.yaml`](compose.yaml) fejlécében olvashatók.

> Eddig ellenőrizve: a képfájl buildje (beállított `APP_UID`/`APP_GID` értékkel is, valamint a `--locked` hibája elavult lock fájl esetén), a rétegek újrahasznosítása az `src/` módosítása után, mindkét Compose konfiguráció, az `app` szolgáltatás fake módban (healthy állapot, a UI a 8501-es porton), a `docker run --mount` parancs Git Bashből, valamint egy 1000-es UID tulajdonában lévő, szimulált linuxos bind mount. Még nem futott: a teljes stack az Ollama szolgáltatásokkal (modell-letöltés, GPU-átadás), natív Linux gép és macOS.

### Konfiguráció

Minden beállítás környezeti változó, amelyet az `agentic_rag.config.Settings` olvas be. A [`.env.example`](.env.example) minden változót felsorol az alapértékével és egy megjegyzéssel; `.env` néven (ez gitignore-olva van) lemásolva a szükséges értékek módosíthatók:

```bash
cp .env.example .env    # PowerShell: Copy-Item .env.example .env
```

A valódi környezeti változók elsőbbséget élveznek a `.env`-del szemben, az üres érték (`KEY=`) pedig az alapértéket jelenti. A `.env` legyen UTF-8 kódolású: a Windows PowerShell 5.1 a `>` operátorral és az `Out-File` paranccsal UTF-16-ot ír, ezért a fájlt a fenti `Copy-Item` paranccsal érdemes lemásolni. Az `uv run agentic-rag config` kiírja az érvényes értékeket; érvénytelen érték, illetve olvashatatlan vagy nem UTF-8 kódolású `.env` esetén a parancssori felület 2-es kilépési kóddal leáll, és megnevezi a problémát, a UI pedig a chat helyén mutatja.

| Változó | Alapértelmezés | Cél |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, vagy `fake` a szkriptelt offline modellhez |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Az Ollama szerver címe |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Az Ollama chatmodell tagje (ideiglenes) |
| `OLLAMA_NUM_CTX` | `8192` | A kontextusablak tokenben, 512–131072, az Ollama `num_ctx` paramétereként elküldve; a prompt és a válasz osztozik rajta, a hosszabb promptot az Ollama szó nélkül levágja |
| `OLLAMA_TIMEOUT_S` | `120.0` | Az egyes Ollama-kérések HTTP-időkorlátja másodpercben, 0-nál nagyobb |
| `LLM_TEMPERATURE` | `0.0` | Mintavételi hőmérséklet, 0,0–2,0 |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface`, vagy `fake` az offline, hash-alapú embeddinghez |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Hugging Face embedding modell (ideiglenes) |
| `DATA_DIR` | `data/raw` | A korpusz könyvtára |
| `CHROMA_DIR` | `data/chroma_db` | A vektorindex könyvtára |
| `CHROMA_COLLECTION` | `documents` | A Chroma kollekció neve: 3–63 karakter (szándékos projektszintű korlát), nem lehet IPv4-cím |
| `TOP_K` | `4` | Lekérdezésenként visszakeresett chunkok száma |
| `MAX_RETRIES` | `2` | Az ellenőrzés → újratervezés ciklus korlátja |
| `INGEST_ON_START` | `true` | Induláskor felépíti az indexet, ha hiányzik (a 6. fázisig nincs hatása) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` vagy `ERROR` |

A Compose stackben a `compose.yaml` az `app` szolgáltatásnak az `OLLAMA_BASE_URL=http://ollama:11434` értéket adja, így a `.env`-ben megadott érték ezt nem változtatja meg; az `LLM_PROVIDER`, az `EMBEDDING_PROVIDER` és az `OLLAMA_MODEL` értékét pedig a shellből vagy a `.env`-ből veszi át. Minden más változó csak a `.env` fájlon keresztül jut be a konténerbe. A teljes referencia az érvényességi szabályokkal a [docs/architecture.md](docs/architecture.md#configuration-reference) fájlban található (angolul).

### Parancssori felület

`agentic-rag <parancs>` (vagy `python -m agentic_rag <parancs>`); a `--help` minden parancs kapcsolóit megmutatja.

| Parancs | Cél | Elérhető |
|---|---|---|
| `config` | Az érvényes beállítások kiírása `KEY=value` sorokként | Most |
| `ingest [--rebuild]` | A vektorindex felépítése a `DATA_DIR` tartalmából | 2. fázis |
| `export-graph [--graph {all,agent,rag}] [--format {markdown,mermaid}] [--output PATH]` | A lefordított gráfok Mermaid diagramjai | 3–4. fázis |
| `eval [--target {graph,node}] [--node NAME] [--dataset PATH] [--output-dir PATH]` | Funkcionális értékelés | 7. fázis |
| `loadtest [--requests N] [--concurrency C] [--warmup W] [--output-dir PATH]` | Terheléses teszt a lefordított gráfon | 8. fázis |

Kilépési kódok:

- 0 siker esetén;
- 1, ha a parancs hibára futott: egy későbbi fázisra tervezett funkció (`PlannedFeatureError`) csak az üzenetét írja ki, minden más hiba a traceback-jét;
- 2 használati és konfigurációs hibák esetén: érvénytelen kapcsolók (ide tartozik az `InvalidArgumentError` is, például egy `NODE_TARGETS`-en kívüli `eval --node`), érvénytelen beállítások, vagy `ConfigurationError` (olvashatatlan vagy nem UTF-8 kódolású `.env`);
- 130 megszakításkor.

A részletek a [docs/architecture.md](docs/architecture.md#errors-and-exit-codes) fájlban találhatók (angolul).

> 🚧 *Kitöltendő:* az adatbetöltés (2. fázis), valamint az értékelés és a terheléses teszt futtatása (7–8. fázis).

## Projektstruktúra

```text
agentic-rag-chatbot-poc/
├── .claude/                        # a fejlesztéshez használt Claude Code ágensek és skillek
├── data/
│   ├── README.md                   # az adatok elrendezése, a korpusz szabályai, az index újraépítése
│   ├── raw/                        # forráskorpusz (DATA_DIR); a 2. fázisig csak .gitkeep
│   └── eval/
│       ├── README.md               # az értékelő készlet sémája és a riportok formátuma
│       └── results/                # commitolt értékelési és terheléses riportok; a 7. fázisig csak .gitkeep
├── docs/
│   ├── architecture.md             # célgráfok, állapot- és átívelő szerződések, konfigurációs referencia (angol)
│   ├── project-structure-plan.md   # a repository terve és felépítési sorrendje (angol)
│   └── project-structure-plan.hu.md  # ugyanez magyarul
├── src/
│   └── agentic_rag/
│       ├── __init__.py             # a csomag verziója
│       ├── __main__.py             # `python -m agentic_rag`
│       ├── cli.py                  # parancsok: ingest · eval · loadtest · export-graph · config
│       ├── config.py               # Settings környezeti változókból és .env-ből; a naplózás beállítása
│       ├── errors.py               # PlannedFeatureError, ConfigurationError, InvalidArgumentError, planned()
│       ├── llm.py                  # chatmodell factory: Ollama vagy a szkriptelt fake
│       ├── embeddings.py           # embedding factory: sentence-transformers vagy offline, hash-alapú fake
│       ├── tracing.py              # TraceEvent és a @traced node-dekorátor
│       ├── reports.py              # RESULTS_DIR; RunReport, az EvalReport és a LoadTestReport alapja
│       ├── agent/                  # fő agentic workflow (a 4. fázisig váz)
│       │   ├── types.py            # Intent, Verdict, SubtaskKind, LangGraph nélkül
│       │   ├── state.py            # AgentState, Subtask, SubtaskResult (kész sémák)
│       │   ├── nodes.py            # a hét node függvénye
│       │   ├── routing.py          # feltételes élek és a Send szétosztás
│       │   ├── tools.py            # search_knowledge_base és a nem visszakeresési eszköz helye
│       │   └── graph.py            # NODE_NAMES és build_agent_graph()
│       ├── rag/                    # RAG algráf (a 3. fázisig váz)
│       │   ├── state.py            # RagState, RagInput, RagOutput, Source (kész sémák)
│       │   ├── nodes.py            # rewrite_query · retrieve · grade_documents · build_context
│       │   └── graph.py            # RAG_NODE_NAMES és build_rag_graph()
│       ├── ingestion/              # adatbetöltés (a 2. fázisig váz)
│       │   ├── loaders.py          # korpuszfájlok → Document-ek hivatkozási metaadatokkal
│       │   ├── chunking.py         # darabolási beállítások (kiindulás: 900/150 karakter)
│       │   └── index.py            # a Chroma index felépítése és megnyitása; IndexStats
│       ├── evaluation/
│       │   ├── dataset.py          # EvalItem és a questions.jsonl betöltője
│       │   ├── metrics.py          # hit@k és routing-pontosság; LLM-mel pontozott metrikák a 7. fázisban
│       │   └── runner.py           # riportmodellek és NODE_TARGETS; run_evaluation() a 7. fázisban
│       ├── loadtest/
│       │   └── runner.py           # válaszidő-statisztikák és riportmodell; run_load_test() a 8. fázisban
│       └── ui/
│           ├── app.py              # Streamlit belépési pont
│           └── components.py       # lépéspanel, visszakeresett kontextus panel, beállítás-összefoglaló
├── task/                           # a feladatkiírás; csak helyben, gitignore-olva
├── tests/                          # offline pytest tesztek (fake providerekkel)
│   ├── conftest.py                 # távol tartja a shellt és a .env-et a tesztektől; `settings` fixture
│   ├── test_cli.py                 # parancsok, kapcsolók és kilépési kódok
│   ├── test_config.py              # alapértékek, környezeti változók és .env, ellenőrzés, naplózás
│   ├── test_embeddings.py          # offline fake és Hugging Face ág, letöltés nélkül
│   ├── test_evaluation.py          # kérdésbetöltő, metrikák és riportmodellek
│   ├── test_llm.py                 # provider-választás, szkriptelt fake; élő Ollama-ellenőrzés (`ollama` marker, alapból kihagyva)
│   ├── test_loadtest.py            # percentilisek, válaszidő-összesítések és a riportmodell
│   ├── test_skeletons_agent.py     # a fő workflow váza: node-ok, routing, eszközök, gráf
│   ├── test_skeletons_rag.py       # az adatbetöltés és a RAG algráf váza
│   ├── test_state.py               # állapotsémák és reducerek
│   ├── test_tracing.py             # a lépésnyomkövetés elemei
│   └── test_ui.py                  # a Streamlit UI AppTest-tel
├── .dockerignore                   # a build context engedélylistája (allowlist)
├── .env.example                    # minden beállítás az alapértékével
├── .gitattributes                  # LF sorvégek minden checkoutban, Windowson is
├── .gitignore
├── .python-version                 # 3.12
├── compose.gpu.yaml                # opcionális NVIDIA GPU override az ollama szolgáltatáshoz
├── compose.yaml                    # app + ollama + egyszeri modell-letöltés
├── Dockerfile                      # deps és futtató lépcső, uv sync --locked, nem root futtatás, healthcheck
├── LICENSE
├── pyproject.toml                  # függőségek, konzolos belépési pont, ruff és pytest beállítások
├── README.md                       # dokumentáció (angol)
├── README.hu.md                    # dokumentáció (magyar)
└── uv.lock                         # rögzített függőségverziók
```

Minden csomagkönyvtárban `__init__.py` is van. A generált és csak helyi útvonalak nem szerepelnek: a `.venv/`, az eszközök cache-ei és a `data/chroma_db/` (a vektorindex, amelyet az `agentic-rag ingest` hoz létre). A [data/README.md](data/README.md) az adatok elrendezését, a korpusz szabályait és az index újraépítésének eseteit, a [data/eval/README.md](data/eval/README.md) az értékelés formátumait írja le (mindkettő angolul).

A struktúra a [felépítési sorrend](docs/project-structure-plan.hu.md#8-felépítési-sorrend) fázisaival bővül: a korpusszal, az értékelő készlettel, a `docs/` mappába kerülő értékelési és teljesítményriportokkal, valamint az egyes fázisok tesztjeivel.

## Licenc

A projekt [MIT licenc](LICENSE) alatt érhető el. © 2026 Csaba Ovari

**Harmadik féltől származó részek:** a `Dockerfile`, a `.dockerignore` és a `compose.yaml` részben a [docker/skills](https://github.com/docker/skills) `docker-project-foundations` skilljének mintafájljaiból készült (maga a skill a `.claude/skills/docker-project-foundations/` mappában található), amelyek az [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) alatt érhetők el. Mindhárom fájl a fejlécében megnevezi a forrását, és jelzi, hogy módosított változat; az átvett részeket az Apache License 2.0 feltételei szerint használjuk.

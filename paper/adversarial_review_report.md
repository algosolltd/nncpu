# Report Completo — Paper nncpu

Data: 28 agosto 2026

---

## Glossario — I concetti chiave spiegati

Prima di tutto, ecco cosa significano i termini che usiamo nel paper e in questo report:

**Prefetcher:** Un componente dentro la CPU che cerca di indovinare quali dati serviranno prossimamente e li carica in anticipo dalla memoria (lenta) alla cache (veloce). Se indovina bene, il programma va più veloce. Se indovina male, spreca banda di memoria.

**Stride predictor:** Il metodo più semplice di prefetching. Se un programma accede alla memoria con un passo regolare (es. ogni 64 byte), lo stride predictor se ne accorge e pre-carica il prossimo blocco. Funziona benissimo per i cicli `for` su array, malissimo per accessi casuali.

**MLP (Multi-Layer Perceptron):** Una piccola rete neurale. Nel nostro caso ha solo 257 parametri (numeri che "impara") — è minuscola. L'idea era: forse una rete neurale può imparare pattern che lo stride predictor non vede.

**Gate (cancello di confidenza):** Un filtro che dice "quanto sei sicuro di questa previsione?". Se la sicurezza è bassa, il gate blocca il prefetch — meglio non fare nulla che fare la cosa sbagliata. È come un buttafuori: lascia passare solo le previsioni di cui si fida.

**Matched-control (controllo a parità di condizioni):** L'idea chiave del paper. Normalmente la gente confronta "rete neurale CON gate" vs "stride predictor SENZA gate". Ma così stai confrontando due cose diverse: sia il predittore sia il filtro. Noi abbiamo messo lo STESSO gate su entrambi, così l'unica differenza è il predittore. Risultato: la rete neurale non è meglio dello stride. Era il gate a fare la differenza, non la rete neurale.

**Proxy metric vs endpoint metric:** 
- **Proxy** = misure intermedie, facili da calcolare: "quanti prefetch hai eliminato?" (35%), "quanto è migliorata l'accuratezza?" (da 11% a 15%). Sembrano numeri buoni.
- **Endpoint** = misure finali, quelle che contano davvero: "il programma va più veloce?" (IPC), "la memoria è meno occupata?" (DRAM reads). 
- Il nostro risultato sorprendente: le proxy migliorano, ma gli endpoint non cambiano quasi per niente. Eliminare il 35% dei prefetch non rende il programma più veloce. Le proxy mentono.

**IPC (Instructions Per Cycle):** Quante istruzioni la CPU completa ogni ciclo di clock. Più è alto, più il programma va veloce. È la misura finale che conta.

**DRAM reads:** Quante volte la CPU deve andare a leggere dalla memoria principale (la RAM, che è lenta). Meno sono, meglio è.

**ChampSim:** Un simulatore di CPU riconosciuto dalla comunità accademica, usato nella competizione DPC-3. Non è nostro — è uno strumento standard che tutti accettano. Lo abbiamo usato per confermare i nostri risultati su un terreno "neutrale".

**DPC-3 (Data Prefetching Championship 3):** Una competizione accademica dove i ricercatori si sfidano a chi costruisce il prefetcher migliore. Usa ChampSim e tracce SPEC CPU2017. Il vincitore nel 2021 è stato Berti.

**SPEC CPU2017:** Una collezione standard di 20+ programmi reali (compilatori, simulazioni fisiche, AI, compressione...) usata per testare le CPU. Noi ne abbiamo usati 20.

**Traccia:** Una registrazione di tutti gli accessi alla memoria che un programma fa durante l'esecuzione. Invece di rieseguire il programma, "riproduciamo" la traccia nel simulatore — più veloce e ripetibile.

**Outlier:** Un valore che si discosta enormemente dalla media. Nel nostro caso GCC (-4.37%) e MCF (-2.92%) sono i due programmi dove il gate fa i danni peggiori. Tutti gli altri 18 stanno dentro ±0.4%.

**Clamp ±16:** Un limite che abbiamo messo al modello: può prevedere salti al massimo di 16 posizioni avanti o indietro. Se un programma fa salti di 32 (come una moltiplicazione di matrici), il modello fallisce per costruzione, non perché sia stupido.

**Strawman:** Termine da debate. Significa "avversario di paglia" — un avversario costruito apposta per essere facile da battere. Il reviewer accusa il nostro MLP a 257 parametri di essere uno strawman: troppo piccolo per dimostrare qualcosa.

**Leave-two-out:** Analisi che mostra cosa succede ai risultati se rimuovi i due casi peggiori (GCC e MCF). Serve a dimostrare che le conclusioni non dipendono solo da quei due outlier.

**Venue:** La conferenza o rivista dove mandi il paper. MICRO e ISCA sono le più prestigiose (e difficili). CAL e ISPASS sono più accessibili.

---

## Parte 1 — Cosa abbiamo fatto in queste due sessioni

### 1. Test su larga scala (27-28 agosto)

Abbiamo fatto girare 80 esperimenti su ChampSim:
- 20 programmi diversi (da SPEC CPU2017)
- 4 configurazioni per ognuno: nessun prefetch, stride standard, nostro stride senza gate, nostro stride con gate
- Ogni esperimento simula 250 milioni di istruzioni
- Tutto su WSL2 con 16 core, ci ha messo ~8 ore

### 2. Riscrittura del paper (28 agosto)

- **Abstract**: tagliato da 284 a 140 parole, reso più chiaro
- **Conclusione**: riscritta da zero, non ripete più l'abstract
- **Due sezioni fuse** per risparmiare spazio (eravamo al limite delle 8 pagine)
- **Grafico nuovo**: bar chart che mostra come cambia la velocità (IPC) per ognuno dei 20 programmi
- **Due citazioni aggiunte**: Berti (vincitore DPC-3) e SGLang (framework LLM)
- **Sezione "limitazioni"**: aggiunto paragrafo sulla dimensione del modello

### 3. Revisione adversarial

Ho simulato un reviewer cattivo di MICRO/ISCA che cerca di bocciare il paper. Ha trovato problemi a tre livelli di gravità (vedi sotto).

### 4. Fix rapidi già applicati

- Conteggio bibliografia corretto (era 27, sono 26)
- Frase "To our knowledge" resa meno arrogante
- Cross-reference verificati: erano tutti OK

---

## Parte 2 — I problemi trovati

### Gravità dei problemi

- 🔴 **FATAL** = da solo basta per bocciare il paper
- 🟠 **MAJOR** = problema serio, va risolto o il paper viene bocciato
- 🟡 **MINOR** = problema piccolo, facile da fixare

### Stato

- ✅ Già risolto
- ⚠️ Parzialmente affrontato
- ❌ Richiede nuovo lavoro

---

### 🔴 F1 — Il paper dice "c'è un problema" ma non propone una soluzione
**Stato: ❌ — è la natura stessa del paper**

Il paper dimostra due cose:
1. Il gate è più importante della rete neurale
2. Le metriche intermedie non predicono il risultato finale

Sono scoperte utili, ma le conferenze top (MICRO, ISCA) vogliono che dopo il "ecco il problema" ci sia un "ecco la soluzione". Noi ci fermiamo alla diagnosi.

**Impatto pratico:** Non possiamo mandarlo a MICRO/ISCA così com'è. Possiamo mandarlo a conferenze che accettano contributi metodologici (CAL, ISPASS) — lì va bene dire "guardate, tutti stavate misurando nel modo sbagliato".

---

### 🔴 F2 — Il nostro simulatore custom non è riconosciuto
**Stato: ⚠️ — parzialmente risolto con ChampSim**

Il simulatore che abbiamo costruito noi è molto semplificato: le istruzioni vanno una dopo l'altra (in-order), non c'è la traduzione degli indirizzi (TLB), non simula la contesa sulla memoria. I reviewer non si fidano di simulatori "fatti in casa".

**Cosa abbiamo fatto:** Aggiunto esperimenti su ChampSim, che è uno strumento standard che tutti accettano. I risultati su ChampSim confermano la stessa direzione.

**Cosa manca:** L'esperimento chiave (rete neurale vs stride, entrambi con gate) l'abbiamo fatto solo sul nostro simulatore, non su ChampSim.

---

### 🟠 M1 — L'esperimento più importante non è stato rifatto sullo strumento standard
**Stato: ❌**

Il cuore del paper è: "metti lo stesso gate sulla rete neurale E sullo stride, e vedrai che la rete neurale non aggiunge nulla". Questo confronto esiste solo nel nostro simulatore. Su ChampSim abbiamo testato il gate, ma non il confronto diretto rete neurale vs stride.

**Per risolverlo:** Bisognerebbe scrivere la rete neurale come prefetcher ChampSim e ripetere il confronto. Circa 2-4 settimane di lavoro.

---

### 🟠 M2 — La rete neurale è troppo piccola per essere un test serio
**Stato: ⚠️ — discusso nelle limitazioni**

La nostra rete neurale ha 257 parametri. Quelle "serie" nella letteratura ne hanno da 10.000 a oltre un milione. Un reviewer dirà: "avete dimostrato che un modellino ridicolo non funziona — bella scoperta".

**La nostra difesa:** Il punto del paper non è "la rete neurale è cattiva", ma "il gate è quello che conta". Anche con un modello più grande, se togli il gate, la differenza dovrebbe sparire. Ma non l'abbiamo dimostrato con i dati.

**Per risolverlo:** Allenare un modello 10-100× più grande e ripetere gli esperimenti. Circa 1-2 settimane.

---

### 🟠 M3 — Abbiamo limitato artificialmente il modello
**Stato: ⚠️ — menzionato nelle limitazioni**

Il modello può prevedere salti al massimo di ±16 posizioni. Se un programma ha bisogno di salti di 32 o 64, il modello fallisce per costruzione — non perché è incapace, ma perché noi gli abbiamo messo un limite troppo stretto.

Un reviewer dirà: "vi siete legati le mani dietro la schiena e poi vi lamentate di non poter nuotare".

**Per risolverlo:** Testare con limiti più ampi (±64, ±128). Circa 1-2 giorni.

---

### 🟠 M4 — Presentare 7 vittorie su 20 come un successo
**Stato: ⚠️ — il framing è onesto ma attaccabile**

Su 20 programmi, il gate vince su 7, perde su 11, pareggia su 2. Il paper non lo presenta come "vinciamo sempre" — dice che il gate preserva la direzione del prefetcher (14 su 20 vanno nella stessa direzione). Ma un reviewer conterà 7 vs 11 e dirà "perdete più di quanto vincete".

**La realtà:** Le differenze sono minuscole (±0.1%) tranne per 2 outlier. In pratica il gate non cambia quasi nulla — che è esattamente il punto del paper: le proxy mentono.

---

### 🟠 M5 — Cross-reference rotti nel paper
**Stato: ✅ — falso allarme, sono tutti OK**

---

### 🟡 m1 — Frase troppo presuntuosa
**Stato: ✅ — corretto**

"To our knowledge, no prior work..." suona arrogante. Cambiato in "No prior work we have found..." — stessa cosa, tono più onesto.

---

### 🟡 m2 — Manca l'analisi senza i due casi peggiori
**Stato: ❌**

GCC e MCF sono i due programmi dove il gate fa più danni (-4.37% e -2.92%). Il reviewer vuole sapere: "se tolgo questi due, le vostre conclusioni reggono?" Bastano 30 minuti per calcolarlo e aggiungere una riga nel paper.

---

### 🟡 m3 — Non abbiamo contato tutto il costo hardware
**Stato: ❌**

Diciamo che il modello occupa 514 byte (257 numeri × 2 byte). Ma il costo reale nel chip include anche i circuiti per fare i calcoli (moltiplicatori, sommatori, registri). Il costo totale è probabilmente 5-10 volte di più.

---

### 🟡 m4 — L'abstract sembra dire che il gate funziona
**Stato: ✅ — già corretto**

L'abstract dice "il gate rimuove il 35% dei prefetch e migliora l'accuratezza" — sembra positivo. Ma subito dopo dice "...ma i DRAM reads cambiano solo dello 0.07%" — ecco il colpo di scena. Il "ma" c'è già.

---

### 🟡 m5 — Numero sbagliato nella bibliografia
**Stato: ✅ — corretto**

Diceva 27 riferimenti, erano 26. Fixato.

---

## Dove mandare il paper?

| Dove | Probabilità di accettazione | Perché |
|------|----------------------------|--------|
| MICRO / ISCA / HPCA | 15-20% | Vogliono soluzioni, noi offriamo diagnosi |
| ISPASS / IISWC | 50-60% | Accettano contributi metodologici |
| IEEE CAL (letter, 4 pag) | 70-80% | Formato ideale per risultati diagnostici |
| Workshop (ML for Systems) | 80%+ | Il posto naturale per questo lavoro |

**Consiglio: IEEE CAL** — 4 pagine, risposta veloce, buona visibilità. Tagliamo il simulatore locale e teniamo solo ChampSim.

---

## Cosa fare adesso, in ordine di priorità

| # | Cosa | Tempo | Impatto |
|---|------|-------|---------|
| 1 | Calcolare i risultati senza GCC e MCF | 30 min | Chiude una critica ovvia |
| 2 | Testare con limite ±64 invece di ±16 | 1-2 giorni | Dimostra che il limite non ha falsato i risultati |
| 3 | Portare la rete neurale su ChampSim | 2-4 settimane | Chiude il problema più grosso |
| 4 | Testare con un modello più grande | 1-2 settimane | Dimostra che non è uno strawman |

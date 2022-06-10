# TO-DO (Mike)

- [ ] Vedere se ha senso parallelizzare le euristiche utilizzando la struttura dati dinamica.

- [ ] Supportare l'aggiunta delle clausole di conflitto.

- [ ] Fare documentazione con Doxygen.

- [ ] Implementare codice che verifica che non ci siano clausole ripetute nel file DIMACS CNF.
  Da terminale ciò si può verificare con il seguente comando:

  ```bash
  sort filename.cnf | uniq -cd
  ```

  il quale stamperà solamente le linee duplicate con i relativi conteggi.
  
- [ ] Sistemare `mrc_dyn` come `mrc` e `mrc_gpu`.

- [ ] Mettere le strutture dati ausiliarie di `miracle.cu` e `miracle_gpu.cu` dentro la struttura dati `SAT_Miracle` al fine di poter spawnare più `SAT_Miracle` autocontenuti.

- [ ] Ottimizzare la PSH di BOHM (?).

- [ ] Ottimizzare la reduction parallela delle funzioni `find_*` in `utils.cu` facendola a livello di warp.

- [ ] Aggiungere la trasposta di `phi->clauses` (la struttura dati *csc*) per avere l'accesso diretto alle clausole in cui è presente un certo letterale.

- [ ] Fare il porting su AMD HIP.

# TO-DO (Andy)

## Esperimenti

- [ ] Reperire almeno una cinquantina di istanze utili a esperimenti per *computation_time_evaluation*.

## Estensioni del software

- [ ] Aggiungere all'output la stampa di info su:
  -  hardware host e device (CPU, versioni CUDA, modello e compute capability della GPU, numero SMs, memoria, cores, ...).
  -  Data/ora in cui il run è stato fatto.
  -  Stampare anche i dati sulla istanza (nome, dimensioni, spazio allocato in memoria, ...).
  -  Sul run (TPB, num blocks, ...) quale euristica si è usata, ed eventuali altri parametri (timeout, sat/unsat/unknown, ...).

  In pratica, nel (file di) output ci deve essere tutto ciò che si potrebbe voler sapere sull'esperimento e per ripetere il run.

- [ ] Aggiungere la possibilità (per esempio con opzioni su linea di comando) per la scelta del numero di TPB e del numero di blocks-per-grid (indipendentemente dal numero di SMs e ammettendo che TPB possa essere anche inferiore a 32).

- [ ] Aggiungere la possibilità (per esempio con opzioni su linea di comando) per la scelta del fattore *K* con cui moltiplicare il numero di SMs per il bound al numero di blocks (quando è calcolato automaticamente: ora *K* è fissato a 2).

- [ ] Aggiungere opzione per la stampa delle statistiche in singola linea CSV (inibendo altri output).

- [ ] Aggiungere opzione per l'interruzione del run dopo *N* valutazioni dell'euristica.

- [ ] Computare le riduzioni con tecniche più efficienti e confrontarle, usando:
   * __reduce_max_sync  (quando è possibile),
   * __shfl_down_sync,
   * cub::DeviceReduce;

  (selezionabili al compile time).

- [ ] Implementare qualche altra euristica di look-back (prima CPU, poi anche GPU) in microsat (oltre a VMTF).

- [ ] Rivedere la distribuzione del carico threads-letterali/clausole.

- [ ] Gestire solo le clausole attive nel calcolo delle euristiche. (Aggiungere la trasposta di `phi->clauses` (la struttura dati *csc*) per avere l'accesso diretto alle clausole in cui è presente un certo letterale. Vedi sopra.)

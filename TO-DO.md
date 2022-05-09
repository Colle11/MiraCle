# TO-DO

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


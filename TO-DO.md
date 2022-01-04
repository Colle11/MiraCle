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


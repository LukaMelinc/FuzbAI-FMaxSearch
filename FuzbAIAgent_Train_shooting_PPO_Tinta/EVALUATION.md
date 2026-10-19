# Evalvacija agentov

Tri skripte zaženejo model v deterministični inferenci v lokalnem FuzbAISim,
z izključenim učenjem. Evalvator prejme stanje pred ponastavitvijo žoge.
Ne priključujejo se na drug proces prek omrežja. Za uporabo v obstoječem
procesu ustvari `Evaluator(task, args)` in ga podaj kot `evaluator=` ob
ustvarjanju `FuzbAISim`; `observe(camera)` omogoča tudi neposredno integracijo
z istim formatom opazovanj. Kontakt `ball_kicked` mora izvirati iz izbrane
palice (`kick_observed_rod_id`).

Primeri (pragovi so zgolj primeri; nastavi jih glede na nalogo):

```bash
python3 evaluate_shooting.py --checkpoint 'Final_models/Shoting/#3-2.pth' --episodes 100 --negative-speed 0.05 --negative-duration 0.3 --episode-seconds 5
python3 evaluate_defense.py --checkpoint POT_DO_OBRAMBNEGA_MODELA.pth --episodes 100 --rod-id 6 --x1 800 --x2 950 --failure-x 780 --spawn-x1 1.02 --spawn-x2 1.16 --spawn-speed 0.45
python3 evaluate_passing.py --checkpoint 'Final_models/Passing/#1.pth' --episodes 100 --rod-id 4 --receiver-rod-id 6 --failure-x 480 --overshoot-x 950 --x1 850 --x2 920 --stop-speed 0.05 --stop-duration 0.2
```

`--gui` odpre prikaz. `--output` določa CSV (privzeto `report/evaluation_TASK.csv`),
poleg njega nastane `.summary.json` z nastavitvami, številom uspešnih, delno
uspešnih in neuspešnih epizod ter deležem polno uspešnih epizod. Rezultati se
shranijo po vsaki zaključeni epizodi. Prekinitev s Ctrl+C shrani zaključene
epizode; nedokončane ne vključuje v delež uspešnosti. Ponoven zagon prepiše
izbrani izhod, zato za ločene evalvacije uporabi različne `--output` poti.

`--episode-pause 1` zamrzne končno stanje za eno sekundo pred začetkom naslednje
epizode. Premor je v realnem času in ne šteje v trajanje simulacije; privzeto je 0.

Položaji in pragovi X so v **mm**, hitrost v **m/s**, trajanje v **sekundah
simulacije**. Koordinate so iz perspektive igralca 1; pozitivni vx pomeni
strel proti nasprotnikovemu golu. Meritve uporabljajo pravo stanje simulacije,
brez šuma kamere. `--spawn-x1` in `--spawn-x2` sta izjemoma v **svetovnih metrih**:
`x_mm = 1000 * x_world - 115`. Začetna žoga pri privzetem mirujočem začetku
ni enaka prihajajoči žogi pri obrambi: tam nastavi območje in hitrost.
Pri podanem območju prihaja žoga iz smeri pozitivnega X (vx < 0).
`--seed` določa seme (privzeto 42); privzete nastavitve ostalih začetnih
položajev in nasprotnika uporablja simulator.

- **Strel:** porast lastnega rezultata pomeni uspeh. Neprekinjeni
  `vx < -negative_speed` vsaj `negative_duration` pomeni zgrešen strel.
  Pozitivna ali dovolj majhna negativna hitrost ponastavi časovnik.
  Nasprotnikov gol ali iztek epizode brez zadetka pomeni neuspeh.
- **Obramba:** uspeh določa končni X žoge v zaprtem pasu `[x1, x2]` ob
  koncu epizode; začetni položaj in predhodni izleti iz pasu ne vplivajo na uspeh.
  Ko `x < failure_x`, se epizoda takoj zaključi:
  predhodni ali v istem vzorcu zaznan kontakt izbrane palice pomeni delni
  uspeh, odsotnost kontakta pomeni neuspeh. Končni X zunaj pasu brez prestopa
  spodnje meje pomeni neuspeh ob koncu epizode.
- **Podaja:** `x < failure_x` pomeni napačno smer; `x > overshoot_x` pomeni
  neuspešno zaustavljanje. Uspeh zahteva skupno ravninsko hitrost
  `sqrt(vx² + vy²) <= stop_speed` neprekinjeno `stop_duration` v `[x1, x2]`.
  To prepreči uspeh pri hitrem gibanju v Y ali zgolj trenutnem obratu žoge.
  Iztek ali agentov predčasni konec brez doseženega pogoja pomeni neuspeh.

CSV beleži razlog, trajanje, končni X, kontakt, število zaznanih brc,
največjo vzorčeno kontaktno silo (`max_kick_force_n`), hitrost žoge ob
zaznani brci (`max_kick_speed_m_s`) in največjo hitrost žoge v epizodi.
`kicks_json` vsebuje vsako zaznano brco s časom, silo, hitrostjo in vx.
Sila je normalna kontaktna sila PyBullet v N; ni mehanska moč v W.
Štetje dogodkov uporablja obstoječi kontaktni cooldown simulatorja;
neprekinjen kontakt lahko zato vsebuje več zaznanih dogodkov.
Evalvacija vzorči vsak kontrolni korak (privzeto 0,05 s), kontakt pa
simulator zaznava med fizikalnimi koraki. Zelo kratki izleti iz pasu med
kontrolnimi vzorci niso zajeti.

Privzeto strel in obramba uporabljata `TwoRodPPOAgent`, podaja `PassPPOAgent`.
Za starejše modele `PPOAgent` z osmimi vhodnimi značilkami uporabi
`--agent-mode single_rod_ppo --rod-id 2` (palico nastavi tako kot pri učenju).
`eval_obramba.sh` uporablja ta način za `Final_models/Defence/#6.pth`.
Checkpoint mora ustrezati arhitekturi. Dodatne konstrukcijske nastavitve,
npr. velikost mreže, podaj kot `--agent-kwargs '{"hidden_size": 512}'`.
Inferenca in izklop učenja sta obvezna in ju ta možnost ne more prepisati.

Preverjanje pravil:

```bash
python3 -m unittest discover -s tests -p test_evaluation.py
```

\section{Simulator}\label{sec:Simulator}

Za treninge algoritmov spodbujevalnega učenja je ključen simulator za zagotavljanje učnega seta agenta. Zanimiv primer so naprimer videoigre kot sta Doom in Dota 2, kjer se agenti učijo skozi kontrolo avatarja v igri ter tako pridobivajo učni set podatkov. 

V kolikor učimo agenta za operiranje v realnem svetu, je ključna natančna izdelava simulatoraj kot digitalnega dvojčka realnega okolja, v našem primeru mize namiznega nogometa. Že manjša odstopanja oblike ali lastnosti simulatorja od realnega okolja lahko vodijo v slabe rezultate učenja. To je še posebej pomembno pri dinamičnih okoljih oziroma pri okoljih, kjer imajo že manjši detajli kot so oblike polinomov velik vpliv na obnašanje elementov v simulaci.

Obstaja več različnih okolij in knjižnjic za razvoj simulacijskih okolij. Med najbolj znanimi sta gotovo Unity, ki je bil zasnovan za razvoj videoiger, sedaj pa se med drugim uporablja za razvoj okolij za učenje umetne inteligence v robotiki. Popularni so tudi NVIDIA Issac, Unreal Engine 5 ter MuJoCo, sami pa smo uporabili simulator na podlagi PyBulleta, ki so ga razvili v LAK za namene prve izvedbe tekmovanja FuzbAI. Pybullet je Python modul za simulacije fizike, v katerega se uvozi URDF datoteka same igralne mize, igralni pogon (game engine) pa nato izračunava trke med poligoni v igri. 

S simulatorjem simuliramo igralno mizo namiznega nogometa. Igrišče vsebuje 8 palic, po štiri na vsaki strani. Vsaka stran ima palico za vratarja (en igralec na palici), za branilca (dva igralca na palici), za veziste (pet igralcev na palici) ter palico za napadalce (trije igralci na palici). 

Vedno smo vodili rdečo stran, modra stran je bila vedno nasprotnik, ki smo ga lahko kontrolirali glede na željeno obnašanje med treningom. Pri določenih treningih smo obrnili modre palice na glavo, da igralci niso predstavljali ovire med treningom, v določenih smo aktivno preko vnaprej programiranega agenta vodili modre palice za določene akcije zavoljo situacije pri treningu.


\subsection{Igralna miza}\label{Strojna oprema}
Igralno mizo so zasnovali in izdelali člana laboratorija LAK FE UL. Gre za klasično mizo za igro namiznega nogometa. Palice so opremljene s grayevimi črtnimi kodami za zaznavanje translacije in rotacije palic. Dve N-herčni kameri, nameščeni nad igrišče zajemata stanja palic ter poziciji ter hitrosti žogic.

Na rdeči strani igrišča so montirani štirje sistemi aktuatorjev, kjer vsak upravlja svojo igralno palico. Ker se v nalogi nismo ukvarjali z vodenjem dejanskih motorjev in smo ostali le v simulaciji, se v podrobnosti ne bomo spuščali.

\subsection{Strojna oprema}\label{Strojna oprema}
Treniranje agenta smo izvedli na domačem serverju Tilna Tinte. Uporabljali smo grafično kartico Nvidia Tesla P4, ki premore 8 gigabajtov videoRAMa, kar se je izkazalo za dovolj za sorazmerno hiter trening modelov.   

\subsection{Omejitve}\label{Strojna oprema}

Pri simulatorju in fizični mizi smo se soočili z nekaj omejitvami. 

Simulator je simuliral realne motorje za kontrolo palic, ki je omejen z rotacijskimi in translacijskimi pospeški. 
Pri samem poganjanju simulatorja smo bili omejeni s zmogljivostmi strojne opreme, v našem primeru s hitrostjo procesorja, zato smo lahko simulacijo pospešili le do določene hitrosti, dodatno smo simulacijo pospešili z izklopom grafičnega vmesnika, kar je še dodatno uspelo pospešiti simulacijo. Na začetku je bila ideja tudi, da bi paralelizirali simulacijsko okolje, vendar pa je paralelizacija simulatorjev v okolju PyBullet izjemno kompleksna in se na koncu nismo poslužili tega.
Pri snemanju trening podatkov za imitacijsko učenje smo naleteli na težavo, da kameri ne uspeta dobro posneti res hitrih akcij. Ko smo trenirali agenta na hitro izvedenih trening epizodah, so bili rezultati slabi, zato smo ob ponovnem snemanju učnih podatkov za imitacijsko učenje akcije izvajali počasneje, da so kamere lahko dobro zajele stanja igralne mize.

\section{Metodologija}\label{sec:Eksperimentalno delo}

Našo nalogo treninga agenta za avtonomno igro smo razdelili na več ločenih delov. To smo storili zaradi kompleksnosti in raznolikosti vseh možnih akcij, ki so prisotne v igri. S tem smo tudi dosegli, da smo lahko treninge izvajali hitreje in z več nadzora ter da smo trening lahko izvedli na široko dostopni strojni opremi in se ni bilo potrebno poslužiti drage in nedostopne opreme.

Igro smo razdelili na tri glavne komponente: Igranje obrambe, podaja med dvema palicama ter strel na gol.
Zavoljo časa smo vsako akcijo trenirali le z eno palico/parom palic.

Akcijo obrambe bi lahko poimenovali tudi prestrezanje žoge, gre za del igre, ko ima žogico v posesti nasprotnik, sami pa želimo ovirati nasprotnika pri podaji naprej oziroma strelu na gol. 
Akcija podaje je precej jasna, cilj je podati žogico med sosednjima palicama tako, da palica, ki žogico sprejema, to umiri do mere, da je pod nadzorom.

Tretja akcija je strel na gol, ki smo jo izvedli s tremi različnimi pristopi treninga - čisto imitacijsko učenje, čisto spodbujevalno učenje in kombinacija obeh pristopov s predtreniranjem z imitacijskim učenjem ter posttreniranjem z spodbujevalnim učenjem.

Vsak eksperiment smo izvedli le na eni palici, ne na vseh, saj bi trening na vseh palicah trajal predolgo, dodaten trening pa ne bi doprinesel k boljšim zaključkom naloge.

IMplementirali smo tri pristope k treningu - spodbujevalno učenje, imitacijsko učenje ter kombinacija obeh pristopov s predtreningom z imitacijskim učenjem ter posttreningom s spodbujevalnim učenjem. Imitacijsko učenje omogoča usvojitev začetnega vedenja pri kompleksnih nalogah, pri katerih bi učenje izključno s spodbujevalnim učenjem zahtevalo veliko časa in postopno učenje posameznih delov naloge. Želeno vedenje je namreč včasih težko ustrezno izraziti z nagradno funkcijo, medtem ko ga lahko demonstrator neposredno prikaže. Z imitacijskim učenjem agent tako pridobi osnovni vzorec izvajanja naloge, ki ga nato s spodbujevalnim učenjem dodatno izboljšuje. Takšna kombinacija lahko zmanjša potrebo po učenju posameznih delov naloge po etapah ter omogoči nadaljnjo optimizacijo hitrosti in uspešnosti izvedbe. Agent lahko pri tem preseže zmogljivost demonstracij, na katerih je temeljilo začetno učenje.

\subsection{Prestrezanje žoge}\label{Prestrezanje žoge}

Osnovna akcija v situacijah, kjer je žoga v posesti nasprotnika in želi izvesti strel na naša vrata oziroma izvesti podajo med svojima palicama. 


Cilj naše akcije je, da je agent zmožen prestreči žogo, ki potuje s hitrostjo proti našemu golu, oziroma idealno, da žogici zmanjša absolutno hitrost na nič v okolici naše palice. Takrat lahko rečemo, da smo žogico od nasprotnika prevzeli v posest.

Treninga agenta smo se lotili na dva načina, saj smo po prvem pristopu k treningu odkrili problematično delovanje.

\subsubsection{Opazovalni in akcijski prostor}

Za to nalogo smo uporabili nevronsko mrežo, ki je zajemala N vrednosti () ter vračala 4 vrednosti (). Pri tej arhitekturi smo opazovali le parametre dotične palice ter parametre žoge, v trening ni bila vključena nobena druga palica. 

\subsubsection{Oblikovanje nagrade in trening}
Prvi pristop

Začeli smo z naključnim agentom.
Žogo smo v prvi fazi treninga mirujočo spawnali po celotni širino igiršča, razen na širinah, kjer je podlaga pod kotom, in bi se žoga začela gibati. 

Razvili smo funkcijo, ki je glede na to, katero palico smo uporabljali ter na trenutno translacijo palice ugotavljali, v območju katerega igralca na palici je trenutno žogica. 
Ko smo vedeli, v območju katerega igralca na palici je žogica, smo lahko razvili reward function, kjer smo nagrajevali poravnanost po y osi igralca na palici z žogo. Okoli trenutne pozicije žogice smo oblikovali območje pripadnosti z gaussovo porazdelitvijo, bližje kot je bil igralec na palici žogici, večjo nagrado smo dodelili agentu. 

V prvih dveh etapah treninga smo trenirali samo akcijsko glavo agenta, ki kontrolira translacijo palice, rotacijsko glavo pa smo zamrznili.

Agenta smo tako trenirali 1M korakov, po kateremo se je agent naučil dobro poravnati igralca na palici z mirujočo žogo.

V drugi fazi smo vzeli trenirani model, ter žogico začeli sprožati z nenično lateralno hitrostjo, s čimer smo želeli agenta naučiti sledenja žogi pri neničelnih hitrostih. Tudi tokrat smo agenta učili 1M korakov in po treningu je agent dobro sledil žogi pri lateralnem gibanju (po širini igralne mize), posebaj pomembno je, da je agent brez težav zamenjal igralca, ko je žoga prešla iz območja enega igralca na palici v območje sosednjega igralca.

V tretjem koraku smo nato dodali še longitudinalne hitrosti žoge v smeri proti našemu golu, hkrati pa smo zamrznili akcijsko glavo za translacijo, saj smo bili zadovoljni z njenim delovanjem po predhodnih dveh treningih. 

V tej etapi treninga smo nagrajevali agenta, da je uspel v okolici palice zmanjšati hitrost potujoče žogice. Določili smo okolico palice v širini dosega igralca na palici. Nagrajevali smo agenta, ki je uspel znotraj te okolice zmanjšati x in y komponento hitrosti žoge, v kolikor pa je hitrost uspel spraviti na okoli nič (dopustili smo manjši pogrešek zaradi lastnostni simulaotorja), pa smo mu prisodili končno nagrado za ustavitev žoge v okolici palice.

Žogico smo sprožali iz x pozicije okoli sredinske rdeče palice s hitrostmi med X in X.

Drugi pristop

Treninga akcije smo se zaradi problema, ki je opisan v rezultatih (poglavje x.x.x), lotili še z drugim pristopom. Pripravili smo funkcijo, ki je na podlagi trenutne pozicije žogice ter hitrosti žogice po x in y osi izračunava, na kateri širini bo žogica prečkala modro sredinsko palico. To vrednost smo vzeli ter izračunali, v območju katerega igralca na palici bo žogica ob prečkanju palice. Na podlagi tega smo nato nagrajevali agenta za pozicioniranje agenta na predvideno točko prečenja. 

Žogico smo nato spawnali pod različnimi koti iz različnih pozicij na način, da žogica ni nikoli prečila rdeče sredinske palice, vendar pa je ves čas potovala proti njen. Skozi trening smo ugotovili, da je pomembno, da žogice ne spawnamo prepočasi, saj takrat predikcije točke prečenja zaradi nizkih hitrosti in potencialno večjih možnih napak postanejo večje, kar lahko poslabša kvaliteto treninga. Zato je bilo potrebno povečati spawn hitrosti, malce skrajšati dolđine epizod in psotaviti spawn točko bližje modrim vratom za dovoljšno dolžino epizod za kvaliteten trening.

V drugem delu treninga smo vzeli predučeni model ter podobno kot v prvem pristopu nagrajevali palico za zmanjšanje hitrosti žogice znotraj območja palice.


\subsection{Strel}\label{Strel}

Akcija, ki jo želimo izvesti za dosego gola na modri gol. Glavni cilj akcije je zadeti gol iz različnih pozicij po igrišču z napadalno rdečo palico.

Treninga akcije smo se lotili le na en način, a nakoncu predlagamo še alternativen.

	Akcijski prostor in prostor opazovanj 
Pri temu modelu smo zajemali parametre dveh palic: napadalne rdeče in branilske modre, poleg tega pa zajema tudi parametre žogice - pozicija in hitrost.

	Oblikovanje nagrade in trening
Začeli smo z naključnim agentom. V prvem koraku treninga smo agenta tako kot pri akciji prestrezanja nagrajevali za poravnavo ustreznega igralca na palici s mirujočo žogico. Hkrati smo zamrznili akcijsko glavo za rotacijo palice, saj smo želeli ohraniti naključno naravo rotacijske glave.
Po prvem treningu smo v drugem koraku treninga zamrznili akcijsko glavo translacije palice ter odmrznili rotacijsko akcijsko glavo agenta. V temu koraku smo preko zaznavanja simulatorja opazovali, kdaj je agent uspel brcniti žogo. Ker se nam je lahko zgodilo, da je agent lahko brcnil žogo tudi v napačno smer, torej proti rdečemu golu, smo zaznavali na podlagi hitrosti po brci, ali je žoga bila brcnjena v pravilni smeri (ali je hitrost pozitivno ali negativno predznačena). V temu treningu smo tudi nastavili pravila konca epizode tako, da se je epizoda končala en korak po zaznani brci. To smo naredili zato, da smo skrajšali epizode na le uporaben del, saj nas v tej točki treninga dogajanje po brci ne zanima več, za trening je zanimivo le dogajanje pred in med brco žoge. Po brci smo zajeli še en korak, da smo lahko na podlagi x komponente hitrosti žoge določili smer žoge.

Po drugem treningu smo vzeli treniranega agenta ter za zadnji del treninga odmrznili obe akcijski glavi za končno fino treniranje agenta. V zadnji fazi smo nagrajevali dosežene gole kot željeni cilj, poleg tega pa smo dodali nagrajevanje, kjer smo na podlagi točke na sredini modrega gola ter pozicije žoge ob trenutku brce izračunali, po kakšni trajektoriji bi morala potovati žogica v gol. Na podlagi tega smo nagrajevali, kako blizu je trajektorija potovanja blizu idealni trajektoriji.

Po izvedeni tretji stopnji treninga smo dobili agenta, ki je zanesljivo dosegal zadetke iz začetnih pogojev. Po temu smo želeli dodati še nadgradnjo. Vse do tega trenutka smo trenirali agenta tako, da je bila obrambna modra palica obrnjena tako, da so bili igralci na palici obrnjeni z glavo navzdol. Tako smo vzeli treniranega agenta ter ogrnili obrambno modro palico, da so bili igralci na palici obrnjeni z nogami navzdol. V tej etapi treninga smo nagrajevali agenta, ko je kljub aktivni statični obrambni palici uspel žogo brcniti v gol oziroma vsaj mimo obrambne modre palice, kaznovali pa smo, ko je agent žogo brcnil tako, da je simulator zaznal, da se je žoga odbila od modrega obrambnega igralca.


\subsubsection{Oblikovanje nagrade in trening}

\subsection{Podaja}\label{Podaja}

Pri tretji trenirani akciji smo agenta trenirali, da se je naučil kvalitetno podati žogo od sredinske rdeče palice do napadalne rdeče palice. Zaradi kompleksnosti akcije smo se zavoljo bolj efektivnega treninga omejili na izvajanje akcije le v območju ob desni strani igralne mize, vendar pa bi se z daljšim treningom celotno akcijo natreniralo po celotni širini igrišča. 


	Prostor opazovanj in akcij
Tokrat smo zajemali parametre treh palic - sredinske rdeče in napadalne rdeče palice ter sredinske modre palice, ob temu pa še parametre žoge. V nadaljevanju bomo rdečo sredinsko palico poimenovali palico za podajanje ali podajalna palica in napadalno rdečo palico palica za sprejemanje ali sprejemalna palica.

	Oblikovanje nagrade in trening
K treningu smo ponovno pristopili v etapah, prvi del treninga je zelo spominjal na učenje akcije za strel, drugi del treninga pa precej na učenje akcije prestrezanja žoge. 

Najprej smo zamrznili mreže sprejemalne palice ter akcijsko glavo rotacije, trenirali smo samo translacijsko akcijsko glavo podajalne palice. Ponovno smo nagrajevali agenta za poravnanost ustreznega igralca na palici z žogo glede na pozicijo žoge. 
V drugem koraku smo kot pri strelu za trening odmrznili plasti rotacijske akcijske glave podajalne palice, zamrznili plasti translacijske glave ter ohranili zamrznjene vse plasti sprejemalne palice. V tej etapi treninga smo kot pri enakovredni etapi treninga strela nagrajevali brco igralca žoge v ustrezni smeri ter kaznovali brco v napačno smer. Tako kot pri treningu strela smo epizode zaključili en korak po zaznanemu kontaktu. Hkrati pa smo v tej etapi treninga s strani simulatorja zaznavali, s kakšno silo je bila žoga brcnjena. Tako smo dodali nagrajevanje, kjer smo nagrajevali moč brce žoge. Povsem empirično smo za željeno moč brce nastavili 12 newtnov sile ter nagrajevali brce čim bliže so bile sile brc.

Po tej epizodi smo začasno zaključili s treningom mrež podajalne palice in nadaljevali trening na plasteh sprejemalne palice. Trening smo izvedli podobno kot pri treningu akcije prestrezanja žoge, žogo smo spawnali premikajočo proti naši palici, iz pozicije žoge in njene hitrosti izračunavali, v kateri točki bo prečkala trenirano palico ter nagrajevali agenta za poravnavanje igralca na palici s to predvideno točko. 

  

\subsubsection{Oblikovanje nagrade in trening}

\subsection{Imitacijsko učenje}\label{Prestrezanje žoge}

Poleg treningov z izklučno spodbujevalnim učenjem smo izvedli tudi trening z izključno imitacijskim učenjem. Osredotičili smo se akcijo strela, a drugačen način kot pri treningu strela s spodbujevalnim učenjem, v temu primeru smo si z desnim igralcem na palici iz desne strani igrišča žogo porinili po sredini, palico zarotirali nazaj ter nato s sredinskim igralcem brcnili žogo v gol ob trenutku, ko je bila žoga pred sredinskim igralcem na palici. 

Upravljali smo samo napadalno rdečo palico, vse modre palice so bile obrnjene z glavo navzdol. 

Pripravili smo učni set z 119 epizodami. V prav vsaki epizodi smo uspeli izvesti manever podaje in strela, vendar pa smo gol zadeli le v 69 epizodah, v 50 epizodah smo gol zgrešili. Epizoda se je vedno začela s stoječo žogo z x koordinatami med X in X ter y koordinatami med Y in Y. Epizodo smo vedno ročno zaključevali po približno 3.5 sekundah.





\subsection{Strel}\label{Strel}

\subsection{Kombinirano učenje}\label{Prestrezanje žoge}

Na koncu smo implementirali še kombinirani pristop imitacijskega in spodbujevalnega učenja. Naš cilj je bil, da z predtrningom z imitacijskim učenjem oblikujemo približno politkiko agenta za strel na gol, nato pa s spodbujevalnim učenjem presežemo zmogljivosti demosntracij, posnetih na mizi. 

Za predtreniran model z imitacijskim učenjem smo uproabili isti model kot v prejšnji točki, nato pa smo dodali še posttreniranje s spodbujevalnim učenjem. Nagrado smo oblikovali tako, da smo nagrajevali zadeti gol, poleg tega pa smo nagrajevali tudi moč druge brce, torej moč brce sredinskega igralca žoge v gol, kjer smo nagrajevali čim močnejši strel.
Hkrati smo morali ohraniti imitacijsko nagrado, s katero smo poskrbeli, da nismo izgubili predtreniranega znanja agenta predhodne lateralne podaje znotraj palice pred strelom.


\subsubsection{Predtrening}

\subsubsection{Posttrening}
\section{Strojno učenje}\label{sec:Teoretično fifififi}

Strojno učenje je področje računalništva, ki se ukvarja z algoritmi, ki niso definirani s pred-napisanimi pravili temveč na podlagi vzorcev v učnih podatkih podaja napovedi ali odločitve.
 
Strojno učenje temelji na podatkih in ne na predhodnem znanju ali pravilih. Model v teh podatkih skozi trening poskuša razbrati statistične vzorce, na podlagi katerih nato poskuša podati svoje napovedi. Po končanem treningu nato trenirani model poskuša generirati napovedi na še-ne videnih podatkih.
 
Glavne tri paradigme strojnega učenja so nadzorovano strojno učenje, kjer model treniramo na podatkih, ki so označeni, nenadzorovano učenje, kjer pri treningu podatkov nimamo označenih ter spodbujevalno učenje, s katerim se ukvarjamo v tej magisterski nalogi.

\subsection{Globoko učenje}\label{Globoko učenje}

V strojnem učenju obstajata dve glavni veji. Prvo predstavljajo bolj tradicionalni modeli strojnega učenja, ki ne temeljijo na globokih nevronskih mrež in so bazirani na statistiki, pravilih odločanja, odločitvenih drevesih in verjetnosti. Modeli te veje so odločitvena drevesa, metoda podpornih vektorjev, logistična regresija, metoda najmanjših sosedov, naključni gozd, ... Taki modeli običajno ne potrebujejo tako velike računske moči za trening, vendar pa so ponavadi rezultati slabši od modelov, ki bazirajo na nevronskih mrežah.
 
Drugo vejo strojnega učenja, lahko jo poimenujemo tudi globoko strojno učenje, predstavljajo modeli z nevronskimi mrežami. Modeli te vrste vključujejo različne arhitekture nevronskih mrež (polno povezana nevronska mreža, CNN, RNN, transformer arhitektura, ...) ki razpoznavajo kompleksne vzorce v podatkih različnih modalitet (slike, zvoh, besedilo, ...). Modeli niso odvisni od vnaprej podanih značilk kot pri tradicionalnih metodah strojnega učenja, temveč jih v podatkih poiščejo oziroma razberejo sami. Taki modeli so primerni za širok nabor nalog od generiranje besedila, slik, zvoka, prepoznavanju in klasifikaciji objektov, avtonomnem vodenju. 

Ta pristop strojnega učenja je prevladal nad tradicionalnimi metodami strojnega učenja šele v zadnjih letih zaradi razvoja močnejše strojne opreme, ki je zmožna hitro izvajati kompleksne matematične operacije - množenje matrik, kar je omogočilo razmah treniranja nevronskih mrež na grafičnih procesnih enotah. Danes obstaja več knjižnic, ki omogočajo enostavno grajenje nevronskih mrež ter njihov trening, najbolj se je uveljvavila knjižnica PyTorch, ki smo jo uporabili v v tej nalogi.


\subsection{Nevronska mreža}

Nevronska mreža je skupek vozlišč ali nevronov, urejenih po plasteh, ki so povezani z vozliščmi v sosednjih plasteh. Gre za računalniški sistem, ki na podlagi analize velikih količin podatkov razpoznava vzorce ter rešuje kompleksne probleme. Nevronska mreža ne temelji na vnaprej programiranih pravilih kot klasični računalniški algoritmi, temveč vsebuje množico parametrov, ki se skozi trening mreže prilagajajo, da so po koncu treninga sposobni opraviti dano nalogo, bodisi je to klasifikacija, napoved ali kaj drugega. 

Glede delovanja lahko potegnemo vzporednico z delovanjem možganov, kjer so nevroni med seboj povezani z sinapsami, po katerih si izmenjujejo signale, informacije pa tako potujejo med nevroni. Potrebna razlika je, da se možgani za razliko od nevronskih mrež ne učijo z mehanizmom vzvratne propagacije napake.

Vozlišča so urejena po plasteh, ponavadi pa so vsa vozlišča v posamezni plasti povezana z vsemi vozlišči v sledeči plasti ter tako naprej. Na začetku mreže je vhodna plast, kjer vsak vsak nevron predstavlja posamezni signal, ki je vključen v mrežo. V našem primeru so vhodni signali vrednosti posameznih parametrov simulatorja, naprimer pozicija in hitrost žogice, pozicije palic, rotacije palic. Na izhodni plasti so izhodni nevroni, ki predstavljajo izhodni signal mreže, v našem primeru so izhodni signali kontrolne veličine za igralne palice. 

Vsak nevron je v polno povezani nevronski mreži povezan z vsakim nevronom v sledeči plasti, povezavam rečemo tudi uteži in predstavljajo koeficient, s katerim se prenaša vrednost prvega nevrona v sledečega. Vsak nevron ima poleg povezav do okolišnih nevronov tudi njemu lasten parameter pristranskosti, ki ga prav tako kot ostale parametre lahko treniramo. 

Vsak nevron izračunava svojo vrednost na podlagi vhodnih singlaov, posamezni nevron izračuna uteženo vsoto svojih vhodov, kateri prišteje pristranskost, vse skupaj pa pomnoži z aktivacijsko funkcijo nevrona. Ta je lahko ReLu, sigmoidna funkcija ali pa kakšna druga. Izhod nevrona lahko zapišemo kot:
  
\begin{equation}
    y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
    \label{eq:neuron_output}
\end{equation}

kjer je $x_i$ vrednost posameznega predhodnega nevrona, $w_i$ vrednost uteži do predhodnega nevrona, $b$ vrednost pristranskosti, $ƒ$ aktivacijska funckija nevrona. Aktivacijske funkcije v nevron vnašajo nelinearnost, kar nevronski funkciji omogoča opisovanje bolj kompleksnih funkcij med vhodnimi in izhodnimi podatki. 



\section{Spodbujevalno in imitacijsko učenje}\label{sec:Teoretično fifififi}

Podrobneje bomo predstavili imitacijsko ter spodbujevalno učenje, pristopa, ki smo ju uporabili za treninge agenta na nalogi. Predstavimo tudi glavne mehanizme, ki so uporabljeni pri učenju.


\subsection{Osnove spodbujevalnega učenja}\label{sec:Teoretično ozadnje}

Spodbujevalno učenje je paradigma strojenega učenja, kjer agent preko interakcij z okolico pridobiva povratno informacijo glede kvalitete svoje trenutne politike odločanja in preko mehanizmov učenja izboljšuje svojo politiko odločanja s ciljem doseganja največje možne nagrade. Tak način strojnega učenja ne potrebuje predpripravljenga nabora učnih podatkov, temveč si učne podatke generira sproti preko interakcij agenta s simulatorjem ter si tako nabira izkušnje.

Agent je na začetku učenja postavljen v njemu neznano okolje, s katerim nima nobenih izkušenj. To okolje je lahko diskretno (Q-učenje) ali pa zvezno, kot je bilo v našem primeru.
Agent nato skozi svoje interakcije z okoljem raziskuje ter pridobiva izkušnje. Agent glede na svojo trenutno odločitveno politiko izvaja akcije v okolju in za to od okolja prejema povratno informacijo v obliki nagrade, ki singalizira kvaliteto odločitve glede na stanje v simulatorju. Na podlagi povratne informacije, prejete nagrade, nato agent opravi učenje in prilagodi svojo odločitveno politiko oziroma parametre v nevronski mreži.

Agent tako deluje v okolju, glede na stanje izvaja akcije, preko katerih preide v novo stanje okolja ter ob temu zbira povratne informacije (nagrade). Ta cikel se ponavlja, dokler se zbiranje zaradi določenega razloga ne zaključi, temu rečemo epizoda učenja. Epizoda predstavlja sekvenco opazovanih značilk stanj okolja, akcije, ki jih je agent izvedel v posameznih stanjih, izračunanih prednosti ter nagrade, ki jih je pridobil za svoje akcije.

Med učenjem agent uporabi zabeležene podatke o stanjih, akcijah agenta in prejetih nagradah in poskuša skozi trening spremeniti svojo politiko odločanja tako, da maksimizira prejeto nagrado. Gre za klasičen primer Markovovega odločitvenega procesa.

Treningi spodbujevalnega učenja se ponavadi izvajajo v namenskih simulatorjih, raje kot pa v fizičnem svetu, saj nam simulatorji omogočajo hitrejši in bolj efektivni trening agentov (omogočajo paralelno poganjanje večih treningov, treniranje s pospešenim časom)





\subsection{Osnove imitacijskega učenja}\label{Spodbujevalno učenje}




Imitacijsko učenje je paradigma strojnega učenja, kjer agent skozi učenje na predpripravljeni učni množici poskuša razpoznati in imitirati glavne vzorce, cilj je, da se agenta nauči odločitveno politiko, ki v situacijah, videnih v učnem setu, prilagodi tako, da se bo odzival podobno kot demonstrator, ki je lahko človek ali pa algoritem. Politika agenta je na začetku treninga inicializirana z naključnimi parametri, skozi trening pa poskuša razpoznavati vzorce iz učne množice, agentova odločitvena funkcija pa spremeni v smer, da v situacijah, podobnih tistim iz učne množice, odločitvena politika vodi agenta v izvajanje akcij, ki so vedno bolj podobne tistim iz učnega nabora. Učni nabor lahko pripravimo bodisi v simulatorju, bodisi preko teleoperacije, kar smo izvedli mi. 

Za uspešno imitacijsko učenje je ključen ustrezen in kvaliteten nabor demonstracij v učnem naboru. V primeru, da v demonstracijah določena območja akcijskgea prostora ali prostora stanj niso bila zajeta, bo to rezultiralo v manj zanesljivih rezultatih agenta. Zato je potreben dovolj raznolik, a hkrati kakovosten učni nabor učnih podatkov.

Velika prednost imitacijskega učenja napram spodbujevalnemu učenju je odsotnost formiranja nagrajevalne funkcije. Imitacijsko učenje je še zelo uporabno pri nalogah, kjer je želeno vedenje lažje demonstrirati kot pa matematično opisati z nagrajevalno funkcijo.
V nalogi smo uporabili pristop imitacijskega učenja, imenovano vedensko kloniranje, ki problem obravnava kod nadzorovano učenje. Ob zajemu učnih podatkov smo zajemali stanja okolja ter akcije, ki smo jih izvajali z igralno palico.



\subsection{Markov odločitveni proces}\label{Markov odločitveni proces}




Markovov odločitveni proces (MDP) je matematična formulacija za predstavitev problema, kjer agent izvaja akcije na podlagi stanj okolja. Formulacija se uporablja v situacijah, kjer se agent sreča z razmerami, v katerih rezultati posameznih akcij niso deterministični, hkrati pa agentova dejanja vplivajo na stanje okolja. MDP je pogosto uporabljen v spodbujevalnem učenju, kjer agent skozi trening išče strategijo, ki bo rezultirala v največji možni prodobiti nagradi.

MDP sestavljajo pet osnovnih komponent: stanja S, v katerih se agent lahko nahaja in v zveznih sistemih predstavlja neskončno množico, množica akcij A, ki jih agent lahko izvede, verjetnosti tranzicij med posameznimi stanji P, nagrada za agenta v posameznem stanju R in diskontni faktor y, ki je ponavadi med 0 in 1. 

Z diskontnim faktorjem uravnavamo, ali dajemo pri učenju povdarek na bolj ali manj preteklih doseženih nagradah. Pri vrednosti faktorja y blizu 1 dajemo povdarek na poznejše nagrade, v kolikor je vrednost y manjša, dajemo večji povdarek na kasneje pridobljene nagrade, v kolikor je vrednost nastavljena na 0, se upošteva samo naslednja pridobljena nagrada.

Prostori stanj in akcij so lahko diskretni ali zvezni (kot v našem primeru), zveznost prostora stanj pa ne nujno zahteva tudi zveznosti prostora akcij.

Agent ob trenutku t začne v stanju s(t), v katerem bo izvedel akcijo za prehod v novo stanje s(t+1). Vsaka akcija prehoda iz stanja s(t) v sledeče stanje s(t+1) je definirana z dinamiko okolja, ki pogojuje verjetnost prehoda, lahko jo zapišemo kot P(s'/s, a). Ko agent preide v novo stanje, za to prejme nagrado r.


Diskontni faktor določa težo prihodnjih nagrad pri izračunu donosa. Pri vrednosti blizu 1 imajo večji vpliv tudi časovno oddaljene nagrade, pri manjši vrednosti pa prevladujejo neposredne nagrade. Če je, se upošteva samo naslednja prejeta nagrada.

Tudi deterministične sisteme, kjer iz vsakega stanja vodi v naslednje stanje akcija z gotovostjo izbire 1, lahko označimo za MDP.

V našem primeru se ukvarjamo s problemom, ki ga lahko razumemo kot problem MDP, naš agent izbira akcije na podlagi politike odločanja, ki izbira akcije a v stanjih s, tako prostor stanj kot akcij sta v našem primeru zvezna. Politika agenta določa izbiro akcije, dinamika sistema P pa na podlagi stanja in akcije določa, v kakšno stanje okolja bo agent prešel po akciji. 

Nač končni cilj je, da skozi trening oblikujemo politiko agenta tako, da bo politika agenta vodila na način, da bo agent maksimiziral doseženo nagrado. Celokupno nagrado posamezne epizode lahko matematično zapišemo kot:


Politiko agenta v našem primeru predstavlja globoka nevronska mreža z učljivimi parametri PI. Učenje politike predstavlja spreminjanje parametrov mreže.




\subsection{Actor-Critic arhitektura}\label{sec:Actor-Critic Arhitektura}



Actor-critic je pristop spodbujevalnega učenja, ki se uporablja za politiko odločanja pri spodbujevalnem učenju in združuje aktorja, ki predstavlja politiko agenta in kritika, ki ocenjuje kakovost stanj oziroma akcij agenta. Implementirana je lahko na različne načine, bodisi z dvema ločenima nevronskima mrežama ali pa kot skupna nevronska mreža z ločenimi izhodnimi glavami za ploitiko agenta in za vrednostno funkcijo. V našem primeru smo arhitekturo implementirali z dvema ločenima nevronskima mrežama. 

Prva nevronska mreža, mreža aktorja, določa akcije agenta in se uči čim bolje voditi agenta v simulatorju. Aktor na podlagi trenutnega stanja okolja določi verjetnostno porazdelitev možnih akcij in iz porazdelitve vzorči. Aktorjevo politkiko lahko zapišemo kot 
\begin{equation}
    \pi_\theta(a_t \mid s_t),
    \label{eq:actor-policy}
\end{equation}
kjer je $s_t$ stanje okolja, $a_t$ izbrana akcija, $\theta$ pa
množica učljivih parametrov aktorjeve nevronske mreže. 


Skozi proces učenja aktor prilagaja verjetnostne porazdelitve akcij v različnih stanjih okolja v želji po maksimizaciji pričakovane skupno pridobljene diskontirane nagrade v epizodi.



Druga mreža v sklopu je kritik, ta je odgovorna za evaluacijo stanj v simulatorju in predvidevanju, kakšna je pričakovana diskontirana skupna nagrada, če se sledi trenutni politiki agenta. Tudi to mrežo skozi trening učimo, skozi trening tako oblikuje model predvidenih nagrad za posamezna stanja v simulatorju. Predstavljamo si, da bo pri treningu strela na gol predvidena nagrada večja, ko bo žoga poleg nasprotnikovega gola in manjša, ko bo žoga daleč od nasprotnikovega gola.
Kritik se uči tako, da primerja svojo napoved vrednosti stanja sistema s pridobljeno nagrado agenta, ki jo prištejemo že prejeti diskuntirani dosedanji akumulirani nagradi. Razlika med tema vrednostima je pogrešek, ki ga s metodo gradientnega spusta poskušamo minimizirati.



Obe mreži torej sodelujeta pri treningu. Mreža aktor glede na svojo trenutno politiko ter stanje v simulatorju izvaja akcije ter za to od okolja pridobiva nagrade, kritik pa glede na stanje ocenjuje pričakovano kumulativno nagrado. S primerjavo dejansko pridobljene nagrade izvedene politike agenta inpričakovane kumulativne nagrade lahko presodimo, ali je bila izbrana akcija boljša ali slabša od pričakovanega s strani kritika. To informacijo mreža aktor uporabi za posodabljanje svoje politike. V primeru, da je bil donos dane akcije boljši od pričakovane, bo svojo politiko prilagodila tako, da bo v podobnih stanjih agent bolj verjetno izvedel podobno akcijo, v primeru, da je bil donos slabši, bo verjetnost za podobno akcijo v sorodnih stanjih manjša.







\subsection{Koncept raziskovanja in izkoriščanja}\label{Spodbujevalno učenje}


Koncept raziskovanja in izkoriščanja (exploration in exploitation) je temeljni koncept v spodbujevalnem učenju, kjer iščemo ravnotežje med izkoriščanjem že preizkušenih, dokazanih akcij ter preizkušanjem še neznanih akcij v situacijah ter odkrivanjem novih akcij za agenta.

Pri izkoriščanju se bo agent obnašal bolj konzervativno, deloval bolj deterministično, pogosteje bo izbiral akcije z visoko verjetnostjo izbire. V analogiji z veliko jezikovnimi modeli bi rekli, da je temperatura nizka.
Pri raziskovanju bi se agent obnašal bolj stohastično, agent bi bolj pogosto preizkušal akcije z nižjo verjetnostjo izbire ter tako pridobival nove izkušnje, s tem bi lahko odkril novo, boljšo politiko od obstoječe.

Ravnotežje med tema pristopoma je ključna za uspešno učenje, skzoi celoten proces je potrebno tako raziskovanje kot izkoriščanje. V kolikor bi agent se agent že v zgodnji fazi učenja obnašal preveč izkoriščevalno, obstaja nevarnost, da bi se agent ujel v lokalni minimum. Če bi se agent vseskozi obnašal preveč raziskovalno, ne bi nikoli konvergiral v končni optimum.

Za primer učenja strela na gol izkoriščanje predstavlja izbiro akcije, ki konsistentno rezultira v zadetku. Raziskovanje pa vključuje preiskušanje različnih moti brce, različnih trenutkov strela in kotov brce. Posamezni poskus je lahko neuspešen, vendar pa skozi proces agent odkrije različne akcije, ki vodijo do uspešnega strela in so v različinh pogojih učinkovite.




\subsection{PPO}\label{sec:PPO}




PPO ali "Proximal policy optimization" je algoritem, ki ga uporablja agent za učenje politike. Gre za gradientno metodo, specifično razvito za spodbujevalno učenje, ki teži k stabilnemu treningu z izogibanjem prevelikih popravkov uteži, saj bi le-te lahko povzročili nestabilen trening. Različica PPO-Clip, ki smo jo uporabili v nalogi, uporablja obrezano ciljno funkcijo, ki zmanjšuje spodbudo za prevelike spremembe politike. S tem spodbuja bolj konservativne posodobitve, vendar ne zagotavlja stroge omejitve velikosti spremembe politike.
Z manjšimi popravki je bolj verjetno, da bo politika agenta konvergirala v optimalno stanje, PPO tako politiko posodablja bolj konzervativno. 

Razmerje med staro in novo politiko lahko izrazimo kot
/enacba/

kjer \(\pi_\theta\) predstavlja trenutno politiko, \(\pi_{\theta_{\mathrm{old}}}\) pa politiko, s katero so bile izkušnje zbrane. Razmerje primerja verjetnosti istega dejanja \(a_t\) v istem stanju \(s_t\). Če je večje od ena, je dejanje po trenutni politiki bolj verjetno, če je manjše od ena, pa manj verjetno. Pri zveznih dejanjih namesto verjetnosti primerjamo gostoti verjetnosti.

PPO torej uporabimo za optimizacijo ciljne funkcije, ki določa, kako spremeniti politiko agenta, kar dosežemo s prilagajanjem parametrov nevronske mreže aktorja v smeri povečevanja ciljne funkcije. 

Ciljno funkcijo lahko izrazimo kot

/enačba/
kjer prvi del funkcije minimuma predstavlja neomejen prispevek ramerja politik in prednosti, drugi del pa omejen prispevek, omejen na interval, kjer je širina intervala določena s hiperparametrov e. Funkcija tako izbere manjšo od dveh vrednosti.





\subsection{Mehanizem prednosti}\label{Optimizacijski algoritem}

vir: https://apxml.com/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/ppo-advantages-returns
Za optimizacijo parametrov s PPO je ključno ugotoviti, koliko je določena akcija v določenem stanju boljša od akcije, ki bi jo v istem stanju izvedla trenutna politika. To je namen funkcije prednosti, ki meri relativno vrednost izvedene akcije glede na predvideno vrednost trenutnega stanja sistema V(st), v kolikor bi akcijo izvedli s trenutno politiko agenta.  

Prednost lahko matematično zapišemo kot
/enačba/
\(\hat A_t
=
r_t+\gamma V(s_{t+1})-V(s_t).\)


kjer lahko člen Q predstavimo kot Oceno Q(s_t, a_t) 

 \(r_t+\gamma V(s_{t+1})\)



kjer Q(s, a) predstavlja predvideno vrednost po izvedbi akcije a iz trenutnega stanja s, v kolikor bi akcijo izvedli glede na trenutno politiko agenta, vrednost V pa nam podaja kritik. Pozitivna prednost pomeni, da je bila akcija boljša od pričakovane, negativna pa, da je bila akcija slabše od pričakovane. PPO za posodobitev svoje politike uporabi oceno prednosti tako, da pri pozitivni vrednosti prednosti povečuje verjetnost izbora dotične akcije, ob negativni prednosti pa verjetnost izbora akcije zmanjša. Uporaba prednost namesto samih donosov, prejetih nagrad zmanjšuje varianco gradientnega posodabljanja politike agenta in prispeva k stabilnejšemu učenju. Sama nagrada ne poda dovolj informacije za učenje, bolj povedna je razlika nagrade glede na pričakovan rezultat.





\subsection{Actor-kritik-Diskriminator arhitektura}\label{sec:Actor-Critic Arhitektura}
vir: https://arxiv.org/abs/1807.06158

V uporabljeni implementaciji imitacijskega učenja uporabljamo tri nevronske mreže, mreži akotr in mreži kritik, ki delujeta enako kot pri imitacijskem učenj, smo dodali mrežo diskriminator, ki na podlagi demonstracij iz učnega nabora epizod dodeljuje nagradni imitacijski signal. 

Tako aktor kot kritik delujeta na enak način kot pri spodbujevalnem učenju, mreža aktor predstavlja politiko odločanja agenta. V stanjih agenta iz verjetnostne porazdelitve akcij vzorči akcije in tako upravlja igralne palice v simulatorju. Cilj aktorja je doseči največji možni diskontni donos. Kritik ocenjuje donos ter sodeluje pri izračunu ocen prednosti za stanja sistema, kar se nato preko PPO optimizacije uporabi v prilagajanju verjetnosti izbora akcij v različnih stanjih sistema. 
 
Diskriminator prejema opazovana stanja sistema, v našem primeru okno opazovanj (5 zaporednih opazovanj) ter na podlagi teh ocenjuje, ali okno opazovanj pripada učni epizodi oziroma epizodi, generirani med treningom, ob temu pa vrača njegovo prepričanje na razponu med 0 in 1 kjer 1 predstavlja največjo možno prepričanost, da okno pripada učni epizodi, 0 pa predstavlja največjo možno prepričanost, da okno pripada trening epizodi. 
Diskriminator torej učimo kot binarni klasifikator, kjer parametre diskriminatorja posodabljamo z minimizacijo binarne križne entropije. IMitacijsko nagrado lahko zapišemo kot:
\(r_t^{\mathrm{imit}}=-\alpha\log\bigl(1-D_\phi(s_t,s_{t+1})\bigr),\)

kjer je alfa faktor skaliranja nagrade, D pa kot izhod diskriminatorja.

Med učenjem izmenično učimo oziroma posodabljamo uteži disktriminatorja na eni strani ter mrež aktorja in kritika na drugi strani. Skozi trening diskriminator vse uspešneje razlikuje med okni prehodov stanj iz učne množice in oknom prehodov stanj učne epizode. Aktor in kritik pa skozi trening tvorita vse bolj podobne prehode med stanji tistim iz učnega nabora podatkov, ki jih diskriminator ocenjuje kot demonstracijske. Tak tekmonalni princip treninga med dvema sklopoma mrež je soroden principu generativnih nasprotujočih mrež (GAN).

Ob idealnem treningu, bi se aktor naučil tako dobro posnemati demonstracijske epizode, da bi diskriminator imel težave z razpoznavanjem pripadnosti posameznega okna prehodov stanj in bi vračal vrednosti okoli 0.5.


\subsection{Curiculum učenje}\label{sec:Curiculum učenje}

Direktno spodbujevalno učenje od začetka do konca v enem samostojnem treningu je ponavadi nepraktično in rezultira v slabših rezultatih. Problemi so potreba po veliki strojni moči, kompleksno oblikovanje nagradne funkcije. Zato agenta namesto naenkrat treniramo v več zaporednih stopnjah, tako imenovanim korikulumnim učenjem. V vsaki posamezni stopnji treniramo agenta z specifično nagradno funkcijo, s katero oblikujemo željeno obnašanje agenta. Z učenjem po etapah tako lažje postopoma usmerjamo agenta do želenega obnašanja, hkrati pa smo bolj fleksibilni pri usmerjanju agenta skozi učenje. Tako sestavljamo kompleksne akcije agenta po komponentah od enostavnejših akcij do kompleksnih nalog.

Do učenja našega agenta bi lahko pristopili na različne načine. Lahko bi mu oblikovali absolutno nagrado za posamezno akcijo, bodisi zadeti gol, prestreči žogo ali kvalitetno podati žogo ter pognati dolgotrajen trening v upanju, da bi se agent skozi veliko število korakov naučil, kako reagirati v različnih pogojih, situacijah. Vendar pa bi za to verjetno potrebovali zelo veliko nevronsko mrežo, trening pa bi trajal zelo dolgo. Vendar pa smo se raje poslužili pristopa, da smo treninge posameznih akcij razdelili na več delov, tako imenovano koračno učenje (curiculum learning). Tako smo agenta zaradi naše arhitekture ločenih akcijskih glav lahko učili po korakih ter tako dosegli, da je agent lahko najprej osvojil osnovne gibe ter skozi korake naučili bolj kompleksnih akcij. Tako smo pridobili večji nadzor nad procesom učenja ter bolj nadzorovano agenta naučili posamezne akcije, vendar pa smo s tem izgubili možnost, da bi agent odkril nepričakovan pristop, ki bi problem lahko rešil bolje kot smo si nastavili učni proces.




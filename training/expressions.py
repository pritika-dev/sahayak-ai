"""
Expression pools for synthetic training-data generation.

WHY THIS FILE EXISTS, AND WHAT IT IS NOT
----------------------------------------
This is NOT a second lexicon. `backend/lexicon.py` holds the phrases the
rule engine matches against; this file holds *different* ways people say
the same things — deliberately written so that most entries here would
NOT match the lexicon. That is the entire point of training a model on
it: the model has to learn the semantics of each category from varied
phrasing, rather than memorising the phrase list the rules already
cover. If these pools duplicated the lexicon, the trained model would
add exactly nothing over a `grep`.

Coverage targeted at the gaps `lexicon.py`'s own docstring names:
tense variants, indirect/hedged disclosure, slang, code-mixing, and
languages beyond English/Hindi (Gujarati is added here and has no
lexicon coverage at all).

HONEST FRAMING: every line below is synthetic — written to resemble
helpline disclosures, not transcribed from real calls. No real caller
data was used. The Hindi and Gujarati sets still need native-speaker
review before any real deployment, exactly as the lexicon does.

Language keys:
  en  — English
  hi  — Hindi, Devanagari script
  gu  — Gujarati, Gujarati script
  rom — romanised Hindi/Gujarati ("Hinglish"), how people actually type
        on a phone keypad. This is the variant with ZERO lexicon
        coverage today and the one a char-n-gram model handles best.
"""

# --------------------------------------------------------------------------
# Per-category expression pools. Each entry is a first-person clause that
# can stand alone or be joined with others.
# --------------------------------------------------------------------------

EXPRESSIONS = {
    "suicidal_ideation": {
        "en": [
            "i keep thinking it would be easier if i just stopped existing",
            "some nights i sit and wonder what the point of any of this is",
            "i have been thinking about ending things for a few weeks now",
            "honestly i dont see myself being here next year",
            "i think everyone would manage fine if i wasnt around",
            "i looked up how many tablets it would take",
            "i wrote a letter to my mother in case something happens to me",
            "there are days i hope i just dont open my eyes in the morning",
            "i am tired of being alive, thats the truth",
            "i have started giving away my things",
            "the thought of finishing it keeps coming back",
            "i stood near the platform edge for a long time yesterday",
            "i dont think i can survive another month like this",
            "i have been cutting myself when it gets bad",
            "part of me has already decided",
            "i keep imagining my own funeral",
            "living feels like a punishment i did not earn",
            "if things dont change i know what i will do",
            "i am scared of what i might do to myself tonight",
            "i have nothing left to hold on for",
        ],
        "hi": [
            "अब और जीने की इच्छा नहीं रह गई है",
            "कई बार लगता है सब खत्म कर दूं",
            "मैं अपने आप को खत्म करने की सोचती रहती हूं",
            "मेरे बिना सब ठीक रहेंगे ऐसा लगता है",
            "सुबह आंख न खुले तो अच्छा है",
            "मैंने सब कुछ सोच लिया है, बस हिम्मत चाहिए",
            "अब सांस लेना भी बोझ लगता है",
            "मैंने चिट्ठी लिखकर रख दी है",
            "रोज सोचती हूं कि खुद को खत्म कर लूं",
            "मुझे डर है कि मैं खुद को कुछ कर लूंगा",
        ],
        "gu": [
            "હવે જીવવાની કોઈ ઈચ્છા બાકી નથી",
            "ઘણી વાર થાય છે કે બધું પતાવી દઉં",
            "મારા વગર બધા સારા રહેશે એવું લાગે છે",
            "સવારે આંખ ન ખૂલે તો સારું",
            "હું મારી જાતને ખતમ કરવાનું વિચારું છું",
            "મેં બધું વિચારી લીધું છે",
            "જીવન એક સજા જેવું લાગે છે",
            "મને ડર છે કે હું મારી જાતને કંઈક કરી બેસીશ",
        ],
        "rom": [
            "ab jeene ka mann nahi karta bilkul",
            "lagta hai sab khatam kar du",
            "mere bina sab theek rahenge",
            "subah aankh na khule to acha hai",
            "maine sab soch liya hai bas himmat chahiye",
            "har roz khud ko khatam karne ka khayal aata hai",
            "hve jivvani ichha nathi rahi",
            "badhu patavi dau evu thay chhe",
            "mujhe dar hai main khud ko kuch kar lungi",
            "jeena ab saza lagti hai",
        ],
    },
    "immediate_danger": {
        "en": [
            "please be quick he is in the next room and the door is open",
            "i can hear him coming up the stairs now",
            "he is outside the door banging on it",
            "i am typing this from the bathroom he is looking for me",
            "he just picked up something from the kitchen",
            "he is in the house i dont have much time",
            "i am locked in and he has the key",
            "he came back drunk ten minutes ago and he is angry",
            "i am hiding right now please help",
            "he is dragging me he wont let go",
            "there is blood on my hands i need someone now",
            "he has something sharp in his hand",
            "he is blocking the only door out",
            "he is on his way here and he said he will finish it today",
        ],
        "hi": [
            "वो अभी बगल के कमरे में है जल्दी कीजिए",
            "वो सीढ़ियों से ऊपर आ रहा है",
            "मैं बाथरूम में छिपकर लिख रही हूं वो मुझे ढूंढ रहा है",
            "उसने अभी रसोई से कुछ उठाया है",
            "वो दरवाजा पीट रहा है",
            "मुझे बंद कर दिया है चाबी उसके पास है",
            "अभी हाथ से खून बह रहा है किसी को भेजिए",
            "वो अभी घर में ही है समय नहीं है",
        ],
        "gu": [
            "એ અત્યારે બાજુના રૂમમાં છે જલદી કરો",
            "એ દાદર ચઢીને આવી રહ્યો છે",
            "હું બાથરૂમમાં સંતાઈને લખું છું",
            "એણે હમણાં રસોડામાંથી કંઈક ઉઠાવ્યું",
            "એ બારણું ધમધમાવે છે",
            "મને પૂરી દીધી છે ચાવી એની પાસે છે",
            "અત્યારે લોહી નીકળે છે કોઈને મોકલો",
        ],
        "rom": [
            "wo abhi bagal ke room me hai jaldi kijiye",
            "wo upar aa raha hai seedhi se",
            "main bathroom me chhup ke likh rahi hu",
            "usne abhi kitchen se kuch uthaya hai",
            "darwaza peet raha hai bahar khada hai",
            "mujhe band kar diya hai chabi uske paas hai",
            "abhi khoon nikal raha hai koi bhejo",
            "e atyare bajuna room ma chhe jaldi karo",
        ],
    },
    "physical_safety_abuse": {
        "en": [
            "last night things got physical again",
            "he raised his hand on me in front of the children",
            "i have marks on my arm from where he held me",
            "it started with pushing and now its more than that",
            "he threw the plate at me during dinner",
            "my lip is swollen i told my office i fell",
            "he shoved me against the wall",
            "this is the third time this month he has done this",
            "i had to go to the clinic after what he did",
            "he grabs me by the hair when he is angry",
            "he pinned me down until i stopped arguing",
            "the bruises are on my back where nobody sees",
            "he twisted my wrist and i could not move it for two days",
            "my husband gets rough with me almost every week",
            "he beat me with his belt",
            "my brother in law has hit me twice now",
            "he does it where the marks dont show",
            "i keep telling people i am clumsy but thats not what happened",
        ],
        "hi": [
            "कल रात फिर हाथ उठाया उसने",
            "बच्चों के सामने मुझ पर हाथ उठाया",
            "मेरी बांह पर निशान हैं जहां उसने पकड़ा था",
            "पहले धक्का देता था अब उससे ज्यादा करता है",
            "खाने के समय उसने थाली फेंक कर मारी",
            "होंठ सूज गया है मैंने ऑफिस में कहा गिर गई थी",
            "गुस्से में बाल पकड़कर खींचता है",
            "इस महीने तीसरी बार ऐसा किया है",
            "बेल्ट से मारा मुझे",
            "जहां निशान नहीं दिखते वहीं मारता है",
        ],
        "gu": [
            "ગઈ રાત્રે ફરી હાથ ઉપાડ્યો",
            "બાળકો સામે મને માર્યું",
            "મારા હાથ પર નિશાન છે",
            "પહેલાં ધક્કો મારતો હતો હવે વધારે કરે છે",
            "જમતી વખતે થાળી ફેંકીને મારી",
            "ગુસ્સામાં વાળ પકડીને ખેંચે છે",
            "આ મહિને ત્રીજી વાર આવું કર્યું",
            "પટ્ટાથી માર્યું મને",
        ],
        "rom": [
            "kal raat phir haath uthaya usne",
            "bachho ke saamne mujhpe haath uthaya",
            "mere haath pe nishan hai jahan pakda tha",
            "pehle dhakka deta tha ab usse zyada karta hai",
            "khane ke time thali fenk ke mari",
            "gusse me baal pakad ke kheenchta hai",
            "is mahine teesri baar aisa kiya",
            "belt se mara mujhe",
            "gai raatre fari haath upadyo",
            "jyaan nishan nathi dekhata tyaan j mare chhe",
        ],
    },
    "sexual_harassment_assault": {
        "en": [
            "he does things at night that i have not agreed to",
            "my manager keeps finding reasons to put his hand on my back",
            "he corners me when nobody else is in the office",
            "something happened two years ago that i have never told anyone",
            "he comes into my room when everyone is asleep",
            "i said no and he did not stop",
            "he keeps sending me pictures i did not ask for",
            "my cousin did something to me when i was younger",
            "he makes comments about my body every single day",
            "i was not able to stop what happened that night",
            "he brushes past me on purpose every time",
            "he told me nobody would believe me if i said anything",
            "i froze and could not say anything while it was happening",
            "he keeps asking me to come to his room alone",
        ],
        "hi": [
            "रात में वो ऐसी चीजें करता है जिसके लिए मैंने हां नहीं कहा",
            "मैनेजर बार बार बहाने से पीठ पर हाथ रखता है",
            "ऑफिस में कोई न हो तो घेर लेता है",
            "दो साल पहले कुछ हुआ था जो मैंने किसी को नहीं बताया",
            "सब सो जाते हैं तो मेरे कमरे में आ जाता है",
            "मैंने मना किया फिर भी नहीं रुका",
            "मेरे शरीर को लेकर रोज टिप्पणी करता है",
            "कहता है कोई मेरी बात पर यकीन नहीं करेगा",
        ],
        "gu": [
            "રાત્રે એ એવું કરે છે જેની મેં હા નથી પાડી",
            "મેનેજર બહાના કરીને પીઠ પર હાથ મૂકે છે",
            "ઓફિસમાં કોઈ ન હોય ત્યારે ઘેરી લે છે",
            "બે વર્ષ પહેલાં કંઈક થયું હતું જે મેં કોઈને કહ્યું નથી",
            "બધા સૂઈ જાય પછી મારા રૂમમાં આવે છે",
            "મેં ના પાડી તોય ન અટક્યો",
            "મારા શરીર વિશે રોજ કોમેન્ટ કરે છે",
        ],
        "rom": [
            "raat me wo aisi cheezein karta hai jiske liye maine haan nahi kaha",
            "manager baar baar bahane se peeth pe haath rakhta hai",
            "office me koi na ho to gher leta hai",
            "sab so jate hain to mere room me aa jata hai",
            "maine mana kiya phir bhi nahi ruka",
            "mere sharir ko lekar roz comment karta hai",
            "kehta hai koi meri baat pe yakeen nahi karega",
            "raatre e evu kare chhe jeni me haa nathi padi",
        ],
    },
    "intimidation_threat": {
        "en": [
            "he said if i go to anyone he knows people who will handle it",
            "every time i pack a bag he reminds me what he will do to my father",
            "he checks my phone every night and asks who i spoke to",
            "i am not allowed to leave without telling him where i am going",
            "he parks outside my office and waits",
            "he said he would put my photos everywhere if i left",
            "he keeps calling from different numbers",
            "he told me the police are his friends so dont bother",
            "he said he would take the children and i would never see them",
            "he turned up at my parents house to make a point",
            "i get messages telling me he knows where i was today",
            "he warned me that talking to a counsellor would be a mistake",
            "he reads every message before i am allowed to send it",
        ],
        "hi": [
            "कहता है किसी को बताया तो मेरे आदमी संभाल लेंगे",
            "जब भी बैग बांधती हूं पिताजी का नाम लेकर धमकाता है",
            "रोज रात को फोन चेक करता है किससे बात की पूछता है",
            "बिना बताए घर से निकलने नहीं देता",
            "ऑफिस के बाहर गाड़ी लगाकर इंतजार करता है",
            "कहता है चली गई तो फोटो सब जगह डाल दूंगा",
            "अलग अलग नंबरों से फोन करता रहता है",
            "कहता है पुलिस उसके जानने वाले हैं",
        ],
        "gu": [
            "કહે છે કોઈને કહ્યું તો મારા માણસો સંભાળી લેશે",
            "બેગ ભરું ત્યારે પપ્પાનું નામ લઈને ધમકાવે છે",
            "રોજ રાત્રે ફોન ચેક કરે છે",
            "કહ્યા વગર બહાર જવા દેતો નથી",
            "ઓફિસ બહાર ગાડી લઈને ઊભો રહે છે",
            "કહે છે જતી રહી તો ફોટા બધે મૂકી દઈશ",
        ],
        "rom": [
            "kehta hai kisi ko bataya to mere aadmi sambhal lenge",
            "jab bhi bag bandhti hu papa ka naam lekar dhamkata hai",
            "roz raat ko phone check karta hai",
            "bina bataye ghar se nikalne nahi deta",
            "office ke bahar gaadi lagakar wait karta hai",
            "kehta hai chali gayi to photo sab jagah daal dunga",
            "alag alag number se call karta rehta hai",
            "kahe chhe koine kahyu to mara mankaso sambhali leshe",
        ],
    },
    "severe_trauma_fear": {
        "en": [
            "my hands start shaking every time i hear the gate",
            "i sleep with the light on and still wake up at every sound",
            "i cannot go past that street anymore",
            "the same scene keeps replaying when i close my eyes",
            "my chest goes tight whenever his name comes up",
            "i have not slept properly in about three weeks",
            "i jump when someone walks up behind me",
            "going home in the evening makes me physically sick",
            "i keep checking the lock four or five times",
            "i cannot be in a room with the door closed anymore",
            "every knock makes me think it is him",
            "i am on edge from the moment i wake up",
        ],
        "hi": [
            "गेट की आवाज सुनते ही हाथ कांपने लगते हैं",
            "लाइट जलाकर सोती हूं फिर भी हर आवाज पर जाग जाती हूं",
            "उस गली से अब नहीं गुजर सकती",
            "आंख बंद करते ही वही सब दिखता है",
            "उसका नाम आते ही सीने में जकड़न हो जाती है",
            "तीन हफ्ते से ठीक से नींद नहीं आई",
            "कोई पीछे से आए तो चौंक जाती हूं",
            "शाम को घर जाने के नाम से घबराहट होती है",
        ],
        "gu": [
            "ગેટનો અવાજ સાંભળતાં જ હાથ ધ્રૂજવા લાગે છે",
            "લાઈટ ચાલુ રાખીને સૂઉં છું તોય જાગી જાઉં છું",
            "એ ગલીમાંથી હવે પસાર નથી થઈ શકતી",
            "આંખ બંધ કરું ત્યાં એ જ દ્રશ્ય દેખાય છે",
            "ત્રણ અઠવાડિયાંથી બરાબર ઊંઘ નથી આવી",
            "કોઈ પાછળથી આવે તો ચમકી જાઉં છું",
        ],
        "rom": [
            "gate ki awaaz sunte hi haath kaanpne lagte hain",
            "light jalakar soti hu phir bhi har awaaz pe jaag jati hu",
            "us gali se ab nahi guzar sakti",
            "aankh band karte hi wahi sab dikhta hai",
            "teen hafte se theek se neend nahi aayi",
            "koi peeche se aaye to chaunk jati hu",
            "shaam ko ghar jane ke naam se ghabrahat hoti hai",
            "gate no awaaj sambhalta j haath dhrujva lage chhe",
        ],
    },
    "depression_hopelessness": {
        "en": [
            "i have not been able to do the smallest things for weeks",
            "nothing really reaches me anymore, good or bad",
            "i used to enjoy cooking and now i just stand there",
            "getting through the day takes everything i have",
            "i cry without any particular reason most days",
            "i have stopped answering my friends messages",
            "i feel like a weight on everyone around me",
            "there is a heaviness i cannot explain to anyone",
            "i dont see this getting better whatever i do",
            "i sleep twelve hours and wake up tired",
            "i have not left the house in about ten days",
            "i keep starting things and abandoning them halfway",
        ],
        "hi": [
            "हफ्तों से छोटे छोटे काम भी नहीं हो पा रहे",
            "अब न अच्छा लगता है न बुरा, कुछ महसूस ही नहीं होता",
            "पहले खाना बनाना अच्छा लगता था अब बस खड़ी रह जाती हूं",
            "दिन काटना ही सबसे मुश्किल काम लगता है",
            "बिना वजह रोना आ जाता है",
            "दोस्तों के मैसेज का जवाब देना बंद कर दिया है",
            "लगता है मैं सबके लिए बोझ हूं",
            "दस दिन से घर से बाहर नहीं निकली",
        ],
        "gu": [
            "અઠવાડિયાંથી નાનાં કામ પણ નથી થતાં",
            "હવે ન સારું લાગે છે ન ખરાબ",
            "દિવસ કાઢવો જ સૌથી અઘરું લાગે છે",
            "કારણ વગર રડવું આવી જાય છે",
            "મિત્રોના મેસેજનો જવાબ આપવાનું બંધ કરી દીધું",
            "લાગે છે હું બધા માટે બોજ છું",
            "દસ દિવસથી ઘરની બહાર નથી નીકળી",
        ],
        "rom": [
            "hafto se chhote chhote kaam bhi nahi ho pa rahe",
            "ab na acha lagta hai na bura kuch mehsoos hi nahi hota",
            "din kaatna hi sabse mushkil kaam lagta hai",
            "bina wajah rona aa jata hai",
            "doston ke message ka jawab dena band kar diya",
            "lagta hai main sabke liye bojh hu",
            "das din se ghar se bahar nahi nikli",
            "divas kaadhvo j sauthi aghru lage chhe",
        ],
    },
    "social_isolation": {
        "en": [
            "there is genuinely nobody i can call about this",
            "my family stopped picking up after the second time i asked for help",
            "he made sure my sister and i stopped speaking",
            "i moved here after marriage and i know nobody in this city",
            "the neighbours hear everything and say nothing",
            "i am not allowed to visit my parents anymore",
            "everyone thinks i am exaggerating so i stopped telling them",
            "i eat alone and i sleep alone and nobody notices either way",
            "the only person i used to talk to has moved abroad",
            "when i told my mother she said to adjust and hung up",
        ],
        "hi": [
            "सच में कोई नहीं है जिसे मैं फोन कर सकूं",
            "दूसरी बार मदद मांगी तो घरवालों ने फोन उठाना बंद कर दिया",
            "उसने मेरी बहन से मेरी बात बंद करवा दी",
            "शादी के बाद यहां आई हूं इस शहर में कोई नहीं जानती",
            "पड़ोसी सब सुनते हैं पर कुछ नहीं कहते",
            "अब मायके जाने नहीं देते",
            "सब कहते हैं मैं बढ़ा चढ़ाकर बोल रही हूं इसलिए बताना छोड़ दिया",
            "मां से कहा तो बोलीं एडजस्ट कर लो और फोन रख दिया",
        ],
        "gu": [
            "ખરેખર કોઈ નથી જેને હું ફોન કરી શકું",
            "બીજી વાર મદદ માંગી તો ઘરવાળાએ ફોન ઉપાડવાનું બંધ કર્યું",
            "લગ્ન પછી અહીં આવી છું કોઈને ઓળખતી નથી",
            "પડોશીઓ બધું સાંભળે છે પણ કંઈ કહેતા નથી",
            "હવે પિયર જવા દેતા નથી",
            "મમ્મીને કહ્યું તો કહે એડજસ્ટ કરી લે",
        ],
        "rom": [
            "sach me koi nahi hai jise main phone kar saku",
            "dusri baar madad maangi to ghar walo ne phone uthana band kar diya",
            "shaadi ke baad yahan aayi hu is sheher me koi nahi jaanti",
            "padosi sab sunte hain par kuch nahi kehte",
            "ab maayke jane nahi dete",
            "maa se kaha to boli adjust kar lo",
            "kharekhar koi nathi jene hu phone kari shaku",
        ],
    },
    "child_safety_concern": {
        "en": [
            "my daughter has started wetting the bed again and she wont say why",
            "my son flinches when his father raises his voice",
            "the school called about marks on my childs arm",
            "he disciplines the children in a way that frightens me",
            "my eight year old asked me if we can live somewhere else",
            "my child has stopped eating and hides in the cupboard",
            "there is a man in the building my daughter is afraid of",
            "my kids saw everything last night and i dont know what to do",
            "my nephew is being beaten at home and nobody will act",
            "my daughter came back from tuition very quiet and wont talk",
        ],
        "hi": [
            "बेटी फिर से बिस्तर गीला करने लगी है और बताती नहीं क्यों",
            "पिता की आवाज ऊंची होते ही बेटा सहम जाता है",
            "स्कूल से फोन आया था बच्चे की बांह पर निशान को लेकर",
            "बच्चों को जिस तरह डांटता है उससे मुझे डर लगता है",
            "आठ साल की बेटी ने पूछा कि क्या हम कहीं और रह सकते हैं",
            "बच्चा खाना छोड़ चुका है और अलमारी में छिप जाता है",
            "कल रात बच्चों ने सब देख लिया",
            "भतीजे को घर में मारते हैं कोई कुछ नहीं करता",
        ],
        "gu": [
            "દીકરી ફરી પથારી ભીની કરવા લાગી છે અને કહેતી નથી કેમ",
            "પપ્પાનો અવાજ ઊંચો થાય કે દીકરો ડરી જાય છે",
            "સ્કૂલમાંથી ફોન આવ્યો બાળકના હાથ પરના નિશાન વિશે",
            "બાળકોને જે રીતે ધમકાવે છે એનાથી મને ડર લાગે છે",
            "બાળક જમવાનું છોડી દીધું છે અને કબાટમાં સંતાઈ જાય છે",
            "ગઈ રાત્રે બાળકોએ બધું જોઈ લીધું",
        ],
        "rom": [
            "beti phir se bistar geela karne lagi hai aur batati nahi kyu",
            "papa ki awaaz unchi hote hi beta seham jata hai",
            "school se phone aaya tha bache ki baanh pe nishan ko lekar",
            "bachho ko jis tarah daantta hai usse mujhe dar lagta hai",
            "bacha khana chhod chuka hai aur almari me chhup jata hai",
            "kal raat bachho ne sab dekh liya",
            "dikri fari pathari bhini karva lagi chhe",
        ],
    },
    "elder_abuse": {
        "en": [
            "my son shouts at my mother and skips her meals when he is angry",
            "nobody has taken my father to the doctor in four months",
            "my daughter in law locks my mother in law in her room",
            "they took my grandmothers pension card and she never sees the money",
            "my parents are being left alone without food for whole days",
            "my elderly aunt is being pressured to sign the house over",
            "they shout at my father and call him a burden in front of guests",
            "my mother is old and they have stopped giving her medicine on time",
        ],
        "hi": [
            "बेटा मां पर चिल्लाता है और गुस्से में खाना नहीं देता",
            "चार महीने से किसी ने पिताजी को डॉक्टर के पास नहीं ले जाया",
            "बहू सास को कमरे में बंद कर देती है",
            "दादी का पेंशन कार्ड ले लिया पैसे उन्हें कभी नहीं मिलते",
            "माता-पिता को पूरे दिन बिना खाने के अकेला छोड़ देते हैं",
            "मेहमानों के सामने पिताजी को बोझ कहकर चिल्लाते हैं",
        ],
        "gu": [
            "દીકરો માં પર બૂમો પાડે છે અને ગુસ્સામાં જમવાનું નથી આપતો",
            "ચાર મહિનાથી કોઈએ પપ્પાને ડોક્ટર પાસે નથી લઈ ગયા",
            "વહુ સાસુને રૂમમાં પૂરી દે છે",
            "દાદીનું પેન્શન કાર્ડ લઈ લીધું પૈસા એમને મળતા નથી",
            "માતા-પિતાને આખો દિવસ ભૂખ્યા એકલા મૂકી દે છે",
        ],
        "rom": [
            "beta maa par chillata hai aur gusse me khana nahi deta",
            "chaar mahine se kisi ne papa ko doctor ke paas nahi le jaya",
            "bahu saas ko room me band kar deti hai",
            "dadi ka pension card le liya paise unhe kabhi nahi milte",
            "maa baap ko pure din bina khane ke akela chhod dete hain",
            "vahu saasu ne room ma puri de chhe",
        ],
    },
    "financial_exploitation": {
        "en": [
            "my salary goes straight to his account and i have to ask for bus fare",
            "they have been asking for a car since the engagement",
            "he made me sign papers i was not allowed to read",
            "my in laws sold my gold without telling me",
            "i am not permitted to keep a bank account of my own",
            "he took a loan in my name and stopped paying it",
            "they keep saying the dowry was not enough",
            "i worked for eleven years and i have nothing in my name",
            "he gives me a hundred rupees a week and counts what i spent",
            "my brother has taken control of my late husbands money",
        ],
        "hi": [
            "मेरी तनख्वाह सीधे उसके खाते में जाती है और बस के पैसे भी मांगने पड़ते हैं",
            "सगाई के बाद से गाड़ी की मांग कर रहे हैं",
            "कागजों पर दस्तखत करवाए पढ़ने भी नहीं दिया",
            "ससुराल वालों ने मेरा सोना बिना बताए बेच दिया",
            "अपना बैंक खाता रखने नहीं देते",
            "मेरे नाम पर कर्ज लिया और चुकाना बंद कर दिया",
            "कहते रहते हैं दहेज कम मिला",
            "ग्यारह साल काम किया मेरे नाम पर कुछ नहीं है",
        ],
        "gu": [
            "મારો પગાર સીધો એના ખાતામાં જાય છે અને બસના પૈસા પણ માંગવા પડે છે",
            "સગાઈ પછીથી ગાડીની માંગણી કરે છે",
            "કાગળો પર સહી કરાવી વાંચવા પણ ન દીધું",
            "સાસરિયાંએ મારું સોનું કહ્યા વગર વેચી દીધું",
            "મારું પોતાનું બેંક ખાતું રાખવા દેતા નથી",
            "મારા નામે લોન લીધી અને ભરવાનું બંધ કર્યું",
        ],
        "rom": [
            "meri tankhwah seedhe uske account me jati hai aur bus ke paise bhi maangne padte hain",
            "sagai ke baad se gaadi ki demand kar rahe hain",
            "kaagzo pe dastkhat karwaye padhne bhi nahi diya",
            "sasural walo ne mera sona bina bataye bech diya",
            "apna bank account rakhne nahi dete",
            "mere naam pe loan liya aur chukana band kar diya",
            "maro pagaar sidho ena khatama jay chhe",
        ],
    },
    "legal_dispute": {
        "en": [
            "my brothers have transferred the ancestral plot into their names",
            "i went to the station three times and they refused to write it down",
            "the landlord has given me seven days to vacate with no notice period",
            "my late husbands family is contesting the will",
            "i need to know how to apply for maintenance",
            "the builder has not given possession for three years",
            "i want to file for custody but i dont know where to begin",
            "the panchayat ruled against me without hearing my side",
            "i cannot afford a private advocate and the free one never shows up",
            "someone forged my thumb impression on the sale deed",
            "my name was removed from the land records",
            "i have a summons and i dont understand what it says",
        ],
        "hi": [
            "भाइयों ने पुश्तैनी जमीन अपने नाम करवा ली",
            "तीन बार थाने गई पर रिपोर्ट नहीं लिखी",
            "मकान मालिक ने सात दिन में घर खाली करने को कहा है",
            "पति के जाने के बाद ससुराल वाले वसीयत पर सवाल उठा रहे हैं",
            "मुझे गुजारा भत्ता के लिए आवेदन करना है",
            "बिल्डर ने तीन साल से कब्जा नहीं दिया",
            "बच्चे की कस्टडी के लिए अर्जी देनी है समझ नहीं आ रहा कहां से शुरू करूं",
            "वकील का खर्च नहीं उठा सकती और सरकारी वकील आता ही नहीं",
            "जमीन के कागजों से मेरा नाम हटा दिया गया",
        ],
        "gu": [
            "ભાઈઓએ વડીલોપાર્જિત જમીન પોતાના નામે કરાવી લીધી",
            "ત્રણ વાર પોલીસ સ્ટેશન ગઈ પણ ફરિયાદ ન લખી",
            "મકાનમાલિકે સાત દિવસમાં ઘર ખાલી કરવા કહ્યું છે",
            "મારે ભરણપોષણ માટે અરજી કરવી છે",
            "બિલ્ડરે ત્રણ વર્ષથી કબજો નથી આપ્યો",
            "વકીલનો ખર્ચ ઉઠાવી શકતી નથી",
            "જમીનના કાગળમાંથી મારું નામ કાઢી નાખ્યું",
        ],
        "rom": [
            "bhaiyo ne pushtaini zameen apne naam karwa li",
            "teen baar thane gayi par report nahi likhi",
            "makaan malik ne saat din me ghar khali karne ko kaha hai",
            "mujhe guzara bhatta ke liye apply karna hai",
            "builder ne teen saal se kabza nahi diya",
            "wakeel ka kharcha nahi utha sakti",
            "zameen ke kaagzo se mera naam hata diya",
            "bhaioe vadilo parjit jameen potana name karavi lidhi",
        ],
    },
    "workplace_harassment": {
        "en": [
            "my supervisor keeps me back after hours for no real reason",
            "i raised it with hr in march and nothing has happened since",
            "they cut my name from the roster after i complained",
            "my team lead makes remarks about my caste in meetings",
            "i was let go the week after i reported him",
            "they have stopped paying me for the last two months",
            "the site manager shouts abuse at the women workers",
            "i am given impossible targets so they have a reason to fire me",
            "my complaint letter was returned to me by the office itself",
            "the internal committee meeting keeps getting postponed",
        ],
        "hi": [
            "सुपरवाइजर बिना वजह देर रात तक रोक कर रखता है",
            "मार्च में एचआर को बताया था तब से कुछ नहीं हुआ",
            "शिकायत करने के बाद मेरा नाम ड्यूटी लिस्ट से हटा दिया",
            "मीटिंग में मेरी जाति को लेकर टिप्पणी करता है",
            "उसकी शिकायत करने के अगले हफ्ते नौकरी से निकाल दिया",
            "दो महीने से तनख्वाह नहीं दी",
            "साइट मैनेजर महिला मजदूरों को गालियां देता है",
            "आंतरिक समिति की बैठक बार बार टल रही है",
        ],
        "gu": [
            "સુપરવાઈઝર કારણ વગર મોડે સુધી રોકી રાખે છે",
            "માર્ચમાં એચઆરને કહ્યું ત્યારથી કંઈ થયું નથી",
            "ફરિયાદ કર્યા પછી મારું નામ ડ્યુટી લિસ્ટમાંથી કાઢી નાખ્યું",
            "એની ફરિયાદ કર્યાના અઠવાડિયામાં નોકરીમાંથી કાઢી મૂકી",
            "બે મહિનાથી પગાર નથી આપ્યો",
            "સાઈટ મેનેજર મહિલા મજૂરોને ગાળો દે છે",
        ],
        "rom": [
            "supervisor bina wajah der raat tak rok kar rakhta hai",
            "march me hr ko bataya tha tab se kuch nahi hua",
            "shikayat karne ke baad mera naam duty list se hata diya",
            "uski shikayat karne ke agle hafte naukri se nikal diya",
            "do mahine se tankhwah nahi di",
            "site manager mahila mazdooro ko gaaliyan deta hai",
            "supervisor karan vagar mode sudhi roki rakhe chhe",
        ],
    },
    "cybercrime_online_abuse": {
        "en": [
            "someone made an account with my photo and my name",
            "he is threatening to send private pictures to my relatives",
            "a stranger got into my email and changed the password",
            "there is a video of me circulating in a whatsapp group",
            "i get abusive messages from new accounts every single day",
            "he morphed my face onto other pictures and shared them",
            "someone is posting my phone number on classified sites",
            "i sent photos to someone i trusted and now he wants money",
            "they are tagging my employer in posts about me",
            "my whatsapp was cloned and messages went out in my name",
        ],
        "hi": [
            "किसी ने मेरी फोटो और नाम से अकाउंट बना लिया",
            "निजी तस्वीरें रिश्तेदारों को भेजने की धमकी दे रहा है",
            "किसी ने मेरा ईमेल खोलकर पासवर्ड बदल दिया",
            "मेरा वीडियो एक व्हाट्सएप ग्रुप में घूम रहा है",
            "रोज नए अकाउंट से गालियां भेजते हैं",
            "मेरा चेहरा दूसरी तस्वीरों पर लगाकर फैला दिया",
            "मेरा नंबर वेबसाइटों पर डाल रहे हैं",
            "भरोसा करके फोटो भेजी थी अब पैसे मांग रहा है",
        ],
        "gu": [
            "કોઈએ મારા ફોટા અને નામથી એકાઉન્ટ બનાવ્યું",
            "ખાનગી ફોટા સગાંઓને મોકલવાની ધમકી આપે છે",
            "કોઈએ મારું ઈમેલ ખોલીને પાસવર્ડ બદલી નાખ્યો",
            "મારો વીડિયો વોટ્સએપ ગ્રુપમાં ફરે છે",
            "રોજ નવા એકાઉન્ટથી ગાળો મોકલે છે",
            "ભરોસો કરીને ફોટો મોકલ્યો હતો હવે પૈસા માંગે છે",
        ],
        "rom": [
            "kisi ne meri photo aur naam se account bana liya",
            "private photos rishtedaro ko bhejne ki dhamki de raha hai",
            "kisi ne mera email khol kar password badal diya",
            "mera video ek whatsapp group me ghoom raha hai",
            "roz naye account se gaaliyan bhejte hain",
            "bharosa karke photo bheji thi ab paise maang raha hai",
            "koie mara photo ane name thi account banavyu",
        ],
    },
    "substance_related": {
        "en": [
            "he is fine until about nine at night and then it changes",
            "the trouble only starts after he has been drinking",
            "he has been using something and he is not himself anymore",
            "my son sold the television to pay for his habit",
            "he spends the whole wage at the shop before he reaches home",
            "when he is sober he apologises and then it happens again",
            "there are bottles hidden all over the house",
            "my husband drinks daily and the shouting follows",
        ],
        "hi": [
            "रात नौ बजे तक ठीक रहता है उसके बाद सब बदल जाता है",
            "पीने के बाद ही सारा झगड़ा शुरू होता है",
            "कुछ नशा करने लगा है अब पहले जैसा नहीं रहा",
            "बेटे ने नशे के लिए टीवी बेच दिया",
            "घर पहुंचने से पहले पूरी मजदूरी दुकान पर खर्च कर देता है",
            "होश में आता है तो माफी मांगता है फिर वही करता है",
            "घर में जगह जगह बोतलें छिपाकर रखी हैं",
        ],
        "gu": [
            "રાત નવ વાગ્યા સુધી ઠીક રહે છે પછી બધું બદલાઈ જાય છે",
            "પીધા પછી જ ઝઘડો શરૂ થાય છે",
            "કંઈક નશો કરવા લાગ્યો છે હવે પહેલા જેવો નથી",
            "દીકરાએ નશા માટે ટીવી વેચી દીધું",
            "ઘરે પહોંચતા પહેલાં આખી મજૂરી દુકાને ખર્ચી નાખે છે",
        ],
        "rom": [
            "raat nau baje tak theek rehta hai uske baad sab badal jata hai",
            "peene ke baad hi saara jhagda shuru hota hai",
            "kuch nasha karne laga hai ab pehle jaisa nahi raha",
            "bete ne nashe ke liye tv bech diya",
            "ghar pahunchne se pehle puri mazdoori dukaan pe kharch kar deta hai",
            "pidha pachhi j jhagdo sharu thay chhe",
        ],
    },
    "vulnerability_context": {
        "en": [
            "i am five months along and i have nowhere to go",
            "i use a wheelchair so i cannot simply walk out",
            "i do not speak the local language and nobody translates for me",
            "i am seventy one and i depend on them for everything",
            "i have been sleeping at the bus stand for four nights",
            "i am on a dependent visa so my status is tied to his",
            "i am the only earning person for four people",
            "i cannot read the forms myself",
            "i am new to this city and i have no documents with me",
            "i am staying in a shelter with my two small children",
        ],
        "hi": [
            "मैं पांच महीने की गर्भवती हूं और जाने के लिए कोई जगह नहीं है",
            "मैं व्हीलचेयर पर हूं इसलिए ऐसे ही निकल नहीं सकती",
            "यहां की भाषा नहीं आती कोई अनुवाद भी नहीं करता",
            "मैं इकहत्तर साल की हूं और हर चीज के लिए उन पर निर्भर हूं",
            "चार रात से बस स्टैंड पर सो रही हूं",
            "चार लोगों में कमाने वाली अकेली मैं हूं",
            "मैं खुद फॉर्म नहीं पढ़ सकती",
            "दो छोटे बच्चों के साथ आश्रय गृह में हूं",
        ],
        "gu": [
            "હું પાંચ મહિનાની ગર્ભવતી છું અને જવા માટે કોઈ જગ્યા નથી",
            "હું વ્હીલચેરમાં છું એટલે એમ જ નીકળી શકતી નથી",
            "અહીંની ભાષા આવડતી નથી",
            "હું એકોતેર વર્ષની છું અને બધા માટે એમના પર નિર્ભર છું",
            "ચાર રાતથી બસ સ્ટેન્ડ પર સૂઉં છું",
            "બે નાનાં બાળકો સાથે આશ્રયગૃહમાં છું",
        ],
        "rom": [
            "main paanch mahine ki pregnant hu aur jane ke liye koi jagah nahi",
            "main wheelchair pe hu isliye aise hi nikal nahi sakti",
            "yahan ki bhasha nahi aati koi translate bhi nahi karta",
            "main ikhattar saal ki hu aur har cheez ke liye un par nirbhar hu",
            "chaar raat se bus stand pe so rahi hu",
            "do chhote bachho ke saath shelter home me hu",
            "hu paanch mahina ni garbhvati chhu",
        ],
    },
}

# --------------------------------------------------------------------------
# Neutral / low-risk material. Without a solid negative class the model
# learns "any helpline message is severe", which destroys precision and
# floods the authority queue — the failure mode that matters most for a
# triage tool a human has to work through.
# --------------------------------------------------------------------------

NEUTRAL = {
    "en": [
        "hello is this the helpline number",
        "what are your working hours",
        "i wanted to ask what documents are needed for the application",
        "can someone call me back in the evening please",
        "i am calling on behalf of a neighbour who wanted information",
        "is this service free of charge",
        "do you have an office in this district",
        "i had registered a complaint last week and wanted an update",
        "thank you for your help earlier, things are better now",
        "can i get the details in writing by email",
        "which department handles ration card issues",
        "i am a student doing a project on support services",
        "my friend suggested i call and just talk to someone",
        "work has been busy and tiring but manageable",
        "i wanted to know if counselling is available in gujarati",
        "how long does the process usually take",
        "i think i dialled the wrong number, sorry",
        "can you tell me the address of the nearest centre",
        "everything is fine at home, i just had a general question",
        "i want to volunteer with your organisation",
    ],
    "hi": [
        "नमस्ते क्या यह हेल्पलाइन नंबर है",
        "आपके काम करने का समय क्या है",
        "आवेदन के लिए कौन से कागज चाहिए",
        "शाम को कोई मुझे वापस फोन कर सकता है",
        "मैं पड़ोसी के लिए जानकारी लेने फोन कर रही हूं",
        "क्या यह सेवा मुफ्त है",
        "क्या इस जिले में आपका दफ्तर है",
        "पिछले हफ्ते शिकायत दर्ज की थी उसकी जानकारी चाहिए",
        "पहले मदद के लिए धन्यवाद अब सब ठीक है",
        "क्या गुजराती में परामर्श उपलब्ध है",
        "इस काम में कितना समय लगता है",
        "घर पर सब ठीक है बस एक सामान्य सवाल था",
    ],
    "gu": [
        "નમસ્તે શું આ હેલ્પલાઈન નંબર છે",
        "તમારો કામ કરવાનો સમય શું છે",
        "અરજી માટે કયા કાગળો જોઈએ",
        "સાંજે કોઈ મને પાછો ફોન કરી શકે",
        "શું આ સેવા મફત છે",
        "શું આ જિલ્લામાં તમારી ઓફિસ છે",
        "ગયા અઠવાડિયે ફરિયાદ નોંધાવી હતી એની માહિતી જોઈએ",
        "અગાઉની મદદ બદલ આભાર હવે બધું સારું છે",
        "આ કામમાં કેટલો સમય લાગે છે",
        "ઘરે બધું સારું છે બસ એક સામાન્ય પ્રશ્ન હતો",
    ],
    "rom": [
        "namaste kya yeh helpline number hai",
        "aapke kaam karne ka samay kya hai",
        "application ke liye kaunse kaagaz chahiye",
        "shaam ko koi mujhe wapas phone kar sakta hai",
        "kya yeh service free hai",
        "kya is district me aapka office hai",
        "pichle hafte shikayat darj ki thi uski jankari chahiye",
        "pehle madad ke liye dhanyavaad ab sab theek hai",
        "kya gujarati me counselling available hai",
        "ghar par sab theek hai bas ek general sawaal tha",
        "tamaro kaam karvano samay shu chhe",
        "aa seva muft chhe ke nahi",
    ],
}

# Openers and closers that carry no severity signal of their own. They
# exist so the model does not learn that "short message = severe";
# real disclosures arrive wrapped in hesitation and politeness.
OPENERS = {
    "en": [
        "hello, i need to talk to someone.", "sorry to bother you.",
        "i dont know if this is the right place but",
        "i have never called anything like this before.",
        "can i ask something.", "please dont tell anyone i called.",
        "i am not sure how to say this.", "madam, please listen.",
        "i have been thinking about calling for months.",
        "i only have a few minutes to talk.",
    ],
    "hi": [
        "नमस्ते मुझे किसी से बात करनी है।", "माफ कीजिए परेशान कर रही हूं।",
        "पता नहीं यह सही जगह है या नहीं पर", "मैंने पहले कभी ऐसे फोन नहीं किया।",
        "एक बात पूछनी थी।", "किसी को मत बताइएगा कि मैंने फोन किया।",
        "समझ नहीं आ रहा कैसे बताऊं।", "मैडम जी सुनिए जरा।",
    ],
    "gu": [
        "નમસ્તે મારે કોઈ સાથે વાત કરવી છે।", "માફ કરજો તકલીફ આપું છું।",
        "ખબર નથી આ સાચી જગ્યા છે કે નહીં પણ", "મેં પહેલાં ક્યારેય આવો ફોન નથી કર્યો।",
        "એક વાત પૂછવી હતી।", "કોઈને કહેતા નહીં કે મેં ફોન કર્યો।",
        "સમજાતું નથી કેવી રીતે કહું।",
    ],
    "rom": [
        "namaste mujhe kisi se baat karni hai.", "maaf kijiye pareshan kar rahi hu.",
        "pata nahi yeh sahi jagah hai ya nahi par", "maine pehle kabhi aise phone nahi kiya.",
        "ek baat puchni thi.", "kisi ko mat bataiyega ki maine phone kiya.",
        "samajh nahi aa raha kaise bataun.", "maaf karjo taklif aapu chhu.",
    ],
}

CLOSERS = {
    "en": [
        "please help me.", "i dont know what to do now.",
        "what are my options.", "is there anything you can do.",
        "please tell me who i should speak to.", "thats all i wanted to say.",
        "sorry for taking your time.", "i just needed someone to hear this.",
    ],
    "hi": [
        "कृपया मेरी मदद कीजिए।", "अब क्या करूं समझ नहीं आ रहा।",
        "मेरे पास क्या रास्ते हैं।", "क्या आप कुछ कर सकते हैं।",
        "बताइए मुझे किससे बात करनी चाहिए।", "बस इतना ही कहना था।",
        "आपका समय लेने के लिए माफी।",
    ],
    "gu": [
        "કૃપા કરીને મારી મદદ કરો।", "હવે શું કરું સમજાતું નથી।",
        "મારી પાસે કયા રસ્તા છે।", "શું તમે કંઈ કરી શકો।",
        "કહો મારે કોની સાથે વાત કરવી જોઈએ।", "બસ આટલું જ કહેવું હતું।",
    ],
    "rom": [
        "kripya meri madad kijiye.", "ab kya karu samajh nahi aa raha.",
        "mere paas kya raste hain.", "kya aap kuch kar sakte hain.",
        "bataiye mujhe kisse baat karni chahiye.", "bas itna hi kehna tha.",
        "hve shu karu samjatu nathi.",
    ],
}

# Negation frames. The model must learn the same distinction the rule
# engine's _NEGATION_WORDS window handles: "i am not going to hurt
# myself" is not a disclosure of suicidal ideation. Rules do this with a
# 4-token lookbehind, which breaks on longer-range negation; the model
# sees the whole sentence, so it should do better here.
NEGATION_FRAMES = {
    "en": [
        "i want to be clear that it is not the case that {x}",
        "people assume {x} but that is not my situation at all",
        "thankfully nothing like this — {x} — has ever happened to me",
        "i am calling about my sister, not myself, so no, {x} is not about me",
        "just so you know, it is not true that {x}",
        "i should say straight away that it is not true that {x}",
        "the counsellor asked whether {x} and the answer is no",
        "i am not saying {x}, i want to be clear about that",
        "it would be wrong to write down that {x}",
        "no, {x} has never happened and i do not want that recorded",
        "everyone keeps asking if {x}, and it is not like that",
        "please do not note that {x}, because it is not accurate",
        "i read a pamphlet that said {x}, but none of that applies to me",
        "my situation is not the one where {x}",
        "to be fair to him, it is not as if {x}",
        "i want to correct something i said earlier — it is not the case that {x}",
    ],
    "hi": [
        "मैं साफ कर दूं कि ऐसा नहीं है कि {x}",
        "लोग समझते हैं कि {x} पर मेरे साथ ऐसा कुछ नहीं है",
        "शुक्र है ऐसा कुछ नहीं हुआ, {x} वाली बात बिल्कुल नहीं है",
        "मैं अपनी बहन के लिए फोन कर रही हूं, मेरे साथ {x} जैसा कुछ नहीं",
        "पहले ही बता दूं कि यह सच नहीं है कि {x}",
        "काउंसलर ने पूछा था कि {x}, जवाब है नहीं",
        "मैं यह नहीं कह रही कि {x}, यह साफ कर देना चाहती हूं",
        "यह लिखना गलत होगा कि {x}",
        "नहीं, {x} कभी नहीं हुआ और मैं यह दर्ज नहीं करवाना चाहती",
        "सब पूछते रहते हैं कि {x}, पर ऐसा कुछ नहीं है",
        "कृपया यह मत लिखिए कि {x}, यह सही नहीं है",
        "मेरी हालत वैसी नहीं है जैसी {x}",
        "उसके बारे में सच कहूं तो ऐसा नहीं है कि {x}",
    ],
    "gu": [
        "હું સ્પષ્ટ કરું કે એવું નથી કે {x}",
        "લોકો માને છે કે {x} પણ મારી સાથે એવું કંઈ નથી",
        "સારું છે કે એવું કંઈ નથી થયું, {x} એવી વાત નથી",
        "પહેલાં જ કહી દઉં કે એ સાચું નથી કે {x}",
        "કાઉન્સેલરે પૂછ્યું હતું કે {x}, જવાબ ના છે",
        "હું એવું નથી કહેતી કે {x}, એ સ્પષ્ટ કરી દઉં",
        "આવું લખવું ખોટું ગણાશે કે {x}",
        "ના, {x} ક્યારેય નથી થયું અને મારે એ નોંધાવવું નથી",
        "બધા પૂછ્યા કરે છે કે {x}, પણ એવું કંઈ નથી",
        "કૃપા કરીને એ ન લખશો કે {x}, એ સાચું નથી",
        "મારી પરિસ્થિતિ એવી નથી જેમાં {x}",
    ],
    "rom": [
        "main saaf kar du ki aisa nahi hai ki {x}",
        "log samajhte hain ki {x} par mere saath aisa kuch nahi hai",
        "shukr hai aisa kuch nahi hua, {x} wali baat bilkul nahi hai",
        "pehle hi bata du ki yeh sach nahi hai ki {x}",
        "counsellor ne pucha tha ki {x}, jawab hai nahi",
        "main yeh nahi keh rahi ki {x}, yeh saaf kar dena chahti hu",
        "yeh likhna galat hoga ki {x}",
        "nahi, {x} kabhi nahi hua aur main yeh darj nahi karwana chahti",
        "sab puchte rehte hain ki {x}, par aisa kuch nahi hai",
        "kripya yeh mat likhiye ki {x}, yeh sahi nahi hai",
        "meri haalat waisi nahi hai jaisi {x}",
        "hu evu nathi kehti ke {x}, e spasht kari dau",
        "na, {x} kyarey nathi thayu",
    ],
}

# Hedge frames. A hedged disclosure still counts — at reduced
# confidence, matching the rule engine's 0.5 for _HEDGE_WORDS.
HEDGE_FRAMES = {
    "en": [
        "i might be overthinking it but {x}",
        "maybe i am wrong, still, {x}",
        "i am not certain, it could be that {x}",
        "i keep wondering whether {x}",
        "perhaps it is nothing, but {x}",
    ],
    "hi": [
        "शायद मैं ज्यादा सोच रही हूं पर {x}",
        "हो सकता है मैं गलत हूं फिर भी {x}",
        "पक्का नहीं है पर लगता है कि {x}",
        "बार बार लगता है कि {x}",
    ],
    "gu": [
        "કદાચ હું વધારે વિચારું છું પણ {x}",
        "બની શકે હું ખોટી હોઉં તોય {x}",
        "પાક્કું નથી પણ લાગે છે કે {x}",
    ],
    "rom": [
        "shayad main zyada soch rahi hu par {x}",
        "ho sakta hai main galat hu phir bhi {x}",
        "pakka nahi hai par lagta hai ki {x}",
        "kadach hu vadhare vicharu chhu pan {x}",
    ],
}

# Third-person reframing — a very common helpline pattern (calling for a
# friend/neighbour/relative). The disclosure is just as real; the
# grammar is different, which is exactly the kind of variation a phrase
# list handles badly.
THIRD_PERSON_FRAMES = {
    "en": [
        "my neighbour told me that {x}",
        "i am calling for my sister — in her words, {x}",
        "a woman in our building is in this situation: {x}",
        "my friend asked me to call because {x}",
    ],
    "hi": [
        "मेरी पड़ोसन ने बताया कि {x}",
        "मैं अपनी बहन के लिए फोन कर रही हूं, उसके शब्दों में {x}",
        "हमारी बिल्डिंग में एक महिला की यही हालत है, {x}",
    ],
    "gu": [
        "મારી પડોશણે કહ્યું કે {x}",
        "હું મારી બહેન માટે ફોન કરું છું, એના શબ્દોમાં {x}",
        "અમારી બિલ્ડિંગમાં એક બહેનની આ જ હાલત છે, {x}",
    ],
    "rom": [
        "meri padosan ne bataya ki {x}",
        "main apni behen ke liye phone kar rahi hu, uske shabdo me {x}",
        "hamari building me ek mahila ki yahi haalat hai, {x}",
    ],
}

# Realistic co-occurrence. Helpline disclosures are rarely single-issue:
# physical abuse arrives with fear and isolation, dowry demands arrive
# with threats. Sampling category combinations uniformly would teach the
# model an independence structure the real domain does not have.
CO_OCCURRENCE = {
    "physical_safety_abuse": ["severe_trauma_fear", "intimidation_threat", "social_isolation",
                              "substance_related", "child_safety_concern", "vulnerability_context"],
    "sexual_harassment_assault": ["severe_trauma_fear", "intimidation_threat", "workplace_harassment",
                                  "depression_hopelessness", "social_isolation"],
    "suicidal_ideation": ["depression_hopelessness", "social_isolation", "severe_trauma_fear",
                          "physical_safety_abuse"],
    "immediate_danger": ["physical_safety_abuse", "severe_trauma_fear", "child_safety_concern"],
    "intimidation_threat": ["severe_trauma_fear", "cybercrime_online_abuse", "financial_exploitation",
                            "social_isolation"],
    "financial_exploitation": ["intimidation_threat", "legal_dispute", "physical_safety_abuse",
                               "social_isolation"],
    "legal_dispute": ["financial_exploitation", "elder_abuse", "vulnerability_context"],
    "workplace_harassment": ["sexual_harassment_assault", "depression_hopelessness", "intimidation_threat"],
    "cybercrime_online_abuse": ["intimidation_threat", "severe_trauma_fear", "depression_hopelessness"],
    "child_safety_concern": ["physical_safety_abuse", "severe_trauma_fear", "substance_related"],
    "elder_abuse": ["financial_exploitation", "social_isolation", "legal_dispute"],
    "substance_related": ["physical_safety_abuse", "intimidation_threat", "financial_exploitation"],
    "depression_hopelessness": ["social_isolation", "suicidal_ideation", "severe_trauma_fear"],
    "social_isolation": ["depression_hopelessness", "vulnerability_context"],
    "severe_trauma_fear": ["depression_hopelessness", "social_isolation"],
    "vulnerability_context": ["social_isolation", "financial_exploitation"],
}

# Joining words, so multi-category samples read as one utterance rather
# than concatenated fragments.
JOINERS = {
    "en": [" and ", ". ", ", and on top of that ", ". also ", ", plus "],
    "hi": [" और ", "। ", ", और इसके अलावा ", "। साथ ही ", ", और "],
    "gu": [" અને ", "। ", ", અને એ ઉપરાંત ", "। સાથે જ ", ", અને "],
    "rom": [" aur ", ". ", ", aur uske alawa ", ". saath hi ", ", plus "],
}

LANGUAGES = ["en", "hi", "gu", "rom"]
LANGUAGE_MIX = [0.42, 0.20, 0.13, 0.25]


# --------------------------------------------------------------------------
# SIH26093 atrocity categories (Round 19) — kept in their own file so the
# additions are easy to review. See atrocity_data.py.
# --------------------------------------------------------------------------
from atrocity_data import ATROCITY_EXPRESSIONS, ATROCITY_NEUTRAL, ATROCITY_CO_OCCURRENCE, HARD_NEGATIVES  # noqa: E402

EXPRESSIONS.update(ATROCITY_EXPRESSIONS)
for _lang, _lines in ATROCITY_NEUTRAL.items():
    NEUTRAL[_lang] = list(NEUTRAL[_lang]) + list(_lines) + list(HARD_NEGATIVES.get(_lang, []))
CO_OCCURRENCE.update(ATROCITY_CO_OCCURRENCE)

"""
Severity + situation-type lexicon for text-based triage.

IMPORTANT: this is a larger but still illustrative demo lexicon, not a
validated clinical/legal instrument. Before any real use this needs:
  - review by a mental-health professional AND a legal-aid professional
  - EVERY Hindi phrase below reviewed/expanded by a native speaker —
    all 14 categories now have a starting Hindi phrase set (the last 6
    — legal, workplace, cybercrime, child-safety, elder-abuse, and
    substance — were added in Round 16 and are the least tested of the
    set), but none of them are a substitute for a native speaker's
    review, and no language beyond English/Hindi has any lexicon
    coverage at all
  - regular calibration against real (anonymized, consented) transcripts
  - this stays exact/fuzzy PHRASE matching, not real NLP — it will
    keep having gaps for tense, phrasing, and slang variants no matter
    how many phrases are added; only semantic/embedding-based matching
    (a real model, not a phrase list) would genuinely close that, and
    that's out of scope for what this lexicon file can do

Each category has:
  - weight: severity contribution (0-1) if this is about psychological/
    physical risk. Categories that are mainly about *what kind of help*
    someone needs (legal, financial) can have a lower weight but still
    drive routing via SERVICE_TAGS in scoring.py — severity and
    service-need are tracked separately on purpose (see round-3 fix).
  - phrases: matched with word-level typo tolerance (see
    _fuzzy_phrase_in_tokens) after normalization, so "don't"/"dont" and
    minor misspellings like "choakd"/"choked" both match consistently.
"""
import unicodedata
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    """Keeps letters, combining marks, and digits from ANY script —
    not just ASCII a-z0-9. The previous version used
    re.sub(r"[^a-z0-9\\s]", " ", ...), which silently stripped every
    non-Latin character to nothing: a Hindi (or any other
    non-English-script) message normalized to an empty string, so it
    could never match anything, no matter how severe. A naive Unicode
    regex fix (\\w) isn't enough either — it drops Devanagari combining
    vowel marks (matras), which mangles words beyond recognition
    ("मुझे" -> "म झ"). Category-based filtering keeps marks (Mn/Mc) too,
    so words stay intact."""
    # NFC + chandrabindu -> anusvara so the common Hindi spelling variants
    # ("हूँ" / "हूं", precomposed vs. combining nukta) compare equal. The
    # same normalization is applied to lexicon phrases (see
    # _phrase_match_confidence), so both sides always agree.
    text = unicodedata.normalize("NFC", text).replace("ँ", "ं")
    text = text.lower().replace("'", "").replace("’", "")
    return "".join(
        ch if (unicodedata.category(ch)[0] in ("L", "M", "N") or ch.isspace()) else " "
        for ch in text
    )


def _word_matches(phrase_word: str, text_word: str) -> bool:
    if phrase_word == text_word:
        return True
    # Typo tolerance only kicks in for longer words — short words (< 5
    # chars) are excluded because fuzzy-matching short words is how you
    # get false positives like "will" matching "kill".
    if len(phrase_word) < 5 or len(text_word) < 4:
        return False
    # Typos almost never change the first letter, but different words
    # that merely contain a lexicon word do ("scraped" vs "raped",
    # "misplaced" vs "displaced") — requiring the same first letter
    # stops those from firing high-severity categories.
    if phrase_word[0] != text_word[0]:
        return False
    return SequenceMatcher(None, phrase_word, text_word).ratio() >= 0.82


# Negation/hedging context, checked in the few words immediately BEFORE
# a matched phrase. This exists because plain phrase matching cannot
# tell "I want to die" from "I do NOT want to die" — both contain the
# literal substring "want to die". A negation in the preceding window
# suppresses the match entirely (confidence 0 — it isn't evidence of
# anything). A hedge word ("might", "may") doesn't suppress it — a
# hedged disclosure is still worth flagging — but it's treated as
# lower-confidence than a direct, certain statement ("he hurts me" vs
# "I'm afraid he MAY hurt me"), which matters for how strongly it
# contributes to the combined score below.
_NEGATION_WORDS = {"not", "no", "never", "dont", "didnt", "doesnt", "wont",
                    "cant", "cannot", "wouldnt", "isnt", "wasnt", "without"}
_HEDGE_WORDS = {"may", "might", "could", "possibly", "perhaps", "maybe"}
_CONTEXT_WINDOW = 4


def _match_confidence(tokens: list, start_idx: int) -> float:
    preceding = tokens[max(0, start_idx - _CONTEXT_WINDOW):start_idx]
    if any(w in _NEGATION_WORDS for w in preceding):
        return 0.0
    if any(w in _HEDGE_WORDS for w in preceding):
        return 0.5
    return 1.0


def _phrase_match_confidence(phrase: str, tokens: list) -> float:
    """Slides a window the length of the phrase across the tokenized
    text; every word in the window must match its phrase counterpart
    (exactly, or within typo tolerance via _word_matches — this is what
    lets "he choakd me" still match "choked me"). Returns the highest
    confidence found across all matching positions (a phrase might
    match multiple times with different surrounding context), or 0.0
    if it never matches at all."""
    phrase_words = _normalize(phrase).split()
    n = len(phrase_words)
    best = 0.0
    for i in range(len(tokens) - n + 1):
        window = tokens[i:i + n]
        if all(_word_matches(pw, tw) for pw, tw in zip(phrase_words, window)):
            best = max(best, _match_confidence(tokens, i))
    return best


CATEGORIES = {
    "suicidal_ideation": {
        "weight": 1.0,
        "phrases": [
            "want to die", "dont want to live", "no point living",
            "better off dead", "end it all", "cant go on",
            "no reason to be here", "wish i was never born",
            "kill myself", "end my life", "take my own life",
            "not worth living", "planning to suicide", "going to suicide",
            "no will to live", "life has no meaning", "ending my life",
            "want to end everything", "cant take this anymore",
            "no way out", "better off without me", "done with life",
            "why should i keep living", "waiting to die",
            "hope i dont wake up", "hurting myself on purpose",
            "thoughts of ending it", "nothing left for me",
            "i have a plan to end my life", "keep thinking about death",
            "suicide", "suicidal",
            # Hindi — starting set, needs native-speaker review/expansion
            "आत्महत्या", "मरना चाहता हूं", "मरना चाहती हूं",
            "जीना नहीं चाहता", "जीना नहीं चाहती", "जीने का मन नहीं",
            "खुद को खत्म कर लूं", "मर जाना चाहती हूं", "मर जाना चाहता हूं",
            "मरना चाहता", "मरना चाहती", "खुदकुशी",
            # Hinglish (romanised Hindi)
            "marna chahta", "marna chahti", "jeena nahi chahta",
            "jeena nahi chahti", "jina nahi chahta", "khudkushi", "aatmahatya",
            "mar jaana chahta", "mar jana chahti",
            # Gujarati
            "આત્મહત્યા", "મરી જવું", "મરી જવું છે", "જીવવું નથી", "જીવવાની ઈચ્છા નથી",
            # Marathi
            "मरायचं आहे", "जगायचं नाही", "जीव द्यायचा", "मरून जावंसं वाटतं",
            # Tamil
            "தற்கொலை", "சாக வேண்டும்", "வாழ விருப்பமில்லை", "சாகப் போகிறேன்",
            # Bengali
            "আত্মহত্যা", "বাঁচতে চাই না", "মরে যেতে চাই", "মরতে চাই",
        ],
    },
    "severe_trauma_fear": {
        "weight": 0.8,
        "phrases": [
            "so scared", "terrified", "he will kill me", "they will hurt me",
            "im not safe", "cant sleep at night", "keeps happening",
            "living in fear", "afraid every day", "constant fear",
            "scared for my life", "panic when i see him", "flashbacks",
            "nightmares about it", "cant stop shaking", "frozen with fear",
            "petrified", "scared to go home", "afraid of what he will do",
            "im not safe", "i am not safe", "scaring me", "scares me",
            # Hindi
            "डर लग रहा है", "बहुत डर लगता है", "डरी हुई हूं",
            "डरा हुआ हूं", "बहुत डरती हूं", "मुझे डर है",
            "जान का खतरा है",
            "cant sleep", "cannot sleep", "every time i close my eyes",
            "keep reliving", "i keep seeing", "नींद नहीं आती",
            "traumatised", "traumatized", "since the incident",
            "in shock", "सदमे में", "sadme mein",
            # Hinglish
            "dar lag raha", "darr lag raha", "bahut dar", "dar lagta hai",
            # Gujarati / Marathi / Tamil / Bengali
            "ડર લાગે છે", "ખૂબ ડર", "भीती वाटते", "खूप भीती",
            "பயமாக இருக்கிறது", "ভয় লাগছে", "খুব ভয়",
        ],
    },
    "immediate_danger": {
        "weight": 1.0,
        "phrases": [
            # The single highest-urgency signal a text can carry: the
            # person causing harm is physically present RIGHT NOW, not a
            # past or ongoing-but-not-current situation. This gets its
            # own category (rather than folding into severe_trauma_fear)
            # because it needs to bypass the normal conversation flow
            # entirely — see chat.py — the same way suicidal ideation
            # does, instead of competing for a turn against other topics.
            "standing right behind me", "he is here right now",
            "hes here right now", "right behind me right now",
            "in the room with me right now", "watching me right now",
            "he wont let me leave right now", "he is standing behind me",
            "hes standing behind me", "he has a knife", "hurting me right now",
            "bleeding right now", "he is here", "hes here",
        ],
    },
    "intimidation_threat": {
        "weight": 0.85,
        "phrases": [
            "threatened me", "said he would", "warned me not to tell",
            "if i tell anyone", "watching me", "following me",
            "shows up at my house", "sends threatening messages",
            "threatened to hurt my family", "blackmailing me",
            "threatened to kill me", "warned me to stay quiet",
            "monitors my phone", "controls where i go",
            "wont let me leave the house", "tracks my location",
            "threatened to hurt me if i tell", "keeps showing up uninvited",
            "threatening me", "been threatening me",
            # Hindi
            "पीछे पड़ा हुआ है", "पीछा कर रहा है", "धमकी दी",
            "मुझे धमकाता है", "धमका रहा है", "पीछे लगा हुआ है",
            "जान से मारने की धमकी", "धमकी दे रहे", "मार डालेंगे", "धमकियां",
            "will kill me", "kill me if", "kill us", "will kill my family",
            # Hinglish
            "dhamki di", "dhamki de rahe", "jaan se maarne", "jaan se marne",
            "maar denge", "maar dalenge", "maar daalenge",
            # Gujarati
            "ધમકી આપી", "મારી નાખશે", "જાનથી મારી નાખવાની",
            # Marathi
            "धमकी दिली", "जीवे मारण्याची धमकी", "मारून टाकीन", "मारून टाकू",
            # Tamil
            "மிரட்டல்", "மிரட்டினார்கள்", "கொன்றுவிடுவோம்", "கொலை மிரட்டல்",
            # Bengali
            "হুমকি", "হুমকি দিয়েছে", "মেরে ফেলবে", "খুনের হুমকি",
        ],
    },
    "depression_hopelessness": {
        "weight": 0.6,
        "phrases": [
            "feel empty", "nothing matters", "so tired of everything",
            "cant feel anything", "hopeless", "worthless",
            "no motivation", "lost interest in everything",
            "cant get out of bed", "everything feels pointless",
            "numb all the time", "havent been happy in so long",
            "exhausted all the time", "want to give up", "feel broken",
            "cant see it getting better", "drowning in all this",
            # Hindi
            "बहुत उदास रहती हूं", "बहुत उदास रहता हूं",
            "कुछ अच्छा नहीं लगता", "हिम्मत नहीं बची",
            "very stressed", "so stressed", "cant eat", "cannot eat", "not eating",
            "tension", "तनाव", "tanav",
        ],
    },
    "social_isolation": {
        "weight": 0.5,
        "phrases": [
            "no one to talk to", "all alone", "no one believes me",
            "cut off from my family", "dont have anyone", "isolated",
            "no friends left", "lost contact with everyone",
            "nobody checks on me", "spend all my time alone",
            "not allowed to see my friends", "cut off from everyone",
            "no one to turn to", "everyone has abandoned me",
            # Hindi
            "कोई मेरी बात नहीं सुनता", "बिल्कुल अकेली हूं",
            "बिल्कुल अकेला हूं", "कोई मदद नहीं करता",
        ],
    },
    "physical_safety_abuse": {
        "weight": 0.85,
        "phrases": [
            "hit me", "hurt me physically", "beats me", "he hits me",
            "pushed me", "grabbed me violently", "hurts me",
            "slapped me", "choked me", "burned me", "locked me in a room",
            "threw something at me", "physically abuses me",
            "leaves bruises", "hurts me when he drinks", "kicked me",
            "twisted my arm", "held me down", "cut me", "cut my finger",
            "stabbed me", "hes hurting me right now", "attacked me with",
            "hurt me with a knife", "bleeding because of him", "hes bleeding me",
            "hitting me", "beating me", "kicking me", "punching me",
            "punched me", "slapping me", "choking me", "pushing me around",
            "throwing things at me", "hurting me",
            # Hindi
            "मुझे मारता है", "बहुत मारता है", "थप्पड़ मारा",
            "मारपीट करता है", "मुझे पीटा",
            "मारपीट की", "मारपीट", "मारा पीटा", "लाठी से मारा", "पीटा", "मारा",
            "attacked", "attacked my", "beat me", "beat us", "beat my",
            "beaten", "with sticks", "lathi",
            # Hinglish
            "maarpeet", "maar peet", "mara peeta", "pitai ki", "lathi se maara",
            "beat him", "beat her", "beat them", "beats us", "beating us",
            "tied to a tree", "tied him", "tied her", "thrashed", "tortured",
            "torture", "injured", "hamla", "hamla kiya", "हमला", "हमला किया",
            "पिटाई", "घायल", "હુમલો", "હુમલો કર્યો", "हल्ला केला",
            # Gujarati
            "માર માર્યો", "મારપીટ", "માર્યો", "ઢોર માર",
            # Marathi
            "मारहाण केली", "मारहाण", "मारलं",
            # Tamil
            "அடித்தார்கள்", "அடித்தான்", "தாக்கினார்கள்", "தாக்குதல்",
            # Bengali
            "মারধর", "মেরেছে", "পিটিয়েছে",
        ],
    },
    "sexual_harassment_assault": {
        "weight": 0.9,
        "phrases": [
            "touched me without", "forced me to", "sexually harassed",
            "assaulted me", "inappropriate touching", "against my will",
            "made unwanted advances", "sent inappropriate messages",
            "exposed himself", "tried to force himself", "non consensual",
            "made me uncomfortable sexually", "wouldnt stop touching me",
            "being harassed", "harassing me", "forcing me to be with him",
            "dont want to be with him", "wont leave me alone",
            "making me stay with him", "forcing me to stay",
            "touching me inappropriately", "keeps touching me",
            # Hindi
            "जबरदस्ती करके", "जबरदस्ती की", "जबरदस्ती करता है",
            "छेड़छाड़ की", "गलत तरीके से छुआ", "रूम में लेकर जाना",
            "फोर्स कर रहा है",
        ],
    },
    "financial_exploitation": {
        "weight": 0.4,
        "phrases": [
            "controls my money", "took my salary", "wont let me work",
            "no access to money", "controls my finances",
            "took my property", "forced me to sign", "stole my savings",
            "demanding dowry", "asking for more dowry", "took my jewelry",
            "made me take a loan", "spent all my savings",
            "wont give me my own money",
            # Hindi
            "पैसे लेने जाता है", "सारे पैसे", "पैसे छीन लेता है",
            "पैसे नहीं देता",
        ],
    },
    "legal_dispute": {
        "weight": 0.35,
        "phrases": [
            "property dispute", "land dispute", "legal problems",
            "custody battle", "filed a police complaint",
            "wont file my fir", "eviction notice", "inheritance dispute",
            "property taken illegally", "denied my rights",
            "cheated me legally", "fraud case", "cant afford a lawyer",
            "court case pending", "want to file a complaint",
            "family land", "father's land", "fathers land",
            "property papers", "legal help",
            # Hindi — starting set, needs native-speaker review/expansion
            "संपत्ति विवाद", "जमीन का विवाद", "एफआईआर नहीं लिख रहे",
            "कोर्ट केस चल रहा है", "वकील का खर्च नहीं उठा सकती",
            "पिता की जमीन", "संपत्ति के कागज", "कानूनी मदद चाहिए",
            "बेदखली का नोटिस",
        ],
    },
    "workplace_harassment": {
        "weight": 0.6,
        "phrases": [
            "harassed at work", "my boss threatened", "workplace harassment",
            "hostile work environment", "unfair termination",
            "discriminated at work", "coworker harassing me",
            "hr ignored my complaint", "harassment by my manager",
            # Hindi — starting set, needs native-speaker review/expansion
            "काम पर परेशान करता है", "ऑफिस में परेशान करते हैं",
            "बॉस ने धमकी दी", "बिना वजह नौकरी से निकाल दिया",
            "सहकर्मी परेशान करता है", "एचआर ने शिकायत नहीं सुनी",
        ],
    },
    "cybercrime_online_abuse": {
        "weight": 0.55,
        "phrases": [
            "leaked my photos", "leaked my private photos",
            "blackmailing me online", "posted my private photos",
            "posted my photos online", "cyberbullying",
            "online harassment", "fake profile of me", "hacked my account",
            "stalking me online", "threatening me online",
            "sharing my photos without permission",
            # Hindi — starting set, needs native-speaker review/expansion
            "मेरी फोटो लीक कर दी", "फोटो ऑनलाइन डाल दी",
            "ऑनलाइन ब्लैकमेल कर रहा है", "फर्जी प्रोफाइल बना दी",
            "अकाउंट हैक कर लिया", "इंटरनेट पर धमकी दे रहा है",
        ],
    },
    "child_safety_concern": {
        "weight": 0.95,
        "phrases": [
            "my child is being hurt", "worried about my childs safety",
            "child abuse", "hurting my kids", "my daughter is scared of him",
            "my son came home injured", "someone is hurting my child",
            "afraid for my childs safety",
            # Hindi — starting set, needs native-speaker review/expansion
            "मेरे बच्चे को मारता है", "बच्ची डरी हुई है",
            "बेटा चोटिल होकर घर आया", "मेरे बच्चे को कोई दुख पहुंचा रहा है",
            "बच्चे की सुरक्षा को लेकर चिंतित हूं",
        ],
    },
    "elder_abuse": {
        "weight": 0.7,
        "phrases": [
            "abusing my elderly parent", "neglecting my grandmother",
            "elder abuse", "financial abuse of my father",
            "mistreating my elderly mother", "neglecting my parents",
            # Hindi — starting set, needs native-speaker review/expansion
            "बुजुर्ग माता-पिता के साथ दुर्व्यवहार", "दादी की उपेक्षा कर रहे हैं",
            "बुजुर्ग मां के साथ बदसलूकी", "माता-पिता की उपेक्षा",
            "बूढ़े पिता का पैसा हड़प लिया",
        ],
    },
    "substance_related": {
        "weight": 0.4,
        "phrases": [
            "he drinks and gets violent", "drunk and abusive",
            "addicted and violent", "substance abuse at home",
            "violent when he drinks",
            # Hindi — starting set, needs native-speaker review/expansion
            "शराब पीकर मारपीट करता है", "नशे में हिंसक हो जाता है",
            "शराबी और हिंसक", "घर में नशे की समस्या",
        ],
    },
    # ---- Atrocity-specific categories (SIH26093 / NHAA 14566) ----------
    # The categories above were written for general distress (domestic
    # abuse, harassment, suicide). The people who reach 14566 describe
    # caste-based violence, sexual violence, killings, boycott,
    # displacement, police inaction and pressure on witnesses — none of
    # which the lexicon had words for, so those disclosures scored Low.
    # Every phrase below (especially the Indian-language ones) is a
    # starting set that still needs review by native speakers and by
    # counsellors who work with atrocity survivors.
    "caste_atrocity": {
        "weight": 0.85,
        "phrases": [
            "caste name", "caste names", "caste slur", "caste slurs",
            "casteist", "caste abuse", "caste based abuse", "abused my caste",
            "insulted my caste", "insulted our caste", "because i am dalit",
            "because i am a dalit", "because we are dalit", "because we are dalits",
            "because of my caste", "because of our caste", "caste discrimination",
            "discriminated because of caste", "untouchability", "untouchable",
            "not allowed to enter the temple", "separate utensils", "separate glass",
            "not allowed to sit", "made to remove shoes", "jaatisuchak",
            "jatisuchak", "jaati ke naam par", "jati ke naam par", "chhuachhut",
            "chuachut", "jativadi gaali",
            "urine", "excreta", "human waste", "faeces", "feces",
            "manual scavenging", "carry dead animals", "bonded labour",
            "bonded labourers", "bonded labor", "bandhua", "wont let our children",
            "wont let us enter", "wont let us take water",
            "denied entry", "बंधुआ", "मैला", "पेशाब", "मल मूत्र",
            # Hindi
            "जाति के नाम पर", "जातिसूचक", "जातिसूचक गाली", "छुआछूत",
            "जातिवादी", "जाति की गाली", "नीची जाति का",
            # Gujarati
            "જાતિના નામે", "જ્ઞાતિના નામે", "જાતિવાચક", "આભડછેટ",
            # Marathi
            "जातीवरून", "जातिवाचक", "जातीवाचक", "जातीवाचक शिवीगाळ",
            # Tamil
            "சாதி பெயரை", "சாதிப் பெயரை", "சாதி பெயரைச்", "தீண்டாமை",
            # Bengali
            "জাত তুলে", "জাতপাত", "অস্পৃশ্যতা",
        ],
    },
    "atrocity_context": {
        # Context, not harm: tells us the case falls under the SC/ST
        # (Prevention of Atrocities) Act context 14566 exists for. Small
        # weight, no floor — it nudges the score and routing, never
        # decides them on its own.
        "weight": 0.2,
        "phrases": [
            "dalit", "dalits", "upper caste", "lower caste", "caste",
            "sc", "sarpanch", "landlord", "dominant caste",
            "scheduled caste", "scheduled tribe", "sc st", "scst", "adivasi",
            "tribal", "atrocity", "atrocities", "sc st act", "poa act",
            "दलित", "आदिवासी", "अत्याचार", "ऊंची जाति", "सवर्ण",
            "જાતિ", "દલિત", "சாதி", "দলিত", "আদিবাসী",
        ],
    },
    "rape_sexual_violence": {
        "weight": 1.0,
        "phrases": [
            "raped", "rape", "gang raped", "gang rape", "gangrape",
            "gangraped", "sexually assaulted", "sexual assault",
            "forced to have sex", "forced sex", "stripped me", "stripped her",
            "paraded naked", "balatkar", "samuhik balatkar",
            "sexually abused", "sexual abuse", "stripped", "paraded",
            "galat kaam", "izzat loot", "यौन शोषण", "गलत काम", "इज्जत लूट",
            # Hindi
            "बलात्कार", "सामूहिक बलात्कार", "दुष्कर्म", "रेप", "गैंगरेप",
            "नग्न घुमाया",
            # Gujarati
            "બળાત્કાર", "સામૂહિક બળાત્કાર",
            # Tamil
            "பாலியல் வன்கொடுமை", "கற்பழிப்பு", "கூட்டு பாலியல்",
            # Bengali
            "ধর্ষণ", "গণধর্ষণ",
        ],
    },
    "family_murder": {
        "weight": 1.0,
        "phrases": [
            "murdered", "murder", "killed my", "they killed", "was killed",
            "were killed", "burnt alive", "burned alive", "lynched",
            "beaten to death", "hacked to death", "killed him", "killed her",
            "hatya kar di", "hatya", "maar dala", "maar daala",
            "ko maar diya", "ko maar dala", "को मार दिया", "जान ले ली",
            # Hindi
            "हत्या", "हत्या कर दी", "मार डाला", "जिंदा जला दिया",
            "जान से मार दिया", "पीट पीट कर मार",
            # Gujarati
            "હત્યા", "મારી નાખ્યો", "મારી નાખ્યા", "મારી નાખી",
            # Marathi
            "खून केला", "ठार मारले", "जीवे मारले",
            # Tamil
            "கொலை செய்தார்கள்", "கொன்றார்கள்", "கொலை",
            # Bengali
            "খুন করেছে", "খুন", "হত্যা",
        ],
    },
    "witness_intimidation": {
        "weight": 0.9,
        "phrases": [
            "take back the case", "take back my case", "take back my complaint",
            "withdraw the case", "withdraw my case", "withdraw my complaint",
            "withdraw the complaint", "if i testify", "not to testify",
            "stop me from testifying", "turn hostile", "change my statement",
            "out on bail", "pressuring me to withdraw", "pressure to compromise",
            "compromise the case", "compromise with them", "settle the case",
            "case wapas", "case wapas lo", "case wapas le lo", "gawahi",
            "samjhauta karo",
            # Hindi
            "केस वापस", "केस वापस लो", "शिकायत वापस", "मुकदमा वापस",
            "गवाही", "गवाही मत देना", "समझौता करने का दबाव", "समझौते का दबाव",
            "जमानत पर बाहर",
            # Gujarati
            "કેસ પાછો ખેંચ", "ફરિયાદ પાછી ખેંચ", "સમાધાન કરવા દબાણ",
            # Marathi
            "केस मागे घे", "तक्रार मागे घे", "साक्ष",
            # Tamil / Bengali
            "வழக்கை வாபஸ்", "মামলা তুলে নিতে",
        ],
    },
    "social_boycott": {
        "weight": 0.75,
        "phrases": [
            "social boycott", "boycott", "boycotted", "hukka paani band",
            "hukka pani band", "hukka pani", "not allowed to draw water",
            "not allowed to take water", "not allowed to use the well",
            "no one sells us", "shops refuse", "outcast", "excommunicated",
            "bahishkar", "samajik bahishkar",
            "no one talks to us", "nobody talks to us", "nobody in the village talks",
            "stopped talking to us",
            # Hindi
            "सामाजिक बहिष्कार", "बहिष्कार", "हुक्का पानी बंद",
            "पानी भरने नहीं देते", "सामान नहीं देते",
            # Gujarati / Marathi
            "બહિષ્કાર", "સામાજિક બહિષ્કાર", "वाळीत टाकले",
            # Tamil / Bengali
            "ஒதுக்கி வைத்தார்கள்", "சமூக புறக்கணிப்பு", "একঘরে",
            "সামাজিক বয়কট",
        ],
    },
    "displacement_arson": {
        "weight": 0.8,
        "phrases": [
            "forced to leave our village", "forced to leave the village",
            "forced to leave our home", "forced to leave our house", "driven out of the village", "thrown out of the village",
            "house was burnt", "house was burned", "burnt our house",
            "burned our house", "burnt our houses", "set fire to our",
            "set our house on fire", "demolished our house", "nowhere to live",
            "evicted", "were displaced", "got displaced", "grabbed our land", "occupied our land",
            "land grabbed", "took our land", "not letting us enter our",
            "ghar jala diya", "gaon chhodna pada",
            "tod diya", "sab tod diya", "destroyed our", "damaged our house",
            "तोड़ दिया", "तोड़फोड़",
            # Hindi
            "घर जला दिया", "गांव छोड़ना पड़ा", "गांव से निकाल दिया",
            "घर से निकाल दिया", "जमीन पर कब्जा", "बेघर हो गए", "घर तोड़ दिया",
            # Gujarati / Marathi
            "ઘર સળગાવી દીધું", "ગામ છોડવું પડ્યું", "જમીન પચાવી",
            "घर जाळले", "गाव सोडावे लागले",
            # Tamil / Bengali
            "வீட்டை எரித்தார்கள்", "ঘর পুড়িয়ে দিয়েছে", "গ্রাম ছাড়তে",
        ],
    },
    "police_noncooperation": {
        "weight": 0.6,
        "phrases": [
            "refused to register", "refused to file", "not registering my fir",
            "wont register my fir", "did not register my fir",
            "didnt register my fir", "did not file my fir", "no fir",
            "not filing chargesheet", "not filing the chargesheet",
            "no chargesheet", "police are not helping", "police did nothing",
            "police are not listening", "police sided with", "police are with them",
            "police took their side", "told me to compromise",
            "asked me to compromise", "no action taken", "no action was taken",
            "pending for years", "been pending for", "case is pending", "investigation is stuck",
            "fir nahi likh", "fir nahi likhi", "police madad nahi",
            "not arresting", "no arrest", "not arrested", "refused to take my complaint",
            "complaint not registered", "shikayat darj nahi", "शिकायत दर्ज नहीं",
            "दर्ज करने से मना", "गिरफ्तार नहीं",
            # Hindi
            "एफआईआर दर्ज नहीं", "रिपोर्ट नहीं लिखी", "एफआईआर नहीं लिखी",
            "पुलिस नहीं सुन रही", "पुलिस कोई कार्रवाई नहीं", "चार्जशीट दाखिल नहीं",
            "पुलिस उनके साथ",
            # Gujarati / Marathi
            "ફરિયાદ નોંધી નહીં", "એફઆઈઆર નોંધી નથી", "तक्रार नोंदवली नाही",
            "एफआयआर नोंदवला नाही",
        ],
    },
    "vulnerability_context": {
        # Not a harm category on its own — these flag heightened
        # vulnerability that should raise the combined score alongside
        # whatever else is disclosed, since the same abuse is more
        # dangerous for someone who is pregnant, a minor, disabled, or
        # has nowhere else to go. Contributes to the cumulative score
        # like everything else; never floors or overrides by itself.
        "weight": 0.3,
        "phrases": [
            "im pregnant", "i am pregnant", "im disabled",
            "i have a disability", "im homeless", "no shelter",
            "nowhere else to go", "i am a minor", "im underage",
            "im elderly", "i am elderly", "dont speak the language",
            "no one else speaks my language",
            # Hindi — starting set, needs native-speaker review/expansion
            "मैं गर्भवती हूं", "मुझे दिव्यांगता है", "रहने की जगह नहीं है",
            "जाने के लिए कहीं नहीं है", "मैं नाबालिग हूं", "मैं बुजुर्ग हूं",
            "भाषा नहीं आती",
        ],
    },
}


def keyword_severity_score(text: str) -> dict:
    """Returns per-category hits, per-category confidence, and an
    aggregate 0-100 severity score.

    The aggregate is a capped CUMULATIVE combination across every
    category that was hit, not just the single highest-weight one —
    "K = 100 x (1 - product(1 - weight_i x confidence_i))" for every
    hit category i. This matters because someone disclosing physical
    abuse AND threats AND isolation is describing a more dangerous
    situation than someone disclosing only one of those, and the old
    max-only approach scored them almost identically. This combination
    (sometimes called "noisy-OR") is a standard way to combine several
    independent risk signals: each additional category pushes the
    score further toward 100 without ever exceeding it, and a single
    very severe category still dominates when nothing else is present.

    confidence_i comes from _phrase_match_confidence: 1.0 for a direct
    match, 0.5 for a hedged one ("he MIGHT hurt me"), and negated
    matches never make it into hits at all (confidence 0 = no match)."""
    text_l = _normalize(text)
    tokens = text_l.split()
    hits = {}
    confidences = {}

    for cat, spec in CATEGORIES.items():
        matched = []
        best_conf = 0.0
        for p in spec["phrases"]:
            conf = _phrase_match_confidence(p, tokens)
            if conf > 0:
                matched.append(p if conf == 1.0 else f"{p} (uncertain)")
                best_conf = max(best_conf, conf)
        if matched:
            hits[cat] = matched
            confidences[cat] = best_conf

    survival = 1.0
    for cat, conf in confidences.items():
        survival *= (1 - CATEGORIES[cat]["weight"] * conf)
    score = round(100 * (1 - survival), 1)

    return {"hits": hits, "score": score, "confidences": confidences}

# Sahayk — test matrix

For each line: run it through the chat (or `/analyze/text`), then check the
officer dashboard for what actually got flagged. "Expected" is a rough
sanity check, not the "correct" answer — use your own judgment on borderline
cases. The real value here is the **mismatches** — note anywhere the actual
result surprises you.

Each category has 4 variants: mild/ambiguous, clear/direct, a typo or
unusual phrasing, and one combined with a second category.

## 1. Suicidal ideation
- "I don't know why I even bother anymore" — *mild, ambiguous, judgment call*
- "I want to end my life, I can't do this anymore" — *expect Critical*
- "i jst wana end evrythng" — *typo-heavy, expect Critical if working*
- "he keeps threatening me and honestly I don't want to live like this" — *expect Critical + intimidation flagged too*

## 2. Severe trauma / fear
- "I've been a bit anxious lately" — *mild*
- "I'm terrified every night, I can't sleep, he could show up anytime" — *expect High*
- "so so scaredd rn cant even think str8" — *typo/slang heavy*
- "I'm terrified of him and he's also stopped giving me any money" — *trauma + financial*

## 3. Intimidation / threats
- "he gets annoyed with me sometimes" — *mild, probably shouldn't flag*
- "he threatened to kill me if I ever told anyone" — *expect Critical-adjacent*
- "he sed hell hurt my family if i tel" — *typo-heavy*
- "he threatens me and watches everything I do online" — *intimidation + cybercrime*

## 4. Depression / hopelessness
- "just feeling a bit low today" — *mild*
- "I feel completely empty, nothing matters anymore, I've given up" — *expect High*
- "so numb lately idk whats wrong w me" — *casual/typo phrasing*
- "I feel hopeless and I have no one to talk to about it" — *depression + isolation*

## 5. Social isolation
- "haven't seen my friends in a while, been busy" — *mild, probably normal*
- "I have no one left, everyone has cut me off, I'm completely alone" — *expect Moderate/High*
- "no1 rlly checks on me anymoree" — *typo/slang*
- "I'm all alone and he won't let me leave the house" — *isolation + physical/control*

## 6. Physical abuse
- "we argue sometimes like any couple" — *mild, ambiguous — should NOT auto-flag as abuse*
- "he hit me again last night, there are bruises" — *expect High/Critical*
- "he choakd me wen he was drunk" — *typo-heavy*
- "he hit me and took away my phone and money" — *physical + financial*

## 7. Sexual harassment / assault
- "a coworker made a comment that felt a bit off" — *mild, ambiguous*
- "he touched me without my consent and I froze" — *expect High/Critical*
- "he tuchd me n i didnt want tht" — *typo-heavy*
- "he assaulted me and is now threatening me not to tell anyone" — *assault + intimidation*

## 8. Financial exploitation
- "money's tight right now" — *mild, probably normal, should NOT flag*
- "he controls all my money and won't let me work" — *expect Moderate*
- "he tuk al my savings n wont giv it bak" — *typo-heavy*
- "he controls my money and gets violent if I ask questions" — *financial + physical*

## 9. Legal disputes
- "just dealing with some paperwork" — *mild, should NOT flag*
- "there's a property dispute over my father's land and I can't afford a lawyer" — *expect Legal aid tag*
- "cant fyl my FIR the police wont tak it" — *typo-heavy*
- "there's a property dispute and my uncle has been threatening me over it" — *legal + intimidation*

## 10. Workplace harassment
- "work's been stressful this month" — *mild, normal*
- "my manager has been harassing me and HR ignored my complaint" — *expect Legal aid tag*
- "my bos keeps makin me uncomfortble at wrk" — *typo-heavy*
- "my manager harasses me and threatened to fire me if I report it" — *workplace + intimidation*

## 11. Cybercrime / online abuse
- "someone commented something rude on my post" — *mild, should NOT flag*
- "he leaked my private photos online and is blackmailing me" — *expect Legal aid / cybercrime tag*
- "he hackd my acount n is postin stuff" — *typo-heavy*
- "he leaked my photos and now shows up outside my house" — *cybercrime + trauma/fear*

## 12. Child safety
- "my kid had a rough day at school" — *mild, normal*
- "I'm scared someone at home is hurting my child" — *expect Critical / immediate review*
- "my son came home injrd again n wont say how" — *typo-heavy*
- "I think someone is hurting my child and I don't know who to tell" — *child safety + isolation*

## 13. Elder abuse
- "my mom's been a bit forgetful lately" — *mild, normal, health-related not abuse*
- "my father is being financially and physically abused by my brother" — *expect High + legal tag*
- "someone is neglectin my grandma n i dont no wat 2 do" — *typo-heavy*
- "my elderly mother is being abused and threatened not to tell anyone" — *elder abuse + intimidation*

## 14. Substance-related
- "he drinks on weekends" — *mild, normal, should NOT flag*
- "he's drunk and violent almost every night now" — *expect High*
- "hes always drnk n hits me wen hes lyk tht" — *typo-heavy*
- "he's violent when he drinks and controls all our money" — *substance + physical + financial*

## False-positive / control cases (should NOT be flagged high)
These test whether the system over-triggers on words used in an unrelated,
hypothetical, or third-person sense — a real weakness of keyword matching.
- "I watched a documentary about suicide prevention today, it was really informative"
- "My psychology class covered depression and trauma this week"
- "That movie was so depressing, I cried the whole time"
- "I used to feel hopeless in my old job but I'm in a much better place now"
- "My friend went through domestic abuse a few years ago and I want to know how to support her"
- "Just checking what kind of services this helpline offers"

## Multi-signal stress test
- A long, rambling message mixing 3+ categories in one go, to see whether the
  priority ordering (suicide > child safety > physical/sexual > threats > ...)
  picks the right thing to ask about first.
- The same message split across 3 separate chat turns instead of one — to check
  whether cumulative-transcript scoring behaves the same either way.

## Voice-specific (needs the mic, not just text)
- Say something calm and neutral in a flat, steady voice.
- Say the same words but rushed, breathless, with a shaky/uneven pitch.
- Say something with a long pause (5+ seconds) in the middle before continuing.
- Compare the `speech_stress` breakdown values across all three on the dashboard.

## Multilingual
- Repeat 3-4 of the above in Hindi (or another language from the dropdown)
  using the mic. Note: transcription will work via the browser, but scoring
  is still English-only under the hood — expect the SVI to under-react to
  non-English text regardless of severity. This is a known, already-flagged
  gap (see README "Next build steps").

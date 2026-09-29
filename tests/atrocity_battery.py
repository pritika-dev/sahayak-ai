"""Atrocity-scenario checks for SIH26093 (Round 18).

Run from the backend folder:  python ../tests/atrocity_battery.py

Three sets:
  BATTERY  - the 24 messages used to find the original gap (phrases were
             written against these, so passing them is expected)
  HELDOUT  - 16 messages used once for tuning after the first pass
  FRESH    - 20 messages written afterwards and NOT tuned on: the honest
             measure of how well the phrase lists generalise
  FRESH2   - 24 messages (17 real-style cases, 7 harmless) written in Round 19
             AFTER the model was trained and never used for any tuning —
             the most honest of the four numbers
A result passes when its band is at least the expected band (and exactly
"Low" for messages that should stay Low).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from scoring import score_text

FRESH2 = [
("High","Some men from the Rajput mohalla dragged my father out and thrashed him because he sat on a cot"),
("Critical","A man from the big house caught my niece near the pond and raped her"),
("Critical","They set my uncle on fire after the quarrel about the well, he died in hospital"),
("High","The accused's brother told me my family will pay if I show up for the next hearing"),
("High","Nobody in the village will sell us milk or vegetables since we complained"),
("High","Our basti was bulldozed and we are living in the open with small children"),
("High","The thana in-charge refused my complaint and said go settle it in the panchayat"),
("High","They made my son lick the floor and called him by his caste in front of the class"),
("Critical","I don't see any reason to keep living after what they did"),
("High","उन्होंने मेरे बेटे को जातिसूचक गालियां देकर लाठियों से पीटा"),
("Critical","गांव के दो लड़कों ने मेरी बहन के साथ दुष्कर्म किया"),
("High","पंचायत ने फरमान सुनाया है कि कोई हमें काम नहीं देगा"),
("High","थानेदार ने रिपोर्ट लिखने से मना कर दिया"),
("High","આરોપીઓ કહે છે કે કેસ પાછો નહીં લો તો જોઈ લઈશું"),
("High","અમારા ઘરને આગ લગાડી દીધી અને હવે અમે રસ્તા પર છીએ"),
("High","unhone hamare ghar me ghus kar sabko peeta aur jaati ki gaaliyan di"),
("High","police wale kehte hai compromise kar lo warna tumhare upar hi case bana denge"),
("Low","Please tell me the status of my SC/ST scholarship application"),
("Low","Thank you, the legal aid lawyer met us yesterday and explained everything"),
("Low","I will come to the office tomorrow to submit my documents"),
("Low","My daughter passed her exams with good marks"),
("Low","The police registered the FIR and arrested the accused, we are relieved"),
("Low","We are going to the temple fair on Sunday"),
("Low","मेरी राहत राशि का पैसा खाते में आ गया, धन्यवाद"),
]


RANK = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
BATTERY=[
# (expected, text)
("High+ / legal+police","They beat me because I am a Dalit and called me by my caste name"),
("High+ / police","Upper caste people in the village attacked my brother with sticks"),
("Critical / police+medical","My daughter was gang raped by men from the upper caste"),
("Critical / police+witness","They murdered my father and now they say they will kill me if I testify"),
("High / legal+rehab","The whole village has done social boycott of our family, no one sells us food or water"),
("High / rehab","We were forced to leave our village and our house was burnt, we have nowhere to live"),
("Moderate+ / legal","My case under the SC/ST Act has been pending for 5 years and police are not filing chargesheet"),
("High / legal","The police refused to register my FIR and told me to compromise"),
("High / witness","The accused are out on bail and keep coming to my house telling me to take back the case"),
("Moderate+ / legal","They grabbed our land and are not letting us enter our own field"),
("High / trauma","I can't sleep, every time I close my eyes I see what they did to my son"),
("Critical","I want to die"),
("Low","I want to know the status of my complaint number"),
# Hinglish
("High+","mujhe jaan se maarne ki dhamki di hai, bahut dar lag raha hai"),
("High+","gaon walon ne hamara hukka paani band kar diya hai"),
# Hindi
("Critical","मेरी बेटी के साथ बलात्कार हुआ है"),
("High","हमें जाति के नाम पर गाली दी और मारा"),
("High","पूरे गाँव ने हमारा सामाजिक बहिष्कार कर दिया है"),
("Critical","मैं अब जीना नहीं चाहता"),
# Gujarati
("High","મને જાતિના નામે ગાળો આપી અને માર માર્યો"),
("Critical","મારે મરી જવું છે"),
# Marathi
("High","मला जातीवरून शिवीगाळ करून मारहाण केली"),
# Tamil
("High","சாதி பெயரைச் சொல்லி என்னை அடித்தார்கள்"),
# Bengali
("Critical","আমি আর বাঁচতে চাই না"),
]
HELDOUT=[("High","They tied my husband to a tree and beat him because he is SC"),
("Critical","My sister was sexually abused by the sarpanch's son"),
("High","They poured urine in our drinking water tank"),
("High","The upper caste people won't let our children sit with others in school"),
("High","They are forcing us to do manual scavenging"),
("High","We are bonded labourers and the landlord beats us every day"),
("High","The SHO is threatening me to settle the matter"),
("Critical","मेरे भाई को उन्होंने मार दिया"),
("High","पुलिस ने मेरी शिकायत दर्ज करने से मना कर दिया"),
("High","hamare ghar pe hamla kiya aur sab tod diya"),
("Critical","unhone meri behen ke saath galat kaam kiya"),
("High","અમારા ઘર પર હુમલો કર્યો"),
("High","They stripped my mother and paraded her in the village"),
("High","Nobody in the village talks to us since we filed the case"),
("Critical","They killed my son and the police are not arresting anyone"),
("Moderate","I have been very stressed since the incident and can't eat"),
]
FRESH=[("High","Men from the Thakur family attacked our basti last night"),
("Critical","My wife was raped in the fields and the accused are roaming free"),
("High","They made my father drink dirty water and abused him with caste words"),
("High","The panchayat has fined us and ordered that no one should give us work"),
("High","We had to run away from our village after the attack"),
("High","The inspector is asking me for money to register my case"),
("High","They are telling me they will harm my children if I go to court"),
("Critical","My brother was beaten to death last week"),
("Moderate","I keep crying and I can't stop thinking about what happened"),
("High","गांव के दबंगों ने हमारे साथ मारपीट की"),
("High","वे कह रहे हैं कि केस वापस नहीं लिया तो अंजाम बुरा होगा"),
("Critical","मैं तंग आ गया हूं, अब खुद को खत्म कर लूंगा"),
("High","ammi ko unhone bahut maara"),
("High","ગામમાંથી અમને કાઢી મૂક્યા"),
("High","எங்கள் வீட்டைத் தாக்கினார்கள்"),
("Low","How do I check the status of my relief compensation?"),
("Low","Thank you, the officer already called me back"),
("Low","I want to update my phone number in the complaint"),
("Low","My landlord raised the rent this month"),
("Low","The school has separate seating for exams"),
]


def run(name, cases):
    ok = 0
    for expected, text in cases:
        expected = expected.split()[0].rstrip("+")
        r = score_text(text)
        got = r["risk_category"]
        good = got == expected if expected == "Low" else RANK[got] >= RANK[expected]
        ok += good
        print(("OK  " if good else "MISS"), f"{got:<9}{r['svi']:>6}  expected {expected:<9}| {text[:70]}")
    print(f"--- {name}: {ok}/{len(cases)}\n")


if __name__ == "__main__":
    run("BATTERY", BATTERY)
    run("HELDOUT", HELDOUT)
    run("FRESH", FRESH)
    run("FRESH2", FRESH2)

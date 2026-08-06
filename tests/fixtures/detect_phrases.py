"""Labelled short phrases for scoring language detection.

Short is the whole point: full sentences were never the problem. What broke
was the two-word case — a greeting or a birthday wish with no function word
in it at all — which the old stopword scorer could only answer by falling
back to the pair's first language.

PHRASES is the realistic workload. TRAPS is the adversarial half: sentences
loaded with words that are function words in the *other* candidate language,
so a detector that leans too hard on the word lists gets caught.
"""

PHRASES = [
    # --- English with no function words: the class that used to always fail
    ("en", "Happy birthday!"),
    ("en", "Merry Christmas!"),
    ("en", "Congratulations!"),
    ("en", "Hello there"),
    ("en", "Safe travels"),
    ("en", "Sounds great"),
    ("en", "Nice job"),
    ("en", "Welcome home"),
    ("en", "Thank you so much"),
    ("en", "See you tomorrow"),
    ("en", "Talk soon"),
    ("en", "Sorry, running late"),
    ("en", "Almost there"),
    ("en", "Feel better soon"),
    ("en", "Good luck today"),
    ("en", "Take care"),
    ("en", "Well done"),
    ("en", "Long time no see"),
    ("en", "Sleep well"),
    ("en", "Cheers"),
    # --- English sentences
    ("en", "Where are you going tomorrow?"),
    ("en", "I love pizza"),
    ("en", "Can we meet at the station at six?"),
    ("en", "The train is delayed again"),
    ("en", "I forgot my keys at home"),
    ("en", "Do you want to grab dinner later?"),
    ("en", "She said the package already arrived"),
    ("en", "My flight lands around nine in the evening"),
    ("en", "That restaurant was surprisingly expensive"),
    ("en", "Let me know when you get there"),
    ("en", "I finished the report this morning"),
    ("en", "It's raining pretty hard outside"),
    ("en", "We should book the tickets soon"),
    ("en", "How was your weekend?"),
    ("en", "Please send me the address"),
    # --- English one-liners that collide with German/Spanish function words
    ("en", "No worries"),
    ("en", "Same here"),
    ("en", "No idea"),
    ("en", "Not sure"),
    # --- German with umlauts / ß
    ("de", "Alles Gute zum Geburtstag!"),
    ("de", "Wie geht's dir heute?"),
    ("de", "Ich möchte einen Kaffee, bitte"),
    ("de", "Die Straße ist gesperrt"),
    ("de", "Können wir uns morgen treffen?"),
    ("de", "Das war wirklich schön"),
    # --- German typed without umlauts, as a phone keyboard often produces
    ("de", "Guten Morgen"),
    ("de", "Schlaf gut"),
    ("de", "Bis morgen"),
    ("de", "Herzlichen Glueckwunsch"),
    ("de", "Frohe Weihnachten"),
    ("de", "Gute Besserung"),
    ("de", "Viel Erfolg heute"),
    ("de", "Vielen Dank"),
    ("de", "Kein Problem"),
    ("de", "Ich bin gleich da"),
    ("de", "Der Zug hat schon wieder Verspatung"),
    ("de", "Wir sollten die Tickets bald buchen"),
    ("de", "Ich habe meine Schluessel zu Hause vergessen"),
    ("de", "Das Paket ist gestern angekommen"),
    ("de", "Sie hat gesagt, dass sie spater kommt"),
    ("de", "Wollen wir heute Abend essen gehen?"),
    ("de", "Mein Flug landet gegen neun Uhr abends"),
    ("de", "Ich habe den Bericht heute Morgen fertig gemacht"),
    ("de", "Wie war dein Wochenende?"),
    ("de", "Schick mir bitte die Adresse"),
    ("de", "Es regnet ziemlich stark draussen"),
    ("de", "Das Restaurant war ueberraschend teuer"),
    ("de", "Sag mir Bescheid, wenn du da bist"),
    ("de", "Ich freue mich darauf"),
    ("de", "Machs gut"),
    # --- German conversational filler: two words, one of them a particle
    ("de", "Und sonst?"),
    ("de", "Und du?"),
    ("de", "Und dir?"),
    ("de", "Kein Ding"),
    ("de", "Na klar"),
    ("de", "Danke dir"),
    # --- Spanish, accented and (as typed in a hurry) unaccented
    ("es", "¡Feliz cumpleaños!"),
    ("es", "¿Dónde estás?"),
    ("es", "Buenos días"),
    ("es", "Muchas gracias"),
    ("es", "Nos vemos mañana"),
    ("es", "Feliz Navidad"),
    ("es", "Buen viaje"),
    ("es", "Que te mejores pronto"),
    ("es", "Mucha suerte hoy"),
    ("es", "Bien hecho"),
    ("es", "Que duermas bien"),
    ("es", "Hasta pronto"),
    ("es", "El tren llega tarde otra vez"),
    ("es", "Olvide mis llaves en casa"),
    ("es", "Quieres cenar mas tarde?"),
    ("es", "Deberiamos reservar los boletos pronto"),
    ("es", "Como estuvo tu fin de semana?"),
    ("es", "Mandame la direccion por favor"),
    ("es", "Mi vuelo aterriza cerca de las nueve de la noche"),
    ("es", "Termine el informe esta manana"),
    ("es", "Ese restaurante era sorprendentemente caro"),
    ("es", "Esta lloviendo bastante fuerte afuera"),
    ("es", "Avisame cuando llegues"),
    ("es", "Sin problema"),
    ("es", "Ya casi llego"),
]

TRAPS = [
    # English carrying German function words (war, an, in, die, hat, man,
    # will, so, am, bin, bis)
    ("en", "The war was over in nineteen forty five"),
    ("en", "I will die if I have to wait an hour"),
    ("en", "She wore a red hat in the rain"),
    ("en", "The man will be here at nine"),
    ("en", "I am in the bar with an old friend"),
    ("en", "So that is the plan"),
    ("en", "Put it in the bin"),
    ("en", "Ist that so"),
    # English carrying Spanish function words (no, a, he, me, son, la, van,
    # sin, ya, es, tan)
    ("en", "No, he gave me a van"),
    ("en", "My son has no time"),
    ("en", "He sang a note in the key of la"),
    ("en", "A tan van, no less"),
    ("en", "He told me a lie, a sin"),
    # German carrying English function words
    ("de", "Der Mann war so nett"),
    ("de", "In der Stadt war es warm"),
    ("de", "Ich hab den Hut in die Hand genommen"),
    ("de", "Man kann hier gut essen"),
    ("de", "Wir waren an dem Tag in Berlin"),
    # Spanish carrying English/German function words
    ("es", "No me gusta la sopa"),
    ("es", "El van a la casa"),
    ("es", "Ya no tengo mas tiempo"),
    ("es", "Son las nueve de la noche"),
    ("es", "Es un dia muy bonito"),
]

# Every auto-* mode the UI offers.
PAIRS = [("de", "en"), ("es", "en"), ("es", "de")]


def cases(data):
    """(pair, expected language, text) for each phrase and each pair that
    actually contains its language — the only pairs it could be asked in."""
    return [(pair, lang, text) for pair in PAIRS
            for lang, text in data if lang in pair]

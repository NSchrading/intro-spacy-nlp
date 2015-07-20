

    # Set up spaCy
    from spacy.en import English
    parser = English()
    
    # Test Data
    multiSentence = "There is an art, it says, or rather, a knack to flying." \
                     "The knack lies in learning how to throw yourself at the ground and miss." \
                     "In the beginning the Universe was created. This has made a lot of people "\
                     "very angry and been widely regarded as a bad move."

# spaCy does tokenization, sentence recognition, part of speech tagging, lemmatization, dependency parsing, and named entity recognition all at once!


    # all you have to do to parse text is this:
    #note: the first time you run spaCy in a file it takes a little while to load up its modules
    parsedData = parser(multiSentence)


    # Let's look at the tokens
    # All you have to do is iterate through the parsedData
    # Each token is an object with lots of different properties
    # A property with an underscore at the end returns the string representation
    # while a property without the underscore returns an index (int) into spaCy's vocabulary
    # The probability estimate is based on counts from a 3 billion word corpus, smoothed using the Simple Good-Turing method.
    for i, token in enumerate(parsedData):
        print("original:", token.orth, token.orth_)
        print("lowercased:", token.lower, token.lower_)
        print("lemma:", token.lemma, token.lemma_)
        print("shape:", token.shape, token.shape_)
        print("prefix:", token.prefix, token.prefix_)
        print("suffix:", token.suffix, token.suffix_)
        print("log probability:", token.prob)
        print("Brown cluster id:", token.cluster)
        print("----------------------------------------")
        if i > 10:
            break

    original: 300 There
    lowercased: 144 there
    lemma: 300 There
    shape: 187 Xxxxx
    prefix: 32 T
    suffix: 66 ere
    log probability: -7.663576126098633
    Brown cluster id: 1918
    ----------------------------------------
    original: 29 is
    lowercased: 29 is
    lemma: 52 be
    shape: 7 xx
    prefix: 14 i
    suffix: 29 is
    log probability: -5.002371311187744
    Brown cluster id: 762
    ----------------------------------------
    original: 59 an
    lowercased: 59 an
    lemma: 59 an
    shape: 7 xx
    prefix: 11 a
    suffix: 59 an
    log probability: -5.829381465911865
    Brown cluster id: 3
    ----------------------------------------
    original: 334 art
    lowercased: 334 art
    lemma: 334 art
    shape: 3 xxx
    prefix: 11 a
    suffix: 334 art
    log probability: -9.482678413391113
    Brown cluster id: 633
    ----------------------------------------
    original: 1 ,
    lowercased: 1 ,
    lemma: 1 ,
    shape: 1 ,
    prefix: 1 ,
    suffix: 1 ,
    log probability: -3.0368354320526123
    Brown cluster id: 4
    ----------------------------------------
    original: 44 it
    lowercased: 44 it
    lemma: 906264 -PRON-
    shape: 7 xx
    prefix: 14 i
    suffix: 44 it
    log probability: -5.498129367828369
    Brown cluster id: 474
    ----------------------------------------
    original: 274 says
    lowercased: 274 says
    lemma: 253 say
    shape: 20 xxxx
    prefix: 27 s
    suffix: 275 ays
    log probability: -7.604108810424805
    Brown cluster id: 244
    ----------------------------------------
    original: 1 ,
    lowercased: 1 ,
    lemma: 1 ,
    shape: 1 ,
    prefix: 1 ,
    suffix: 1 ,
    log probability: -3.0368354320526123
    Brown cluster id: 4
    ----------------------------------------
    original: 79 or
    lowercased: 79 or
    lemma: 79 or
    shape: 7 xx
    prefix: 8 o
    suffix: 79 or
    log probability: -6.262600898742676
    Brown cluster id: 404
    ----------------------------------------
    original: 1400 rather
    lowercased: 1400 rather
    lemma: 1400 rather
    shape: 20 xxxx
    prefix: 357 r
    suffix: 131 her
    log probability: -9.074186325073242
    Brown cluster id: 6698
    ----------------------------------------
    original: 1 ,
    lowercased: 1 ,
    lemma: 1 ,
    shape: 1 ,
    prefix: 1 ,
    suffix: 1 ,
    log probability: -3.0368354320526123
    Brown cluster id: 4
    ----------------------------------------
    original: 11 a
    lowercased: 11 a
    lemma: 11 a
    shape: 12 x
    prefix: 11 a
    suffix: 11 a
    log probability: -4.003841400146484
    Brown cluster id: 19
    ----------------------------------------



    # Let's look at the sentences
    sents = []
    # the "sents" property returns spans
    # spans have indices into the original string
    # where each index value represents a token
    for span in parsedData.sents:
        # go from the start to the end of each span, returning each token in the sentence
        # combine each token using join()
        sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
        sents.append(sent)
    
    for sentence in sents:
        print(sentence)

    There is an art, it says, or rather, a knack to flying.
    The knack lies in learning how to throw yourself at the ground and miss.
    In the beginning the Universe was created.
    This has made a lot of people very angry and been widely regarded as a bad move.



    # Let's look at the part of speech tags of the first sentence
    for span in parsedData.sents:
        sent = [parsedData[i] for i in range(span.start, span.end)]
        break
    
    for token in sent:
        print(token.orth_, token.pos_)

    There DET
    is VERB
    an DET
    art NOUN
    , PUNCT
    it PRON
    says VERB
    , PUNCT
    or CONJ
    rather ADV
    , PUNCT
    a DET
    knack NOUN
    to ADP
    flying NOUN
    . PUNCT



    # Let's look at the dependencies of this example:
    example = "The boy with the spotted dog quickly ran after the firetruck."
    parsedEx = parser(example)
    # shown as: original token, dependency tag, head word, left dependents, right dependents
    for token in parsedEx:
        print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])

    The det boy [] []
    boy nsubj ran ['The'] ['with']
    with prep boy [] ['dog']
    the det dog [] []
    spotted amod dog [] []
    dog pobj with ['the', 'spotted'] []
    quickly advmod ran [] []
    ran ROOT ran ['boy', 'quickly'] ['after', '.']
    after prep ran [] ['firetruck']
    the det firetruck [] []
    firetruck pobj after ['the'] []
    . punct ran [] []



    # Let's look at the named entities of this example:
    example = "Apple's stocks dropped dramatically after the death of Steve Jobs in October."
    parsedEx = parser(example)
    for token in parsedEx:
        print(token.orth_, token.ent_type_ if token.ent_type_ != "" else "(not an entity)")
    
    print("-------------- entities only ---------------")
    # if you just want the entities and nothing else, you can do access the parsed examples "ents" property like this:
    ents = list(parsedEx.ents)
    for entity in ents:
        print(entity.label, entity.label_, ' '.join(t.orth_ for t in entity))

    Apple ORG
    's (not an entity)
    stocks (not an entity)
    dropped (not an entity)
    dramatically (not an entity)
    after (not an entity)
    the (not an entity)
    death (not an entity)
    of (not an entity)
    Steve PERSON
    Jobs (not an entity)
    in (not an entity)
    October DATE
    . (not an entity)
    -------------- entities only ---------------
    274530 ORG Apple
    112504 PERSON Steve Jobs
    71288 DATE October


# spaCy is trained to attempt to handle messy data, including emoticons and other web-based features


    messyData = "lol that is rly funny :) This is gr8 i rate it 8/8!!!"
    parsedData = parser(messyData)
    for token in parsedData:
        print(token.orth_, token.pos_, token.lemma_)
        
    # it does pretty well! Note that it does fail on the token "gr8", taking it as a verb rather than an adjective meaning "great"
    # and "lol" probably isn't a noun...it's more like an interjection

    lol NOUN lol
    that DET that
    is VERB be
    rly ADV rly
    funny ADJ funny
    :) PUNCT :)
    This DET This
    is VERB be
    gr8 VERB gr8
    i PRON i
    rate VERB rate
    it PRON -PRON-
    8/8 NUM 8/8
    ! PUNCT !
    ! PUNCT !
    ! PUNCT !


# spaCy has word vector representations built in!


    from numpy import dot
    from numpy.linalg import norm
    
    # you can access known words from the parser's vocabulary
    nasa = parser.vocab['NASA']
    
    # cosine similarity
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    
    # gather all known words, take only the lowercased versions
    allWords = list({w for w in parser.vocab if w.has_repvec and w.orth_.islower() and w.lower_ != "nasa"})
    
    # sort by similarity to NASA
    allWords.sort(key=lambda w: cosine(w.repvec, nasa.repvec))
    allWords.reverse()
    print("Top 20 most similar words to NASA:")
    for word in allWords[:20]:   
        print(word.orth_)
        
    # Let's see if it can figure out this analogy
    # Man is to King as Woman is to ??
    king = parser.vocab['king']
    man = parser.vocab['man']
    woman = parser.vocab['woman']
    
    result = king.repvec - man.repvec + woman.repvec
    
    # gather all known words, take only the lowercased versions
    allWords = list({w for w in parser.vocab if w.has_repvec and w.orth_.islower() and w.lower_ != "king" and w.lower_ != "man" and w.lower_ != "woman"})
    # sort by similarity to the result
    allWords.sort(key=lambda w: cosine(w.repvec, result))
    allWords.reverse()
    print("\n----------------------------\nTop 3 closest results for king - man + woman:")
    for word in allWords[:3]:   
        print(word.orth_)
        
    # it got it! Queen!

    Top 20 most similar words to NASA:
    jpl
    noaa
    esa
    cern
    nih
    norad
    fema
    isro
    usaid
    nsf
    nsa
    dod
    usda
    caltech
    defra
    raytheon
    cia
    unhcr
    fermilab
    cdc
    
    ----------------------------
    Top 3 closest results for king - man + woman:
    queen
    monarch
    princess


# You can do cool things like extract Subject, Verb, Object triples from the dependency parse if you use my code in subject_object_extraction.py. Note: Doesn't work on complicated sentences. Fails if the dependency parse is incorrect.


    from subject_object_extraction import findSVOs
    
    # can still work even without punctuation
    parse = parser("he and his brother shot me and my sister")
    print(findSVOs(parse))
    
    # very complex sample. Only some are correct. Some are missed.
    parse = parser("Far out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the Galaxy lies a small unregarded yellow sun. "
                    "Orbiting this at a distance of roughly ninety-two million miles is an utterly insignificant little blue green planet whose ape-descended "
                    "life forms are so amazingly primitive that they still think digital watches are a pretty neat idea. "
                    "This planet has – or rather had – a problem, which was this: most of the people living on it were unhappy for pretty much of the time. "
                    "Many solutions were suggested for this problem, but most of these were largely concerned with the movements of small green pieces of paper, "
                    "which is odd because on the whole it wasn’t the small green pieces of paper that were unhappy. And so the problem remained; lots of the "
                    "people were mean, and most of them were miserable, even the ones with digital watches.")
    print(findSVOs(parse))

    [('he', 'shot', 'me'), ('he', 'shot', 'sister'), ('brother', 'shot', 'me'), ('brother', 'shot', 'sister')]
    [('orbiting', 'is', 'planet'), ('watches', 'are', 'idea'), ('problem', 'was', 'this'), ('it', 'wasn’t', 'pieces'), ('most', 'were', 'ones')]


# If you want to include spaCy in your machine learning it is not too difficult


    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    from sklearn.metrics import accuracy_score
    from nltk.corpus import stopwords
    import string
    import re
    
    # A custom stoplist
    STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
    # List of symbols we don't care about
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]
    
    # Every step in a pipeline needs to be a "transformer". Define a custom transformer to clean text using spaCy
    class CleanTextTransformer(TransformerMixin):
        """
        Convert text to cleaned text
        """
    
        def transform(self, X, **transform_params):
            return [cleanText(text) for text in X]
    
        def fit(self, X, y=None, **fit_params):
            return self
    
        def get_params(self, deep=True):
            return {}
        
    # A custom function to clean the text before sending it into the vectorizer
    def cleanText(text):
        # get rid of newlines
        text = text.strip().replace("\n", " ").replace("\r", " ")
        
        # replace twitter @mentions
        mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
        text = mentionFinder.sub("@MENTION", text)
        
        # replace HTML symbols
        text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
        
        # lowercase
        text = text.lower()
    
        return text
    
    # A custom function to tokenize the text using spaCy
    # and convert to lemmas
    def tokenizeText(sample):
    
        # get the tokens using spaCy
        tokens = parser(sample)
    
        # lemmatize
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas
    
        # stoplist the tokens
        tokens = [tok for tok in tokens if tok not in STOPLIST]
    
        # stoplist symbols
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
    
        # remove large strings of whitespace
        while "" in tokens:
            tokens.remove("")
        while " " in tokens:
            tokens.remove(" ")
        while "\n" in tokens:
            tokens.remove("\n")
        while "\n\n" in tokens:
            tokens.remove("\n\n")
    
        return tokens
    
    def printNMostInformative(vectorizer, clf, N):
        """Prints features with the highest coefficient values, per class"""
        feature_names = vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        topClass1 = coefs_with_fns[:N]
        topClass2 = coefs_with_fns[:-(N + 1):-1]
        print("Class 1 best: ")
        for feat in topClass1:
            print(feat)
        print("Class 2 best: ")
        for feat in topClass2:
            print(feat)
    
    # the vectorizer and classifer to use
    # note that I changed the tokenizer in CountVectorizer to use a custom function using spaCy's tokenizer
    vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
    clf = LinearSVC()
    # the pipeline to clean, tokenize, vectorize, and classify
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    
    # data
    train = ["I love space. Space is great.", "Planets are cool. I am glad they exist in space", "lol @twitterdude that is gr8", 
            "twitter &amp; reddit are fun.", "Mars is a planet. It is red.", "@Microsoft: y u skip windows 9?", "Rockets launch from Earth and go to other planets.",
            "twitter social media &gt; &lt;", "@someguy @somegirl @twitter #hashtag", "Orbiting the sun is a little blue-green planet."]
    labelsTrain = ["space", "space", "twitter", "twitter", "space", "twitter", "space", "twitter", "twitter", "space"]
    
    test = ["i h8 riting comprehensibly #skoolsux", "planets and stars and rockets and stuff"]
    labelsTest = ["twitter", "space"]
    
    # train
    pipe.fit(train, labelsTrain)
    
    # test
    preds = pipe.predict(test)
    print("----------------------------------------------------------------------------------------------")
    print("results:")
    for (sample, pred) in zip(test, preds):
        print(sample, ":", pred)
    print("accuracy:", accuracy_score(labelsTest, preds))
    
    print("----------------------------------------------------------------------------------------------")
    print("Top 10 features used to predict: ")
    # show the top features
    printNMostInformative(vectorizer, clf, 10)
    
    print("----------------------------------------------------------------------------------------------")
    print("The original data as it appeared to the classifier after tokenizing, lemmatizing, stoplisting, etc")
    # let's see what the pipeline was transforming the data into
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
    transform = pipe.fit_transform(train, labelsTrain)
    
    # get the features that the vectorizer learned (its vocabulary)
    vocab = vectorizer.get_feature_names()
    
    # the values from the vectorizer transformed data (each item is a row,column index with value as # times occuring in the sample, stored as a sparse matrix)
    for i in range(len(train)):
        s = ""
        indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
        numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
        for idx, num in zip(indexIntoVocab, numOccurences):
            s += str((vocab[idx], num))
        print("Sample {}: {}".format(i, s))

    ----------------------------------------------------------------------------------------------
    results:
    i h8 riting comprehensibly #skoolsux : twitter
    planets and stars and rockets and stuff : space
    accuracy: 1.0
    ----------------------------------------------------------------------------------------------
    Top 10 features used to predict: 
    Class 1 best: 
    (-0.52882810587037121, 'planet')
    (-0.35193565503626856, 'space')
    (-0.2182987490483107, 'mar')
    (-0.2182987490483107, 'red')
    (-0.15592826214493352, 'earth')
    (-0.15592826214493352, 'launch')
    (-0.15592826214493352, 'rocket')
    (-0.1482804579342584, 'great')
    (-0.1482804579342584, 'love')
    (-0.099226355509375405, 'blue')
    Class 2 best: 
    (0.41129938045689757, 'twitter')
    (0.34038557663231445, '@mention')
    (0.23401502570811406, 'lol')
    (0.23401502570811406, 'gr8')
    (0.20564996854629114, 'social')
    (0.20564996854629114, 'medium')
    (0.20564941191060651, 'reddit')
    (0.20564941191060651, 'fun')
    (0.10637055092420053, 'y')
    (0.10637055092420053, 'window')
    ----------------------------------------------------------------------------------------------
    The original data as it appeared to the classifier after tokenizing, lemmatizing, stoplisting, etc
    Sample 0: ('love', 1)('space', 2)('great', 1)
    Sample 1: ('space', 1)('planet', 1)('cool', 1)('glad', 1)('exist', 1)
    Sample 2: ('lol', 1)('@mention', 1)('gr8', 1)
    Sample 3: ('twitter', 1)('reddit', 1)('fun', 1)
    Sample 4: ('planet', 1)('mar', 1)('red', 1)
    Sample 5: ('@mention', 1)('y', 1)('u', 1)('skip', 1)('window', 1)('9', 1)
    Sample 6: ('planet', 1)('rocket', 1)('launch', 1)('earth', 1)
    Sample 7: ('twitter', 1)('social', 1)('medium', 1)
    Sample 8: ('@mention', 3)('hashtag', 1)
    Sample 9: ('planet', 1)('orbit', 1)('sun', 1)('little', 1)('blue', 1)('green', 1)



    

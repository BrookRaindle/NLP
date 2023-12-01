import string
from collections import defaultdict

#download stemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

positive_dir = "positiveMessages.txt"
negetive_dir = "negativeMessages.txt"

class NLP:

    def __init__(self):
        self.positive_strings = list()
        self.negetive_strings = list()

        self.processed_positive = set()
        self.processed_negetive = set()

    def getPositive(self):
        return self.positive_strings
    
    def getNegetive(self):
        return self.negetive_strings
    
    def getProcessedPos(self):
        return self.processed_positive
    
    def getProcessedNeg(self):
        return self.processed_negetive

    def buildStringListFromTxt(self, dir, List):
        with open(dir, "r") as text:
            sentences = text.readlines()
        for s in sentences:
            List += s.split(".")
        for i in List:
            if i == "\n":
                List.remove(i)
        #print(f"Display: {List}")
        #print(List)

    def processLists(self, List, Processed):
        for i in List:
            j = self.preprocessString(i)
            Processed.add(j)
        #print(f"Display: {Processed}")

            
    def preprocessString(self, sentence):
        # Convert to lowercase
        outputSentence = sentence.lower()
        wordsToRemove = [" if ", " and ", " on ", " a ", " s ", " the ", " my "]
        # Remove Stop Words
        if wordsToRemove:
            for word in wordsToRemove:
                outputSentence = outputSentence.replace(word, " ")

        # Remove punctuation
        translator = str.maketrans("", "", string.punctuation)
        complete_output = outputSentence.translate(translator)

        # Stemming
        stemmer = PorterStemmer()
        words = word_tokenize(complete_output)
        stemmedWords = [stemmer.stem(word) for word in words]
        complete_output = " ".join(stemmedWords)

        return complete_output
    
    def bagOfWords(self):
        self.bagOfWords = defaultdict(lambda: {'good': 1, 'bad': 1})
        self.totalGood = 0
        self.totalBad = 0
        
        for sentence in self.processed_positive:
            words = sentence.split()
            for word in words:
                self.totalGood += 1
                self.bagOfWords[word]['good'] += 1
        for sentence in self.processed_negetive:
            words = sentence.split()
            for word in words:
                self.bagOfWords[word]['bad'] += 1
                self.totalBad += 1
        self.bagOfWords = dict(self.bagOfWords)
        print(self.bagOfWords)
        print(len(self.bagOfWords))
        print(f"In total there are: {self.totalGood} Good Words, and {self.totalBad} Bad Words")

    def probabilityPositive(self, word):
        if word in self.bagOfWords:
            return (self.bagOfWords[word]['good'])/(self.totalGood+(len(self.bagOfWords)))
        else: 
            #print(f"Word: '{word}' not in bag")
            pass

    def probabilityNegetive(self, word):
        if word in self.bagOfWords:
            return (self.bagOfWords[word]['bad'])/(self.totalBad+(len(self.bagOfWords)))
        else: 
            #print(f"Word: '{word}' not in bag")
            pass

    def probabilityString(self, string):
        processed = self.preprocessString(string)
        words = processed.split()
        totalPos = 0
        totalNeg = 0
        Decision = None
        for word in words:
            #print(word)
            probPos = self.probabilityPositive(word)
            probNeg = self.probabilityNegetive(word) 
            if probPos != None:
                totalPos += probPos
            if probNeg != None:
                totalNeg += probNeg

        tot = totalPos+totalNeg

        if totalNeg < totalPos:
            Decision = "Positive"
            Certainty = round((totalPos/tot)*100)
        else:
            Decision = "Negetive"
            Certainty = round((totalNeg/tot)*100)
        return (f"Positive: {round(totalPos*100)}%, Negetive: {round(totalNeg*100)}%", Decision, Certainty)

    def test(self):
        test = "The hotel lobby welcomed guests with its modern decor and a friendly receptionist at the front desk."
        test_result = self.probabilityString(test)
        print(f"\n{test}")
        print(test_result[0])
        print(f"Overall Sentiment of Sentence = {test_result[1]}")
        print(f"Certainty = {test_result[2]}%\n")
#Count every word (seperates by " ") if there are duplicates, then add a counter for them

def main():
    processor = NLP()
    #Read data-set off txt files
    processor.buildStringListFromTxt(positive_dir, processor.getPositive())
    processor.buildStringListFromTxt(negetive_dir, processor.getNegetive())
    #Process each data-Set
    processor.processLists(processor.getPositive(),processor.getProcessedPos())
    processor.processLists(processor.getNegetive(),processor.getProcessedNeg())
    processor.bagOfWords()
    processor.test()
if __name__ == "__main__":
    main()


# Credits - 
# Jack Fermer = (50x Pos+Neg Sentences for training)
# Connor Wood = (50x Pos+Neg Sentences for training)
# OpenAI GPT = (200x Pos+Neg Sentences for training)
import nltk
import re
import heapq
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
# Load the models here
# nltk.download('all')
print("Loading models...")
stopwords = nltk.corpus.stopwords.words('english')

print("Loading gtp2 model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gtp2_model = TFGPT2LMHeadModel.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id)

print("Loading bert model...")
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
print("All models  are loaded.")
###############################################################################


def get_important_sentences(text, n=None):
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    formatted_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)
    sentence_list = nltk.sent_tokenize(text)

    if n is None:
        n = max(int(len(sentence_list) * 0.6), 3)

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_text):
        if word not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    sentences = heapq.nlargest(n, sentence_scores, key=sentence_scores.get)
    return sentences


def complete_sentence(sentence, compare_to):
    encoded_input = tokenizer.encode(sentence, return_tensors='tf')
    num_return_sequences = 3
    outputs = gtp2_model.generate(
        encoded_input,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        top_p=0.80,
        top_k=60,
        repetition_penalty=10.0,
    )
    sentences = [
        tokenizer.decode(outputs[i],
                         skip_special_tokens=True).split(",;.?!")[0].replace(
                             ", ",
                             ". ").replace(u'\xa0', u' ').replace("; ",
                                           ". ").replace("? ",
                                                         ". ").split(".")[0]
        for i in range(num_return_sequences)
    ]

    # Compare to the original sentence and return the dissimilar sentence
    if compare_to:
        sentence_embeddings = bert_model.encode([compare_to] + sentences)
        similarities = cosine_similarity([sentence_embeddings[0]],
                                         sentence_embeddings[1:])
    return sorted(sentences, key=lambda x: similarities[0][sentences.index(x)])




# print("\n\nResult\n")
# res = complete_sentence("The boy is sitting",
#                         "The boy is sitting under the tree")
# print(res)
# print()

# res = complete_sentence(
#     "The goals of information security is ",
#     "The goals of information security is confidentiality.")
# print(res)
# print()

# res = complete_sentence("Computers take time ",
#                         "Computers take time to do complicated calculations.")
# print(res)
# print()

# text = 'The quick brown fox jumped over the lazy dog.'
# text = """That’s great on the theory and logic behind the process — but how do we apply this in reality?

# We’ll outline two approaches — the easy way and the slightly more complex way.

# Easy — Sentence-Transformers
# The easiest approach for us to implement everything we just covered is through the sentence-transformers library — which wraps most of this process into a few lines of code.

# First, we install sentence-transformers using pip install sentence-transformers. This library uses HuggingFace’s transformers behind the scenes — so we can actually find sentence-transformers models here."""
# # print(get_important_sentences(text))

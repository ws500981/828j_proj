import pandas as pd
import numpy as np
import random

import kagglehub
import shutil

import random
random.seed(49)
import string

#source_dir = kagglehub.dataset_download("duketemon/antonyms-wordnet") # https://www.kaggle.com/datasets/duketemon/antonyms-wordnet
#shutil.move(source_dir, destination_dir)
antonyms = pd.read_csv('/home/wuw15/data_dir/cwproj/data/antonyms_chosen.csv')[['lemma', 'antonyms']]

#source_dir = kagglehub.dataset_download("hserdaraltan/countries-by-continent")
#destination_dir = "/home/wuw15/data_dir/cwproj/data/" # https://www.kaggle.com/datasets/hserdaraltan/countries-by-continent
#shutil.move(source_dir, destination_dir)
countries = pd.read_csv('/home/wuw15/data_dir/cwproj/data/countriesbycontinents.csv').rename(columns={"Country": "name", "Continent": "type"})
country_list = list(countries['name'])
nations = pd.DataFrame({'type': ['nation'] * len(country_list), 'name': country_list})
kingdoms = pd.DataFrame({'type': ['kingdom'] * len(country_list), 'name': country_list})
places = pd.DataFrame({'type': ['place'] * len(country_list), 'name': country_list})

# https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
animal_list = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", 
    "butterfly", "cat", "caterpillar", "chimpanzee", "cockroach", "cow", 
    "coyote", "crab", "crow", "deer", "dog", "dolphin", "donkey", 
    "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", 
    "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", 
    "hare", "hedgehog", "hippopotamus", "hornbill", "horse", "hummingbird", 
    "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", 
    "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus", 
    "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", 
    "parrot", "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", 
    "possum", "raccoon", "rat", "reindeer", "rhinoceros", "sandpiper", 
    "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", 
    "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", 
    "wolf", "wombat", "woodpecker", "zebra"
]
animals = pd.DataFrame({'type': ['animal'] * len(animal_list), 'name': animal_list})
creatures = pd.DataFrame({'type': ['creature'] * len(animal_list), 'name': animal_list})

# https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
flower_list = [
    "Daffodil", "Snowdrop", "Lily Valley", "Bluebell", "Crocus", 
    "Iris", "Tigerlily", "Tulip", "Fritillary", "Sunflower", 
    "Daisy", "Coltsfoot", "Dandelion", "Cowslip", "Buttercup", 
    "Windflower", "Pansy"
]
flowers = pd.DataFrame({'type': ['flower'] * len(flower_list), 'name': flower_list})
plants = pd.DataFrame({'type': ['plant'] * len(flower_list), 'name': flower_list})

combined = pd.concat([flowers, plants, countries, places, nations, kingdoms, animals, creatures], axis = 0)

#-------------------------------------------------------------------
# generate synthetic words: no semantic meaning at all
# variable lengths
def generate_random_words_var_len(num_words, word_length_range=(3, 8)):
    words = set()
    for _ in range(num_words):
        length = random.randint(*word_length_range)
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        words.add(word)
    return list(words)

#no_or_words = 20
#synthetic_words = generate_random_words_var_len(no_or_words)

# fixed length
def generate_random_words_fixed_len(num_words, word_length):
    words = set()
    while len(words) < num_words:
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.add(word)
    return list(words)

#synthetic_words = generate_random_words_fixed_len(20, 4)
#-------------------------------------------------------------------
# generate synthetic numbers as words: some semantic meaning when we use > as forward and < as reverse but = as independent
def generate_random_numbers_fixed_range(num_words, word_range):
    words = set()
    while len(words) < num_words:
        word = random.randint(1, word_range)
        words.add(word)
    return list(words)

#synthetic_words = generate_random_numbers_fixed_range(20, 1e3)
#-------------------------------------------------------------------
# create pair sets
def create_pair_set(words, pair_type, no_pairs, typee='word'):
    pairs = set()
    if pair_type == 'contradict':
        while len(pairs) < no_pairs:
            word = random.sample(words,1)[0] if typee == 'synthetic' else words.iloc[random.randint(0, len(words)-1)]
            pair = (word, word) if typee == 'synthetic' else (word.iloc[0], word.iloc[1])
            pairs.add(pair)
    elif pair_type == 'entail':
        while len(pairs) < no_pairs:
            word = sorted(random.sample(words,2)) if typee == 'synthetic' else words.iloc[random.randint(0, len(words)-1)]
            pair = (word[0], word[1]) if typee == 'synthetic' else (word.iloc[1], word.iloc[0])
            pairs.add(pair)
    return list(pairs)

#-------------------------------------------------------------------
# add quantifier and create the final dataset
def generate_entailment_sequences(dataset0, dataset1, n_samples):
    sequences = set()
    for _ in range(n_samples):
        # Randomly choose forward/reverse entailment or contradiction, we do not need independence
        # A reverse entails B, which means not B forward entails not A
        dataset_type = random.choices([0, 1], weights=(1/3, 2/3))[0]

        if dataset_type == 0:
            pair = random.choice(dataset0)
            A, B = pair
            
            # For dataset0: A is a contradiction of B
            template_types = [
                (f"Given {A} contradicts {B}, does {A} forward entail {B}?", "No"),
                (f"Given {A} contradicts {B}, does {A} forward entail not {B}?", "Yes"),
                (f"Given {A} contradicts {B}, does not {A} forward entail {B}?", "Yes"),
                (f"Given {A} contradicts {B}, does not {A} forward entail not {B}?", "No"),
                (f"Given {A} contradicts {B}, does {B} forward entail {A}?", "No"),
                (f"Given {A} contradicts {B}, does {B} forward entail not {A}?", "Yes"),
                (f"Given {A} contradicts {B}, does not {B} forward entail {A}?", "Yes"),
                (f"Given {A} contradicts {B}, does not {B} forward entail not {A}?", "No"),

                (f"Given {B} contradicts {A}, does {A} forward entail {B}?", "No"),
                (f"Given {B} contradicts {A}, does {A} forward entail not {B}?", "Yes"),
                (f"Given {B} contradicts {A}, does not {A} forward entail {B}?", "Yes"),
                (f"Given {B} contradicts {A}, does not {A} forward entail not {B}?", "No"),
                (f"Given {B} contradicts {A}, does {B} forward entail {A}?", "No"),
                (f"Given {B} contradicts {A}, does {B} forward entail not {A}?", "Yes"),
                (f"Given {B} contradicts {A}, does not {B} forward entail {A}?", "Yes"),
                (f"Given {B} contradicts {A}, does not {B} forward entail not {A}?", "No"),

                (f"Given {A} contradicts {B}, does {A} reverse entail {B}?", "No"),
                (f"Given {A} contradicts {B}, does {A} reverse entail not {B}?", "Yes"),
                (f"Given {A} contradicts {B}, does not {A} reverse entail {B}?", "Yes"),
                (f"Given {A} contradicts {B}, does not {A} reverse entail not {B}?", "No"),
                (f"Given {A} contradicts {B}, does {B} reverse entail {A}?", "No"),
                (f"Given {A} contradicts {B}, does {B} reverse entail not {A}?", "Yes"),
                (f"Given {A} contradicts {B}, does not {B} reverse entail {A}?", "Yes"),
                (f"Given {A} contradicts {B}, does not {B} reverse entail not {A}?", "No"),

                (f"Given {B} contradicts {A}, does {A} reverse entail {B}?", "No"),
                (f"Given {B} contradicts {A}, does {A} reverse entail not {B}?", "Yes"),
                (f"Given {B} contradicts {A}, does not {A} reverse entail {B}?", "Yes"),
                (f"Given {B} contradicts {A}, does not {A} reverse entail not {B}?", "No"),
                (f"Given {B} contradicts {A}, does {B} reverse entail {A}?", "No"),
                (f"Given {B} contradicts {A}, does {B} reverse entail not {A}?", "Yes"),
                (f"Given {B} contradicts {A}, does not {B} reverse entail {A}?", "Yes"),
                (f"Given {B} contradicts {A}, does not {B} reverse entail not {A}?", "No"),

                (f"Given {A} contradicts {B}, does {A} contradict not {B}?", "No"),
                (f"Given {A} contradicts {B}, does not {A} contradict {B}?", "No"),
                (f"Given {A} contradicts {B}, does not {A} contradict not {B}?", "Yes"),
                (f"Given {A} contradicts {B}, does {B} contradict {A}?", "Yes"),
                (f"Given {A} contradicts {B}, does {B} contradict not {A}?", "No"),
                (f"Given {A} contradicts {B}, does not {B} contradict {A}?", "No"),
                (f"Given {A} contradicts {B}, does not {B} contradict not {A}?", "Yes"),

                (f"Given {B} contradicts {A}, does {A} contradict {B}?", "Yes"),
                (f"Given {B} contradicts {A}, does {A} contradict not {B}?", "No"),
                (f"Given {B} contradicts {A}, does not {A} contradict {B}?", "No"),
                (f"Given {B} contradicts {A}, does not {A} contradict not {B}?", "Yes"),
                (f"Given {B} contradicts {A}, does {B} contradict not {A}?", "No"),
                (f"Given {B} contradicts {A}, does not {B} contradict {A}?", "No"),
                (f"Given {B} contradicts {A}, does not {B} contradict not {A}?", "Yes")

            ]
        
        elif dataset_type == 1:
            # Choose random pair from dataset1
            pair = random.choice(dataset1)
            A, B = pair
            
            # For dataset1: A forward entails B
            template_types = [
                (f"Given {A} forward entails {B}, does {A} forward entail not {B}?", "No"),
                (f"Given {A} forward entails {B}, does not {A} forward entail {B}?", "No"),
                (f"Given {A} forward entails {B}, does not {A} forward entail not {B}?", "No"),
                (f"Given {A} forward entails {B}, does {B} forward entail {A}?", "No"),
                (f"Given {A} forward entails {B}, does {B} forward entail not {A}?", "No"),
                (f"Given {A} forward entails {B}, does not {B} forward entail {A}?", "No"),
                (f"Given {A} forward entails {B}, does not {B} forward entail not {A}?", "Yes"),

                (f"Given {A} forward entails {B}, does {A} reverse entail {B}?", "No"),
                (f"Given {A} forward entails {B}, does {A} reverse entail not {B}?", "No"),
                (f"Given {A} forward entails {B}, does not {A} reverse entail {B}?", "No"),
                (f"Given {A} forward entails {B}, does not {A} reverse entail not {B}?", "Yes"),
                (f"Given {A} forward entails {B}, does {B} reverse entail {A}?", "Yes"),
                (f"Given {A} forward entails {B}, does {B} reverse entail not {A}?", "No"),
                (f"Given {A} forward entails {B}, does not {B} reverse entail {A}?", "No"),
                (f"Given {A} forward entails {B}, does not {B} reverse entail not {A}?", "No"),

                (f"Given {A} forward entails {B}, does {A} contradict {B}?", "No"),
                (f"Given {A} forward entails {B}, does {A} contradict not {B}?", "No"),
                (f"Given {A} forward entails {B}, does not {A} contradict {B}?", "No"),
                (f"Given {A} forward entails {B}, does not {A} contradict not {B}?", "No"),
                (f"Given {A} forward entails {B}, does {B} contradict {A}?", "No"),
                (f"Given {A} forward entails {B}, does {B} contradict not {A}?", "No"),
                (f"Given {A} forward entails {B}, does not {B} contradict {A}?", "No"),
                (f"Given {A} forward entails {B}, does not {B} contradict not {A}?", "No"),

                (f"Given {B} reverse entails {A}, does {A} contradict {B}?", "No"),
                (f"Given {B} reverse entails {A}, does {A} contradict not {B}?", "No"),
                (f"Given {B} reverse entails {A}, does not {A} contradict {B}?", "No"),
                (f"Given {B} reverse entails {A}, does not {A} contradict not {B}?", "No"),
                (f"Given {B} reverse entails {A}, does {B} contradict {A}?", "No"),
                (f"Given {B} reverse entails {A}, does {B} contradict not {A}?", "No"),
                (f"Given {B} reverse entails {A}, does not {B} contradict {A}?", "No"),
                (f"Given {B} reverse entails {A}, does not {B} contradict not {A}?", "No"),

                (f"Given {B} reverse entails {A}, does {A} forward entail {B}?", "Yes"),
                (f"Given {B} reverse entails {A}, does {A} forward entail not {B}?", "No"),
                (f"Given {B} reverse entails {A}, does not {A} forward entail {B}?", "No"),
                (f"Given {B} reverse entails {A}, does not {A} forward entail not {B}?", "No"),
                (f"Given {B} reverse entails {A}, does {B} forward entail {A}?", "No"),
                (f"Given {B} reverse entails {A}, does {B} forward entail not {A}?", "No"),
                (f"Given {B} reverse entails {A}, does not {B} forward entail {A}?", "No"),
                (f"Given {B} reverse entails {A}, does not {B} forward entail not {A}?", "Yes"),

                (f"Given {B} reverse entails {A}, does {A} reverse entail not {B}?", "No"),
                (f"Given {B} reverse entails {A}, does not {A} reverse entail {B}?", "No"),
                (f"Given {B} reverse entails {A}, does not {A} reverse entail not {B}?", "Yes"),
                (f"Given {B} reverse entails {A}, does {A} reverse entail {B}?", "No"),
                (f"Given {B} reverse entails {A}, does {B} reverse entail not {A}?", "No"),
                (f"Given {B} reverse entails {A}, does not {B} reverse entail {A}?", "No"),
                (f"Given {B} reverse entails {A}, does not {B} reverse entail not {A}?", "No")

            ]
        
        # Choose random template
        question, answer = random.choice(template_types)
        sequences.add(f"{question} {answer}")
    return sequences

#-------------------------------------------------------------------
# use synthetic words with fixed length
no_pairs = 20
synthetic_words = generate_random_words_fixed_len(20, 4)
dataset0 = create_pair_set(synthetic_words, 'contradict', no_pairs, typee = 'synthetic')
dataset1 = create_pair_set(synthetic_words, 'entail', no_pairs, typee = 'synthetic') # includes forward and reverse entail

# use synthetic words with variable length
no_pairs = 20
synthetic_words = generate_random_words_var_len(20,)
dataset0 = create_pair_set(synthetic_words, 'contradict', no_pairs, typee = 'synthetic')
dataset1 = create_pair_set(synthetic_words, 'entail', no_pairs, typee = 'synthetic') # includes forward and reverse entail

# use numbers
no_pairs = 20
synthetic_words = generate_random_numbers_fixed_range(20, 1e3)
dataset0 = create_pair_set(synthetic_words, 'contradict', no_pairs, typee = 'synthetic')
dataset1 = create_pair_set(synthetic_words, 'entail', no_pairs, typee = 'synthetic') # includes forward and reverse entail

# use actual words
dataset0 = create_pair_set(antonyms, 'contradict', no_pairs, typee = 'word')
dataset1 = create_pair_set(combined, 'entail', no_pairs, typee = 'word')

#-------------------------------------------------------------------

n_samples = 20
sequences = generate_entailment_sequences(dataset0, dataset1, n_samples)

# create dataset
sequences = list(sequences)
texts = [item.split('? ')[0]+'?' for item in sequences]
labels = [0 if item.split('? ')[1] == 'No' else 1 for item in sequences]
dataset = pd.DataFrame({'text': texts, 'label': labels})
dataset.to_csv('/home/wuw15/data_dir/cwproj/dataset.csv', index=False)

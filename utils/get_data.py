# import kagglehub
import shutil
import os

import random
random.seed(49)
import string
import pandas as pd
import numpy as np
import json

import argparse

# utility functions
# creates a dataframe for generating real data
def create_real_data(countries, temp):
    # countries
    country_list = list(countries['name'])
    nations = pd.DataFrame({'type': ['nation'] * len(country_list), 'name': country_list})
    kingdoms = pd.DataFrame({'type': ['kingdom'] * len(country_list), 'name': country_list})
    places = pd.DataFrame({'type': ['place'] * len(country_list), 'name': country_list})

    # flowers and plants
    flower_list = temp["flowers"]
    flowers = pd.DataFrame({'type': ['flower'] * len(flower_list), 'name': flower_list})
    plants = pd.DataFrame({'type': ['plant'] * len(flower_list), 'name': flower_list})

    # animal and creature
    animal_list = temp["animals"]
    animals = pd.DataFrame({'type': ['animal'] * len(animal_list), 'name': animal_list})
    creatures = pd.DataFrame({'type': ['creature'] * len(animal_list), 'name': animal_list})

    # combine everything
    combined = pd.concat([countries, nations, kingdoms, places, flowers, plants, animals, creatures], axis = 0)

    return combined

# generate synthetic words: no semantic meaning at all
def generate_random_words(num_words, word_length, typ=0):
    words = set()
    # 0 -> numbers, 1 -> variable, 2 -> fixed
    if typ == 0:
        # generate synthetic numbers as words: some semantic meaning when we use > as forward and < as reverse but = as independent
        while len(words) < num_words:
            word = random.randint(1, word_length)
            words.add(word)
    elif typ == 1:
        # for fixed length 
        while len(words) < num_words:
            word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
            words.add(word)
    elif typ == 2:
        # for random length
        for _ in range(num_words):
            length = random.randint(*word_length)
            word = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
            words.add(word)

    return list(words)

# add quantifier and create the final dataset
def generate_entailment_sequences(dataset0, dataset1, n_samples):
    sequences = set()
    for _ in range(n_samples):
        # Randomly choose forward/reverse entailment or contradiction, we do not need independence
        # A reverse entails B, which means not B forward entails not A
        dataset_type = random.choices([0, 1], weights=(1/3, 2/3))[0]
        f = open("./../data/templates.json","r")
        json_obj = json.load(f)

        if dataset_type == 0:
            pair = random.choice(dataset0)
            A, B = pair
            # For dataset0: A is a contradiction of B
            template_types = [[i[0].format(A,B),i[1]] for i in json_obj["contradiction"]]
        
        elif dataset_type == 1:
            # Choose random pair from dataset1
            pair = random.choice(dataset1)
            A, B = pair
            # For dataset1: A forward entails B
            template_types = [[i[0].format(A,B),i[1]] for i in json_obj["entailment"]]
        
        # Choose random template
        question, answer = random.choice(template_types)
        sequences.add(f"{question} {answer}")
    return sequences

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

# classes
# downloading and preparing base
class download_data(object):
    def __init__(self, pth):
        super(download_data, self).__init__()
        self.pth = pth
        if os.path.isdir(os.path.join(self.pth,"data")):
            try:
                os.mkdir(os.path.join(self.pth,"data"))
                self.pth = os.path.join(self.pth,"data")
            except Exception as e:
                print("directory not created")
                      
    def forward(self):
        # https://www.kaggle.com/datasets/duketemon/antonyms-wordnet
        # antonyms
        source_dir = kagglehub.dataset_download("duketemon/antonyms-wordnet") 
        destination_dir = self.pth
        shutil.move(source_dir, destination_dir)

        # https://www.kaggle.com/datasets/hserdaraltan/countries-by-continent
        # countries
        source_dir = kagglehub.dataset_download("hserdaraltan/countries-by-continent")
        destination_dir = self.pth 
        shutil.move(source_dir, destination_dir)

        # https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
        # animals
        temp = {0}
        animal_list = ["antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", 
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
            "wol", "wombat", "woodpecker", "zebra"
        ]
        temp['animals'] = animal_list

        # https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
        # flowers
        flower_list = [
            "Daffodil", "Snowdrop", "Lily Valley", "Bluebell", "Crocus", 
            "Iris", "Tigerlily", "Tulip", "Fritillary", "Sunflower", 
            "Daisy", "Coltsfoot", "Dandelion", "Cowslip", "Buttercup", 
            "Windflower", "Pansy"
        ]
        temp['flowers'] = flower_list

        # save temp as a json file
        with open(os.path.join(self.pth, "extras.json"), "w") as outfile: 
            json.dump(temp, outfile)

# loading real and synthetic data
class load_data(object):
    def __init__(self):
        super(load_data, self).__init__()
        f = open("./../data/extras.json","r")
        self.temp = json.load(f)
        self.countries = pd.read_csv('./../data/countriesbycontinents.csv').rename(columns={"Country": "name", "Continent": "type"})
        self.combined = create_real_data(self.countries, self.temp)
        self.antonyms = pd.read_csv('./../data/antonyms_chosen.csv')[['lemma', 'antonyms']]

    def create_data_syn(self, n_samples, num_words, word_length, word_type):
        no_pairs = 20
        # get the sequences
        words = generate_random_words(num_words, word_length, typ = word_type)
        dataset0 = create_pair_set(words, 'contradict', no_pairs, "synthetic")
        dataset1 = create_pair_set(words, 'entail', no_pairs, "synthetic")
        sequences = generate_entailment_sequences(dataset0, dataset1, n_samples)

        # prepare the dataset
        sequences = list(sequences)
        texts = [item.split('? ')[0]+'?' for item in sequences]
        labels = [0 if item.split('? ')[1] == 'No' else 1 for item in sequences]
        dataset = pd.DataFrame({'text': texts, 'label': labels})
        return dataset

    def create_data_real(self, n_samples):
        no_pairs = 20
        # get the sequences
        dataset0 = create_pair_set(self.antonyms, 'contradict', no_pairs,'word')
        dataset1 = create_pair_set(self.combined, 'entail', no_pairs,'word')
        sequences = generate_entailment_sequences(dataset0, dataset1, n_samples)

        # prepare the dataset
        sequences = list(sequences)
        texts = [item.split('? ')[0]+'?' for item in sequences]
        labels = [0 if item.split('? ')[1] == 'No' else 1 for item in sequences]
        dataset = pd.DataFrame({'text': texts, 'label': labels})
        return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='This code will be legend - wait for it - ary.. legendary!')
    parser.add_argument('n_samples', type = int, help ='insert number of samples.')
    parser.add_argument('num_words', type = int, help ='insert number of words.')
    parser.add_argument('word_length', nargs='+', type=int, help ='insert word length.')
    parser.add_argument('word_type', type = int, help ='insert type of word. 0-numbers, 1-fixed, 2-variable')
    parser.add_argument("is_real", type=int, help = "real or synthetic data.")
    args = parser.parse_args()

    n_samples = args.n_samples
    num_words = args.num_words
    word_length = args.word_length
    word_type = args.word_type
    is_real = args.is_real

    obj = load_data()

    if word_type != 2:
        word_length = word_length[0]
    else:
        word_length = tuple(word_length)

    if is_real == 0:
        # create synthetic dataset
        data = obj.create_data_syn(n_samples, num_words, word_length, word_type)
    else:
        # create real dataset
        data = obj.create_data_real(n_samples)
    data.to_csv("./../test.csv")

    
    
    


    








### Authors: Tomer Gotesdyner and Yanai Levi ###

### Imports ###
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from Bio import SeqIO
import random

class SeqDataset(Dataset):

    # constructor
    def __init__(self, pathToData, metaDataFile):
        self.pathToData = pathToData
        self.metaDataFile = metaDataFile
        # load the meta data as dataframe
        self.labels = pd.read_csv(metaDataFile, sep="\t", header=0)

        # Open the FASTA file and parse the records
        records = SeqIO.parse(pathToData, "fasta")

        counter = 0
        self.id_to_seq_map = {}
        print(len(self.labels)-1)
        # Iterate over the records and print the headers and sequences
        for record in records:
            if counter >= len(self.labels):
                break
            id = record.id
            # class_id = self.labels[self.labels.id == id].loc[:, 'class_id'].values[0]
            # if class_id != 5 and class_id != 7:
            #     continue
            # Check if the record.id is in the meta data file
            if id in self.labels['id'].values and not id in self.id_to_seq_map:
                # save the record in the map
                self.id_to_seq_map[id] = record.seq
                counter += 1
                # print("counter: ", counter)

        for id in self.labels['id'].values:
            if id not in self.id_to_seq_map:
                self.labels = self.labels[self.labels.id != id]
        
        self.char_to_value = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3
        }
        
        self.possible_errors = {'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'],
                 'K': ['G', 'T'], 'M': ['A', 'C'], 'B': ['C', 'G', 'T'],
                 'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A']}

    # Return the length of the dataset
    def __len__(self):
        return len(self.id_to_seq_map)

    # get an item by index
    def __getitem__(self,idx):
        # get the idx line from the meta data file
        line = self.labels.iloc[idx]

        # get the sequence from the map
        seq = self.id_to_seq_map[line['id']]

        list_of_ints = []
        tempInt = 0
        counter = 0

        for char in str(seq):
            counter += 1
            if char in self.char_to_value:
                tempInt += self.char_to_value[char]*(4**(4-counter))
            elif char in self.possible_errors:
                tempInt += self.char_to_value[random.choice(self.possible_errors[char])]*(4**(4-counter))
            else:
                print("Error: char is not A, C, G, or T")
            
            if counter >= 4:
                # print(tempInt)
                list_of_ints.append(tempInt)
                tempInt = 0
                counter = 0

        image = torch.tensor(list_of_ints, dtype=torch.float)
        # if it is longer than 13000, cut it
        image = image[:13000]
        # padd the image with zeros to size 13000
        image = F.pad(image, (0, 13000 - len(image)), 'constant', 0)

        # normalize the image to mean 0 and std 1
        image = (image - torch.mean(image)) / torch.std(image)
        # normalize the image to values between -1 to 1
        image = (image) / torch.max(torch.abs(image))
        
        return image, line['class_id']
    
    
    def getNumOfClasses(self):
        return 10#len(self.labels['class_id'].unique())
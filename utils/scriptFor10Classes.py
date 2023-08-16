import pandas as pd
from Bio import SeqIO

# Open the FASTA file and parse the records
records = SeqIO.parse("C:/Users/ylevi/Downloads/viral_sequences/sequences/fasta/20230611-1654.fasta", "fasta")

# print len of one record
print(len(next(records).seq))

metaDataFile = "download"
df = pd.read_csv(metaDataFile, sep="\t", header=0)

# create a map of lineage to class id
map = {}

new_df = pd.DataFrame(columns=['id', 'lineage', 'class_id'])

counter = [0 for i in range(10)]
# iterate over the rows of the dataframe
for index, row in df.iterrows():
    if not pd.isna(row['lineage']):
        if row['lineage'] in ['P.1', 'AY.43', 'B.1.351', 'B.1.1.529', 'B.1.617.2', 'B.1.525', 'R.1', 'B.1.1.7', 'B.1.1', 'Q.4']: #['B.1.1.285', 'AY.43', 'BA.1.1.2', 'BC.1', 'B.1.1.214', 'B.1.1', 'R.1', 'B.1.1.7', 'B.1.346', 'Q.4']
            # check if the lineage is already in the map
            if row['lineage'] not in map:
                # add the lineage to the map
                map[row['lineage']] = int(len(map))
                print(row['lineage'], len(map))
            if counter[map[row['lineage']]] < 1000:  
                # concat a new row to the new dataframe
                new_df = pd.concat([new_df, pd.DataFrame([[row['id'], row['lineage'], map[row['lineage']]]], columns=['id', 'lineage', 'class_id'])], ignore_index=True)

                counter[map[row['lineage']]] += 1
                if counter[map[row['lineage']]] >= 1000:
                    print("finished with class id: ", map[row['lineage']])
                    print("status: ", counter)
                    if sum(counter) >= 10000:
                        break
      
id_to_seq_map = {}       
counter = 0   
for record in records:
    if counter >= len(new_df):
        break
    id = record.id.split("|")[1]
    # Check if the record.id is in the meta data file
    if id in new_df['id'].values and not id in id_to_seq_map:
        # save the record in the map
        id_to_seq_map[id] = record.seq
        counter += 1
        print("counter: ", counter)

for id in new_df['id'].values:
    if id not in id_to_seq_map:
        new_df = new_df[new_df.id != id]
        
# save all sequences of the map to a file
# create a file to save the sequences
new_file = open("10_classes_new_data.fasta", "w")
for id in id_to_seq_map:
    new_file.write(">" + id + "\n")
    new_file.write(str(id_to_seq_map[id]) + "\n") 
new_file.close()  
        
print("status: ", counter)        
# save the dataframe to a file
new_df.to_csv("download_with_class_id_10_classes_new_data", sep="\t", index=False)
print(map)
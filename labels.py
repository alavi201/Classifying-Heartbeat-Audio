import pandas as pd
INPUT_LIB = 'heartbeat-sounds/'
df = pd.read_csv(INPUT_LIB + 'set_a.csv')

#print(df)

new_labels = {}
label_array = []

def read_labels():

    for index, row in df.iterrows():
        file_name = row['fname'].split('/')[1]
        if file_name[:2] == '__':
            file_name = 'Aunlabelledtest' + file_name
        new_labels[file_name] = row['integer_label']
        label_array.append(row['integer_label'])

read_labels()
#print(labels)


#print(labels['normal__201108011118.wav'])

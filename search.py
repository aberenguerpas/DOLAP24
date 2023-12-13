# Leer tablas de datos, hacer query usando el 10% del contenido de la tabla.

# Sacar precision, recall y MMR

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import math
import csv
import pandas as pd
import faiss
import numpy as np
import operator
import time
import argparse
from angle_emb import AnglE


def enconde_text(model_name, model, text):
    if model_name == 'uae-large':
        return model.encode(text, to_numpy=True)
    else:
        return model.encode(text, show_progress_bar=False)


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


def read_files(path):
    return os.listdir(path)


def content_embs(model, df, model_name, size):
    all_embs = np.empty((0, size), dtype=np.float32)
    for col in df.columns:

        # Split content of each column in to chunks
        content = df[col].values.tolist()
        col_cont_split = []
        max_tokens = 200
        for i in range(0, len(content), max_tokens):
            x = i
            chunk = content[x:x+max_tokens]
            chunk = " ".join(str(x) for x in chunk)
            col_cont_split.append(chunk)

        # Create embedding from chunks
        embs = enconde_text(model_name, model, col_cont_split)
        avg_embs = np.mean(embs, axis=0)
        all_embs = np.append(all_embs, [avg_embs], axis=0)

    return all_embs


def main():

    parser = argparse.ArgumentParser(description='Process Darta')
    parser.add_argument('-i', '--input', default='./data/', help='Name of the input folder storing CSV tables')
    parser.add_argument('-m', '--model', default='all',
                        choices=['all', 'uae-large', 'bge-large', 'bge-base', 'embe', 'gte-large', 'gte-base', 'stb'])
    parser.add_argument('-r', '--result', default='./indexs',
                        help='Name of the output folder that stores the indexs files')

    args = parser.parse_args()

    files_path = args.input

    models = []

    if args.model == 'all' or args.model == 'uae-large':
        models.append('uae-large')
    if args.model == 'all' or args.model == 'bge-large':
        models.append('bge-large')
    if args.model == 'all' or args.model == 'bge-base':
        models.append('bge-base')
    if args.model == 'all' or args.model == 'ember':
        models.append('ember')
    if args.model == 'all' or args.model == 'gte-large':
        models.append('gte-large')
    if args.model == 'all' or args.model == 'bge-large':
        models.append('gte-base')
    if args.model == 'all' or args.model == 'stb':
        models.append('stb')

    for model_name in models:
        dimensions = 768

        if model_name == 'uae-large':
            model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
            dimensions = 1024

        if model_name == 'bge-large':
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            dimensions = 1024

        if model_name == 'bge-base':
            model = SentenceTransformer('BAAI/bge-base-en-v1.5')

        if model_name == 'ember':
            model = SentenceTransformer('llmrails/ember-v1')
            dimensions = 1024

        if model_name == 'gte-large':
            model = SentenceTransformer('thenlper/gte-large')
            dimensions = 1024

        if model_name == 'gte-base':
            model = SentenceTransformer('thenlper/gte-base')

        if model_name == 'stb':
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            dimensions = 384

        index = faiss.read_index("./indexs/"+model_name+".index")

        map = pd.read_csv("indexs/"+model_name+"_map.csv")

        files = read_files(files_path)
        n_docs = len(files)

        precision = 0
        mmr = 0

        st = time.time()

        for file in tqdm(files):

            try:
                # Read dataframe
                delimiter = find_delimiter(files_path + file)
                df = pd.read_csv(files_path + file, sep=delimiter, nrows=120)

                if len(df.index) > 100:
                    df = df[:100]

                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')

                # Split dataframe, get 10%
                size = math.ceil(len(df.index) * 0.1)
                df = df[:size]

                # Get embeddings
                embs = content_embs(model, df, model_name, dimensions)

                rank = {}
                results = pd.DataFrame()
                for col, emb in enumerate(embs):
                    distances, ann = index.search(np.array([emb]), k=n_docs)
                    results_aux = pd.DataFrame({'distances': distances[0], 'ann': ann[0], 'col': col})
                    results_aux = pd.merge(results_aux, map, left_on="ann", right_on="id")
                    results = pd.concat([results, results_aux])

                for candidate in list(dict.fromkeys(results['dataset'].tolist())):
                    aux = results[results['dataset'] == candidate]

                    total = 0
                    for i in range(len(df.columns)):
                        similarity = aux[aux['col'] == i]['distances'].max()
                        if similarity is not np.nan:
                            total += similarity
                        else:
                            total += 0

                    rank[candidate] = total/len(df.columns)

                sorted_d = dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True))

                counter = 1
                #print("------")
                #print(file)
                #print("------")
                for key, value in sorted_d.items():
                    #print(key, round(value, 2))
                    if counter == 1 and key == file:
                        precision += 1
                        mmr += 1

                    if counter != 1 and key == file:
                        mmr += 1/counter

                    counter += 1
            except Exception as e:
                print('Error en archivo', file)
                print(e)

        et = time.time()

        print("Model", model_name)
        print('Execution time:', round(et-st, 2), 'seconds')
        print("Precision P@1:", precision/n_docs)
        print("MMR:", mmr/n_docs)


if __name__ == "__main__":
    main()

# Leer tablas de datos e indexar el 90 % de su contenido e indexar

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import math
import pandas as pd
import csv
import numpy as np
import faiss
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


# Saca embeddings de cada columna
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


def read_files(path):
    return os.listdir(path)


def main():

    parser = argparse.ArgumentParser(description='Process Darta')
    parser.add_argument('-i', '--input', default='./data/', help='Name of the input folder storing CSV tables')
    parser.add_argument('-m', '--model', default='all',
                        choices=['all', 'uae-large', 'bge-large', 'bge-base', 'ember', 'gte-large', 'gte-base', 'stb'])
    parser.add_argument('-t', '--type', default='col',
                        choices=['col', 'tab']) # Si indexa las columnas o las promedia en un único embedding
    parser.add_argument('-r', '--result', default='./indexs',
                        help='Name of the output folder that stores the indexs files')

    args = parser.parse_args()

    files_path = args.input

    # select models
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

        files = read_files(files_path)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))  # create index of embeddings 768 dimensions
        id = 0
        map = pd.DataFrame()

        for file in tqdm(files):

            try:
                # Read dataframe
                delimiter = find_delimiter(files_path + file)
                df = pd.read_csv(files_path + file, sep=delimiter, nrows=120)

                if len(df.index) > 100:
                    df = df[:100]
                # Remove columns with all NaNs
                df = df.dropna(axis='columns', how='all')
                df.dropna(how='all', inplace=True)  # Remove rows where all elements are NaN

                # Split dataframe
                size = math.ceil(len(df.index) * 0.9)

                df = df[-size:]
                embs = content_embs(model, df, model_name, dimensions)

                faiss.normalize_L2(embs)

                if args.type == 'tab':
                    # Se promedian los embeddigs de cada tabla
                    embs = np.mean(embs, axis=0, dtype=np.float32)
                    index.add_with_ids(np.array([embs]), np.array([id]))

                    new_row = {"id": id, "dataset": file}
                    map = pd.concat([map, pd.DataFrame([new_row])], ignore_index=True)
                    id += 1
                else:
                    index.add_with_ids(embs, np.array(range(id, id+len(embs))))

                    for i in range(id, id+len(embs)):
                        new_row = {"id": i, "dataset": file}
                        map = pd.concat([map, pd.DataFrame([new_row])], ignore_index=True)

                    id += len(embs)
            except Exception as e:
                print('Error en archivo', file)
                print(e)

        # Save index and ids datasets
        if args.type == 'tab':
            faiss.write_index(index, "./indexs/tab_"+model_name+".index")
            map.to_csv("./indexs/tab_"+model_name+"_map.csv", index=False)
        else:
            faiss.write_index(index, "./indexs/"+model_name+".index")
            map.to_csv("./indexs/"+model_name+"_map.csv", index=False)


if __name__ == "__main__":
    main()

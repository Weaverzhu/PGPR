from __future__ import absolute_import, division, print_function

import os
import pickle
import gzip
import argparse

from utils import *
from data_utils import AmazonDataset
from knowledge_graph import KnowledgeGraph

source_file_path = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(source_file_path, "../config/pgpr.yaml")
with open(config_file_path, "rb") as f:
    config = yaml.load(f, yaml.SafeLoader)


def generate_labels(dataset, mode="train"):
    review_file = "{}/{}.txt.gz".format(DATASET_DIR[dataset], mode)
    user_products = {}  # {uid: [pid,...], ...}
    with gzip.open(review_file, "r") as f:
        for line in f:
            line = line.decode("utf-8").strip()
            arr = line.split("\t")
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    save_labels(dataset, user_products, mode=mode)


def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.dataset = config["dataset"]

    # Create AmazonDataset instance for dataset.
    # ========== BEGIN ========== #
    print("Load", args.dataset, "dataset from file...")
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = AmazonDataset(DATASET_DIR[args.dataset])
    save_dataset(args.dataset, dataset)

    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print("Create", args.dataset, "knowledge graph from dataset...")
    dataset = load_dataset(args.dataset)
    kg = KnowledgeGraph(dataset)
    kg.compute_degrees()
    save_kg(args.dataset, kg)
    # =========== END =========== #

    # Genereate train/test labels.
    # ========== BEGIN ========== #
    print("Generate", args.dataset, "train/test labels.")
    generate_labels(args.dataset, "train")
    generate_labels(args.dataset, "test")
    # =========== END =========== #


if __name__ == "__main__":
    main()

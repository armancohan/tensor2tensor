""" convert tfrecord to json"""

import argparse
import os
import json
import tensorflow.compat.v1 as tf
import multiprocessing as mp
from tqdm.auto import tqdm
from tensor2tensor.data_generators.wikisum import wikisum

def tfdataset_to_json(tfdataset):
    data = []
    for example in tf.python_io.tf_record_iterator(tfdataset):
        ex = tf.train.Example.FromString(example)
        parsed = {}
        for k, v in ex.features.feature.items():
            parsed[k] = [e.decode() for e in  v.bytes_list.value]
        data.append(parsed)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', help='')
    parser.add_argument('--out_dir', help='')
    parser.add_argument('--for_commoncrawl', help='', default=False, action='store_true')
    args = parser.parse_args()
    if args.for_commoncrawl:
        problem = wikisum.WikisumCommoncrawl()
    else:
        problem = wikisum.WikisumWeb()
    prefix = problem.dataset_filename()
    data_files = tf.gfile.Glob(os.path.join(args.in_dir, "%s*" % prefix))
    for tfdataset_file in tqdm(data_files, desc='processing shards'):
        json_list = tfdataset_to_json(tfdataset_file)
        file_name = os.path.split(tfdataset_file)[1]
        with tf.gfile.Open(
            os.path.join(args.out_dir, file_name), "w") as f:
            for json_obj in json_list:
                f.write(json.dumps(json_obj))
                f.write('\n')


if __name__ == '__main__':
    main()

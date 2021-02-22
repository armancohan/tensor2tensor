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


def process_file(function_args):
    tfdataset_file = function_args['file']
    args = function_args['args']
    json_list = tfdataset_to_json(tfdataset_file)
    file_name = os.path.split(tfdataset_file)[1]
    with tf.gfile.Open(
        os.path.join(args.out_dir, file_name), "w") as f:
        for json_obj in json_list:
            f.write(json.dumps(json_obj))
            f.write('\n')
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', help='')
    parser.add_argument('--out_dir', help='')
    parser.add_argument('--for_commoncrawl', help='', default=False, action='store_true')
    parser.add_argument('--shard', help='if passed it will only process certain shard (file ids in that range)', default=None)
    parser.add_argument('--total_shards', help='if passed it will only process certain shard (total files in the shard)', default=None)
    parser.add_argument('--workers', default=1, type=int)
    args = parser.parse_args()

    if args.for_commoncrawl:
        problem = wikisum.WikisumCommoncrawl()
    else:
        problem = wikisum.WikisumWeb()

    prefix = problem.dataset_filename()
    data_files = sorted(tf.gfile.Glob(os.path.join(args.in_dir, "%s*" % prefix)))

    if args.shard is not None and args.total_shards is not None:
        total_num_files = len(data_files)
        shard_len = total_num_files // args.total_shards
        start_offset = args.shard * shard_len
        end_offset = (args.shard + 1) * shard_len
        current_files = data_files[start_offset: end_offset]
    else:
        current_files = data_files
        
    if args.workers > 1:
        function_args = [{'file': e, 'args': args} for e in current_files]
        with mp.Pool(mp.cpu_count()) as p:
            res = list(tqdm(p.imap(process_file, function_args), total=len(function_args)))  
    else:
        for tfdataset_file in tqdm(current_files, desc='processing shards'):
            process_file({'file': tfdataset_file, 'args': args})
        
    print(done)


if __name__ == '__main__':
    main()

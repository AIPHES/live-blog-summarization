import rouge_papier

import argparse
import pathlib

from multiprocessing import Pool, cpu_count
import spacy
import re
import ujson as json
import os
import json
import codecs

def get_article_text(xml):
    return "\n\n".join([p.get_text() for p in xml.find_all("p")])

def init_worker():
    global nlp
    nlp = spacy.load('en', parser=False)

def load_liveblog(data_path, corpus):
    """Load the live blogs corpus data

    Args:
        data_path: path to the json file

    return:
        doc_data: list of input documents represented as a list of sentences.
        summaries: list of summaries represented as a list of sentences. 

    """
    doc_data = []
    post_ids = []
    post_times = []
    with codecs.open(data_path, "r", encoding='utf-8') as fp:
        json_text = fp.readline()
        json_data = json.loads(json_text)
        item_id = json_data['blog_id']
        genre = json_data['genre']
        url = json_data['url']
        title = json_data['title']

        # Return if the summary is empty or the collection is of low quality
        if not json_data['summary'] or json_data['quality'] == 'low':
            return doc_data, "", doc_id, genre

        summary = json_data['summary']

        documents = json_data['documents']

        for doc in documents:
            if 'is_key_event' in doc:
                if doc['is_key_event'] == False:
                    post_ids.append(doc['block_id'])
                    time = ""
                    if doc['time']:
                        time = "%s-%s-%s %s:%s" % (doc['time'][0], doc['time'][1], doc['time'][2], doc['time'][3], doc['time'][4])
                    post_times.append(time)
                    doc_data.append(" ".join(doc['text']))
            else:
                #print(type(doc['text']))
                try:
                    post_ids.append(doc['block_id'])
                    post_times.append(doc['time'])
                    doc_data.append(" ".join(doc['text']))
                except:
                    pass

        return review_data, "\n".join(summary), user_id, item_id
    
def prepare_example(article_text, abstract_text, user_id, item_id):
    global nlp
    inputs = []
    #print("Doc_id" , doc_id)
    ind = -1

    for post in nlp.pipe(article_text):
        ind += 1
        for sent in post.sents:
            tokens_all = [w for w in sent
                          if w.text.strip() != '']
            if len(tokens_all) <= 3:
                continue
            tokens = [w.text.strip().lower() for w in tokens_all]
            pos = [w.pos_ for w in tokens_all]
            ne = [w.ent_type_ for w in tokens_all]
            pretty_text = sent.text.strip()
            pretty_text = re.sub(r"\r|\n|\t", r" ", pretty_text)
            pretty_text = re.sub(r"\s+", r" ", pretty_text)
            inputs.append({"tokens": tokens, "text": pretty_text,
                           "pos": pos, "ne": ne, 
                           "word_count": len(pretty_text.split()),
                           "post_id": post_ids[ind],
                           "post_time": post_times[ind]})
            
    for i, inp in enumerate(inputs, 1):
        inp["sentence_id"] = i

    summary_texts = []
    if len(abstract_text) > 0:
        summary_texts.append(abstract_text)
        
    input_texts = [inp["text"] if inp["word_count"] > 2 else "@@@@@"
                   for inp in inputs[:50]]
    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, summary_texts, mode="sequential", ngram=1, 
        remove_stopwords=True, length=100)
    labels = [1 if r > 0 else 0 for r in ranks]
    if len(labels) < len(inputs):
        labels.extend([0] * (len(inputs) - len(labels)))
    labels = {"id": doc_id, "labels": labels}
    example = {"id": doc_id, "inputs": inputs, "genre": genre, "url": url, "title": title}
    return example, labels, abstract_text


def worker(args):
    file_path, inputs_dir, labels_dir, abs_dir, corpus = args

    # Process xml to get document and summary text. 
    article_text, abs_txt, doc_id, genre, post_ids, post_times, url, title = load_liveblog(file_path, corpus)
    
    if abs_txt == "" or article_text == None:
        return False
    
    example, labels, abstract_text = prepare_example(article_text, abs_txt, doc_id, genre, post_ids, post_times, url, title)
    
    if not example["inputs"]:
        return False
    
    assert abstract_text == abs_txt
    
    inputs_path = inputs_dir / "{}.json".format(example["id"])
    inputs_path.write_text(json.dumps(example))
    labels_path = labels_dir / "{}.json".format(example["id"])
    labels_path.write_text(json.dumps(labels))

    if len(abs_txt) > 0:
        abs_path1 = abs_dir / "{}.1.txt".format(example["id"])
        abs_path1.write_text(abs_txt)

    return True


def preprocess_part(data_path, file_paths, inputs_dir, labels_dir, abs_dir, corpus, procs=16):

    inputs_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)
    abs_dir.mkdir(exist_ok=True, parents=True)

    def data_iter():
        for file_path in file_paths:
            yield data_path + "/" + file_path, inputs_dir, labels_dir, abs_dir, corpus
    pool = Pool(procs, initializer=init_worker)
    count = 0
    for i, is_good in enumerate(pool.imap(worker, data_iter()), 1):
        if is_good:
            count += 1
            print("{}".format(count), end="\r", flush=True)
    print()

def get_paths(path):
    print(path)
    train_paths = [ f for f in os.listdir(path + '/train') if os.path.isfile(path + '/train/' + f)]
    valid_paths = [ f for f in os.listdir(path + '/valid') if os.path.isfile(path + '/valid/' + f)]
    test_paths = [ f for f in os.listdir(path + '/test') if os.path.isfile(path + '/test/' + f)]
    return train_paths, valid_paths, test_paths
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liveblogs", type=pathlib.Path, required=False, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/raw/liveblogs"))
    parser.add_argument("--data-dir", type=pathlib.Path, required=True,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/liveblogs"))
    parser.add_argument("--procs", type=int, required=False, default=None)
    args = parser.parse_args()

    if args.procs is None:
        args.procs = min(cpu_count(), 16)

    for corpus in ['guardian', 'bbc']:
        data_path = str(args.liveblogs.joinpath(corpus))
        paths = [ file for file in os.listdir(data_path) if os.path.isfile(data_path + "/" + file)]

        num_topics = len(paths)
        
        train_size = int((8/10) * num_topics)
        valid_size = int((1/10) * num_topics)
        
        train_paths = paths[:train_size]
        valid_paths = paths[train_size:-valid_size]
        test_paths = paths[-valid_size:]
        
        print("Train Size:", train_size)
        print("Valid Size:", valid_size)
        print("Test Size", num_topics-train_size-valid_size)
               
        preprocess_part(
            data_path,
            valid_paths, 
            args.data_dir / "liveblogs" / corpus / "inputs" / "valid",
            args.data_dir / "liveblogs" / corpus / "labels" / "valid",
            args.data_dir / "liveblogs" / corpus / "human-abstracts" / "valid",
            corpus,
            procs=args.procs)

        preprocess_part( 
            data_path,
            test_paths, 
            args.data_dir / "liveblogs" / corpus / "inputs" / "test",
            args.data_dir / "liveblogs" / corpus / "labels" / "test",
            args.data_dir / "liveblogs" / corpus / "human-abstracts" / "test",
            corpus,
            procs=args.procs)

        preprocess_part(
            data_path,
            train_paths, 
            args.data_dir / "liveblogs" / corpus / "inputs" / "train",
            args.data_dir / "liveblogs" / corpus / "labels" / "train",
            args.data_dir / "liveblogs" / corpus / "human-abstracts" / "train",
            corpus,
            procs=args.procs)

if __name__ == "__main__":
    main()

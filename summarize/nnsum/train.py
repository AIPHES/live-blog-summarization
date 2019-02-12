import argparse
import os, sys
from utils.misc import mkdirp
from subprocess import check_output

# python train.py --batch_size 16 --gpu_num 0

def train(args):
    corpora = ['bbc', 'guardian']
    models = ['s2s', 'rnn', 'cl', 'sr']
    encoder_types= ["avg", "rnn", "cnn"]
    summary_sizes = [50, 100]
    sent_limit = "500"
    emb_path = "~/workspace/embeddings/glove.6B.200d.txt"
    
    for summary_length in summary_sizes:
        for corpus in corpora:
            models_path = "%s/%s/%s/" % (args.models_path, corpus, summary_length)
            if not os.path.isdir(models_path):
                mkdirp(models_path)
            results_dir = "%s/%s/%s/valid" % (args.results_path, corpus, summary_length)
            if not os.path.isdir(results_dir):
                mkdirp(results_dir)

            for model in models:
                for encoder in encoder_types:
                    model_path = "%s/%s_%s_%s" % (models_path, corpus, model, encoder)
                    result_path = "%s/%s_%s_%s" % (results_dir, corpus, model, encoder)
                    # Arguments for evaluation
                    train_inputs = "--train-inputs %s/%s/inputs/train" % (args.data_dir, corpus)
                    train_labels = "--train-labels %s/%s/labels/train" % (args.data_dir, corpus)

                    valid_inputs = "--valid-inputs %s/%s/inputs/valid" % (args.data_dir, corpus)
                    valid_labels = "--valid-labels %s/%s/labels/valid" % (args.data_dir, corpus)
                    valid_refs = "--valid-refs %s/%s/human-abstracts/valid" % (args.data_dir, corpus)

                    arg_summary_length = "--summary-length %s" % (summary_length)
                    args_model = "--model %s" % (model_path)
                    args_result = "--results %s.json" % (result_path)

                    trainer_args = "--trainer --sentence-limit %s %s %s %s %s %s --weighted --gpu %s %s %s %s --seed 12345678 --batch-size %s" % (sent_limit, train_inputs, train_labels, valid_inputs, valid_labels, valid_refs, args.gpu_num, args_model, args_result, arg_summary_length, args.batch_size)

                    emb_args = "--emb --pretrained-embeddings %s --at-least 1 --filter-pretrained" % (emb_path)
                    enc_args = "--enc %s --ext %s " % (encoder, model)


                    if model in ['s2s', 'rnn']:
                        command = "python script_bin/train_model.py %s %s %s --bidirectional" % (trainer_args, emb_args, enc_args)
                    if model in ['cl', 'sr']:
                        command = "python script_bin/train_model.py %s %s %s" % (trainer_args, emb_args, enc_args)

                    print(command)
                    try:
                        check_output(command, shell=True)
                    except:
                        continue



def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Evaluate the Neural Summarization Models')

    #parser.add_argument('--summary_length', type=str, help='Summary Length ex:100', required=True)

    parser.add_argument('--batch_size', type=str, help='Batch Size ex:32', default="16", required=True)

    parser.add_argument('--gpu_num', type=str, help='GPU number ex:0,1', required=True)

    parser.add_argument('--data_dir', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/liveblogs"))

    parser.add_argument('--models_path', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/liveblogs"))

    parser.add_argument('--results_path', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/liveblogs"))


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    train(args)


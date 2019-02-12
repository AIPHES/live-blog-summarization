import argparse
import os
from utils.misc import mkdirp
from subprocess import check_output

def evaluate(args):
    corpora = ['bbc', 'guardian']
    summary_sizes = [50, 100]
    for summary_length in summary_sizes:
        for corpus in corpora:
            models_path = "%s/%s/%s/" % (args.models_path, corpus, summary_length)
            print(models_path)
            models = os.listdir(models_path)
            results_dir = "%s/%s/%s/test" % (args.results_path, corpus, summary_length)
            if not os.path.isdir(results_dir):
                mkdirp(results_dir)

            for model in models:
                model_path = "/".join([models_path, model])
                if os.path.isfile(model_path):
                    # Arguments for evaluation
                    inputs = "--inputs %s/%s/inputs/test" % (args.data_dir, corpus)
                    refs = "--refs %s/%s/human-abstracts/test" % (args.data_dir, corpus)
                    arg_summary_length = "--summary-length %s" % (summary_length)
                    args_model = "--model %s" % (model_path)
                    args_result = "--results %s/%s.json" % (results_dir, model[:-4])
                    command = "python script_bin/eval_model.py %s %s %s %s %s --gpu 0 --batch-size 4" % (inputs, refs, args_model, args_result, arg_summary_length)
                    print(command)
                    try:
                        output = check_output(command, shell=True)
                    except:
                        continue
                    #exit(0)


def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Evaluate the Neural Summarization Models')

    #parser.add_argument('--summary_length', type=str, help='Summary Length ex:100', required=True)

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
    evaluate(args)


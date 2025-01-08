from dependency_parser_canevas_salome import read_treebank, compute_evaluation, HyperParameters
import yaml
from copy import deepcopy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test script the evaluation function')
    parser.add_argument('config',nargs ='?',default=r'C:\Users\nikyt\Documents\IDL M2\Syntaxe et Sémantique\TD5_projet_parsing-20241016\config.yaml', help='Path to yaml config file')
    args = parser.parse_args()
    
    with open(args.config) as instream:
        hp = HyperParameters(yaml.load(instream, Loader=yaml.SafeLoader))

    # treebank = list of DependencyTree objects
    treebank = read_treebank(hp.treebank)

    treebank = [t for t in treebank if len(t) == 10][:3]

    pred_treebank = deepcopy(treebank)
    
    pred_treebank[1].pos_tags[3] = "$"
    pred_treebank[2].pos_tags[2] = "$"
    pred_treebank[2].pos_tags[4] = "$"
    
    pred_treebank[1].dep_labels[3] = "$"
    pred_treebank[1].dep_labels[4] = "$"
    
    pred_treebank[2].heads[6] = -1
    pred_treebank[2].heads[9] = -1
    pred_treebank[2].dep_labels[7] = "$"
    
    
    
    print("Evaluation:", compute_evaluation(treebank[:1], pred_treebank[:1]))
    print("Expected  :", "{'pos': 100.0, 'uas': 100.0, 'las': 100.0}")
    print()
    print("Evaluation:", compute_evaluation(treebank[1:2], pred_treebank[1:2]))
    print("Expected  :", "{'pos': 90.0, 'uas': 100.0, 'las': 80.0}")
    print()
    print("Evaluation:", compute_evaluation(treebank[2:3], pred_treebank[2:3]))
    print("Expected  :", "{'pos': 80.0, 'uas': 80.0, 'las': 70.0}")
    print()

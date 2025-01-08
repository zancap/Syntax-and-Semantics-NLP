from dependency_parser_canevas_salome import static_oracle_eager, ArcEager, compute_evaluation, read_treebank, HyperParameters
import yaml


def test_oracle(tree):

    actions = static_oracle_eager(tree)
    state = ArcEager(tree.get_tokens())
    for action in actions:
        state.do_action(*action)
    assert(state.is_final())
    pred_tree = state.get_tree()
    
    eval_dict = compute_evaluation([tree], [pred_tree])
    if eval_dict["las"] < 100:
        if tree.is_projective():
            print("Oracle failed")
            print("Gold tree")
            print(tree)
            print()
            print("Oracle tree")
            print(pred_tree)
            print()
            print(actions)
            print()
            return
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test script for the oracle function')
    # parser.add_argument('config', help='Path to yaml config file')
    parser.add_argument('config',nargs ='?',default=r'C:\Users\nikyt\Documents\IDL M2\Syntaxe et Sémantique\TD5_projet_parsing-20241016\config.yaml', help='Path to yaml config file')
    args = parser.parse_args()
    
    with open(args.config) as instream:
        hp = HyperParameters(yaml.load(instream, Loader=yaml.SafeLoader))

    # treebank = list of DependencyTree objects
    treebank = read_treebank(hp.treebank)
    
    projective_trees = [t for t in treebank if t.is_projective()]

    for tree in projective_trees:
        test_oracle(tree)

    print("All test passed: Oracle is correct !")

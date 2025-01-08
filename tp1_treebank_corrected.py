# -*- coding: utf-8 -*-
"""
Date rendu: 23 Sept
Cours: Syntaxe/Sémantique -- M2 SDL IDL
Autrice: Paola Zancanaro
Nom du fichier: TP1_PaolaZancanaro_TreeBank.py
"""

import nltk
from nltk import Tree

## Question 1:
def read_constituency_treebank(filename):
    """Renvoie une liste d’objets de type nltk.Tree"""
    trees = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            _, tree_string = line.strip().split("\t", 1)
            tree_string = tree_string[2:-1].strip()
            tree = Tree.fromstring(tree_string)
            trees.append(tree)
    return trees

## Question 2:
def depth(tree):
    """Renvoie un entier correspondant à la profondeur de l'arbre"""
    if isinstance(tree, str):
        return 0
    else:
        return 1 + max(depth(child) for child in tree)

## Question 3
def count_nodes(tree):
    """Renvoie le nombre de noeuds de l'arbre sans compter les tokens eux-mêmes"""
    if len(tree) == 1 and isinstance(tree[0], str):
        return 1
    else:
        return 1 + sum(count_nodes(child) for child in tree)

## Question 4.1
def stats_POS(corpus):
    """Renvoie un dictionnaire de fréquences des POS tags dans le corpus"""
    stats = {}
    for tree in corpus:
        update_pos_dict(tree, stats)
    return stats

def update_pos_dict(tree, stats):
    """Met à jour le dictionnaire stats {str: int} pour cet arbre, uniquement pour les étiquettes de terminales (POS tags)"""
    if isinstance(tree, Tree) and len(tree) == 1 and isinstance(tree[0], str):
        pos_tag = tree.label()
        stats[pos_tag] = stats.get(pos_tag, 0) + 1
    else:
        for child in tree:
            update_pos_dict(child, stats)

## Question 4.2
def stats_syntagmes(corpus):
    """Renvoie un dictionnaire de fréquences des syntagmes dans le corpus, excluant les POS tags"""
    stats = {}
    for tree in corpus:
        update_syntagmes_dict(tree, stats)
    return stats

def update_syntagmes_dict(tree, stats):
    """Met à jour le dictionnaire stats {str: int} pour les syntagmes dans cet arbre, excluant les POS tags"""
    syntagme = tree.label()
    stats[syntagme] = stats.get(syntagme, 0) + 1

    for child in tree:
        if isinstance(child, Tree) and not isinstance(child[0], str):
            update_syntagmes_dict(child, stats)

## Question 5: version simple
def tree_to_string(tree):
    """Renvoie une string qui représente l'arbre au format parenthésé (version simple)"""
    if isinstance(tree, str):
        return tree
    else:
        return f"({tree.label()} {' '.join(tree_to_string(child) for child in tree)})"

## Question 5: version efficace
def tree_to_string_efficient(tree):
    """Renvoie une string qui représente l'arbre au format parenthésé (version efficace)"""
    result = []

    def helper(subtree):
        if isinstance(subtree, str):
            result.append(subtree)
        else:
            result.append(f"({subtree.label()}")
            for child in subtree:
                ##@@@@@ ligne ajoutée (pour résoudre le bug des espaces)
                result.append(" ") #@@@@
                helper(child)
            result.append(")")

    helper(tree)
    return ''.join(result)

## Question 6:
def stats_NP(corpus):
    stats = {}
    for tree in corpus:
        update_NP_dict(tree, stats)
    return stats

def update_NP_dict(tree, stats):
    """Met à jour le dictionnaire stats {str: int} pour les syntagmes NP dans cet arbre"""
    if tree.label() == "NP":
        child_labels = tuple(child.label() if isinstance(child, Tree) else type(child).__name__ for child in tree)

        stats[child_labels] = stats.get(child_labels, 0) + 1

    for child in tree:
        if isinstance(child, Tree):
            update_NP_dict(child, stats)

## Enlève les fonctions syntaxiques sur les étiquettes des syntagmes
def remove_functions_on_labels(tree):
    if type(tree) == str:
        return
    tree.set_label(tree.label().split("-")[0])
    for child in tree:
        remove_functions_on_labels(child)

def test_q1(corpus):
    ## Question 1: test
    print()
    print("premier & dernier arbres:")
    print()
    print(corpus[0])
    print(corpus[-1])
    print()

    print()
    print("Résultats attendus:")
    print()
    tree_first = "(SENT (NP (NPP Gutenberg)))"
    tree_last =  "(SENT (NP (NC Affaire) (AP (PREF politico-) (ADJ financière))))"
    print(tree_first)
    print(tree_last)
    print()
    if tree_first == str(corpus[0]) and  tree_last == str(corpus[-1]):
        print("Q1: test réussi")
    else:
        print("Q1: échec :(")

    return

def test_q2(corpus):
    ## Question 2: test
    print()
    print("Test Q2")
    depth_first = depth(corpus[0])
    depth_last = depth(corpus[-1])
    print(f"depth {corpus[0]} = {depth_first}")
    print(f"correct = {depth_first == 3}")
    print(f"depth {corpus[-1]} = {depth_last}")
    print(f"correct = {depth_last == 4}")

    depths = []
    for tree in corpus:
        depths.append(depth(tree))

    avg = sum(depths)/len(depths)
    maximum = max(depths)
    minimum = min(depths)

    print(f"Average depth: {avg} \trésultat attendu: 7.8, correct = {abs(avg - 7.833) < 0.001}")
    print(f"min depth: {minimum} \t\t\t\trésultat attendu: 2,  correct = {minimum == 2}")
    print(f"max depth: {maximum} \t\t\t\trésultat attendu: 26, correct = {maximum == 26}")

def test_q3(corpus):
    ## Question 3: test
    print()
    print("Test Q3")
    n_first = count_nodes(corpus[0])
    n_last = count_nodes(corpus[-1])
    print(f"number of nodes for {corpus[0]} = {n_first}")
    print(f"correct = {n_first == 3}")
    print(f"number of nodes for {corpus[-1]} = {n_last}")
    print(f"correct = {n_last == 6}")
    print()

    nums = []
    for tree in corpus:
        nums.append(count_nodes(tree))

    avg = sum(nums)/len(nums)
    maximum = max(nums)
    minimum = min(nums)

    print(f"Average number of nodes: {avg} \trésultat attendu: 36.987\tcorrect = {abs(avg - 36.987) < 0.001}")
    print(f"min number of nodes: {minimum} \t\t\t\trésultat attendu: 2\t\tcorrect = {minimum == 2}")
    print(f"max number of nodes: {maximum} \t\t\trésultat attendu: 220\t\tcorrect = {maximum == 220}")

def test_q4(corpus):
    ## Question 4: test
    print("Stats POS")
    expected_freqs = {'NPP': 3542, 'DET': 9189, 'NC': 14997, 'CLO': 278, 'V': 3746, 'CS': 705, 'P': 8806, 'ADJ': 4661, 'PONCT': 7873, 'ADV': 2506, 'P+D': 1830, 'VPP': 2506, 'CC': 1656, 'PROREL': 587, 'VINF': 1403, 'CLS': 1157, 'VPR': 369, 'ADVWH': 38, 'CLR': 340, 'PREF': 76, 'PRO': 425, 'VS': 88, 'I': 3, 'PROWH': 20, 'P+PRO': 7, 'ADJWH': 17, 'ET': 171, 'VIMP': 31, 'DETWH': 11}

    pos_freqs = stats_POS(corpus)
    print(f"test passed: {pos_freqs==expected_freqs}")
    print()

    print("Fréquences POS par ordre décroissant")
    for k, v in sorted(pos_freqs.items(), key = lambda x: x[1], reverse=True):
        print(k, v)
    print()

    ## Question 4: test syntagmes
    print("Stats syntagmes")
    expected_phrases_freqs = {'SENT': 3099, 'NP': 17975, 'VN': 5603, 'Ssub': 725, 'Sint': 951, 'PP': 10589, 'AP': 3447, 'COORD': 1945, 'VPpart': 1165, 'Srel': 593, 'VPinf': 1365, 'AdP': 129}
    phrase_freqs = stats_syntagmes(corpus)
    print(f"test passed: {phrase_freqs==expected_phrases_freqs}")
    print()

    if phrase_freqs != expected_phrases_freqs:
        print("Les résultats sont differents !")
        print("Résultats attendus:", expected_phrases_freqs)
        print("Résultats reçu:", phrase_freqs)

    print("Fréquences syntagmes par ordre décroissant")
    for k, v in sorted(phrase_freqs.items(), key = lambda x: x[1], reverse=True):
        print(k, v)
    print()

def test_q5():
    ## Question 5: test
    ptree = "(S (NP (D le) (N chat)) (VP (V dort)))"
    tree = Tree.fromstring(ptree)
    print(ptree)
    print(tree_to_string(tree))
    print(tree_to_string_efficient(tree))

def main():
    corpus = read_constituency_treebank("sequoia.surf.const")
    for tree in corpus:
        remove_functions_on_labels(tree)

    test_q1(corpus)

    test_q2(corpus)

    test_q3(corpus)

    test_q4(corpus)

    test_q5()


if __name__ == "__main__":
    main()

import random
from collections import defaultdict
import torch
from torch import nn
from torch.optim import AdamW
import yaml

class HyperParameters(object):
    """"Namespace to store hyperparameters"""
    def __init__(self, adict):
        self.__dict__.update(adict)

class DependencyTree():
    
    def __init__(self, tokens, pos_tags=None, heads=None, dep_labels=None, sentence_text=None):
        """
        Only obligatory argument: tokens.
        pos_tags, heads, dep_labels are defaulted to "_" if not provided.
        sentence_text: the full sentence as a string.
        """
        self.tokens = tokens
        self.pos_tags = pos_tags if pos_tags is not None else ["_" for _ in tokens]
        self.heads = heads if heads is not None else ["_" for _ in tokens]
        self.dep_labels = dep_labels if dep_labels is not None else ["_" for _ in tokens]
        self.sentence_text = sentence_text  # on récupère la phrase analysée pour notre fichier Conll externe
        self.dependents = [set() for _ in range(len(self.tokens)+1)]
        for i in range(len(self.heads)):
            if self.heads[i] is not None:
                self.dependents[self.heads[i]].add(i+1)
        self.n = len(self.tokens)

    def __len__(self):
        """Returns the number of tokens in the tree"""
        return len(self.tokens)
    
    def get_token(self, i):
        """Returns token number i (where 0 is ROOT and 1 is for the first token"""
        if i == 0:
            return "ROOT"
        assert(i - 1 < len(self.tokens) and i >= 1)
        return self.tokens[i-1]
    
    def get_tokens(self):
        """Returns a copy of the list of tokens in the sentence"""
        return [t for t in self.tokens]

    def set_pos_tags(self, tags):
        """Assigns a POS tag list (`tags`) to the tree. The tag list should have the same length as the tree"""
        assert(len(tags) == len(self))
        self.pos_tags = tags
    
    def get_pos_tags(self):
        """Returns the list of POS tags"""
        return self.pos_tags

    def get_head(self, i):
        """Returns the head of token 'i'"""
        if i == 0:
            return "ROOT"
        assert(i - 1 < len(self.heads) and i >= 1)
        return self.heads[i-1]

    def get_heads(self):
        """Returns the list of heads"""
        return self.heads

    def get_dep(self, i):
        """Returns the dependency label of token 'i'"""
        assert(i - 1 < len(self.dep_labels) and i >= 1)
        return self.dep_labels[i-1]

    def get_deps(self):
        """Returns the list of dependency labels"""
        return self.dep_labels

    def __str__(self):
        """
        Sortir une répresentation texteulle
        des dependency tree dans le format CoNLL (or CoNLL-U).
        """
        lines = []

        # On ajoute la phrase qui est analysée
        if self.sentence_text:
            lines.append(f"# sentence-text: {self.sentence_text}")

        # Boucle des tokens de 1 à n (en sachant 0 = ROOT) afin de récupérer les informations nécessaires
        for i in range(1, self.n + 1):
            form = self.tokens[i - 1]  
            lemma = "_"  
            upos = self.pos_tags[i - 1] if self.pos_tags else "_"  
            xpos = "_" 
            feats = "_"  
            head = self.heads[i - 1] if self.heads[i - 1] != "_" else "0"  
            deprel = self.dep_labels[i - 1] if self.dep_labels[i - 1] != "None" else "_"  
            deps = "_"  
            misc = "_"  

            # Création et formattage de chaque ligne
            line = f"{i}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}"
            lines.append(line)

        return "\n".join(lines) + "\n"

    def is_projective(self):
        """Returns True if the tree is projective and False if it isn't"""
        # copy the dependents stuff
        projections = [{i for i in s} for s in self.dependents]
        
        for i in range(len(projections)):
            projections[i].add(i)

        while True:
            new_projections = [{i for i in s} for s in projections]
            for i in range(len(projections)):
                for j in projections[i]:
                    new_projections[i].update(projections[j])
            if new_projections == projections:
                break
            projections = new_projections
        
        for s in projections:
            span = sorted(s)
            if span != list(range(span[0], span[-1]+1)):
                return False
        return True


class ArcEager():
    
    def __init__(self, tokens):
        self.tokens = tokens
        length = len(tokens)
        # le buffer contient les tokens dans l'ordre inverse de la phrase
        # le premier token est le dernier de la liste!
        # cela permet d'utiliser le buffer comme une pile (.pop())
        self.buffer = list(reversed(range(1, length+1)))
        self.stack = [0]
        self.arcs = [None] * (length + 1)
        self.labels = [None] * (length + 1)
    
    def left_arc(self, label):
        """
        Applique l'action left-arc avec le label donné en argument
        """
        # Le sommet de la pile (stack) devient dépendant du sommet du buffer
        # On retire le sommet de la pile (stack)
        dependent = self.stack.pop()
        # On récupère le sommet du buffer
        head = self.buffer[-1]
        # On définit le sommet du buffer comme le "head" du dépendant
        self.arcs[dependent] = head
        # On associe l'étiquette de dépendance au lien
        self.labels[dependent] = label

    def right_arc(self, label):
        """
        Applique l'action right-arc avec le label donné en argument
        """
        # On récupère le sommet de la pile (stack)
        head = self.stack[-1]
        # On retire le sommet du buffer
        dependent = self.buffer.pop()
        # On ajoute le sommet du buffer dans la pile
        self.stack.append(dependent)
        # On définit le sommet de la pile comme le "head" du dépendant
        self.arcs[dependent] = head
        # On associe l'étiquette de dépendance au lien
        self.labels[dependent] = label

    def reduce(self):
        """
        Applique l'action reduce
        """
        # On retire le sommet de la pile
        self.stack.pop()

    def shift(self):
        """
        Applique l'action shift
        """
        # On retire le sommet du buffer pour l'ajouter dans la pile
        self.stack.append(self.buffer.pop())

    def do_action(self, action, label):
        match action:
            case "SH":
                self.shift()
            case "LA":
                self.left_arc(label)
            case "RA":
                self.right_arc(label)
            case "RE":
                self.reduce()

    def can_shift(self):
        return len(self.buffer) > 0
    
    def can_left_arc(self):
        return len(self.stack) > 0 and len(self.buffer) > 0 and self.stack[-1] != 0 and self.arcs[self.stack[-1]] == None
    
    def can_right_arc(self):
        return len(self.stack) > 0 and len(self.buffer) > 0 and self.arcs[self.buffer[-1]] == None

    def can_reduce(self):
        return len(self.stack) > 0 and self.arcs[self.stack[-1]] is not None

    def is_final(self):
        return self.buffer == []

    def get_buffer_top(self):
        return self.buffer[-1]
    
    def get_stack_top(self):
        return self.stack[-1]

    def get_tree(self):
        assert(self.is_final())
        return DependencyTree(self.tokens, heads=self.arcs[1:], dep_labels=self.labels[1:])

    def get_features(self):
        # https://aclanthology.org/P16-2006.pdf
        """Une Valeur None signifie que cet élément n'existe pas (ex: le buffer est vide)"""
        i = self.stack[-1] if len(self.stack) > 0 else None
        j1 = self.buffer[-1] if len(self.buffer) > 0 else None
        j2 = self.buffer[-1] if len(self.buffer) > 1 else None
        return i, j1, j2

    def get_repr(self):
        return ["0" if i == 0 else self.tokens[i-1] for i in self.stack], [0 if i == 0 else self.tokens[i-1] for i in reversed(self.buffer)]

class BiLstmDependencyParser(nn.Module):
    
    def __init__(self, vocabulary, hp):
        super(BiLstmDependencyParser, self).__init__()
        # Vocabulaires
        self.i2pos = vocabulary["pos"]
        self.pos2i = {pos: i for i, pos in enumerate(self.i2pos)}
        self.i2word = ["UNK"] + vocabulary["words"]
        self.word2i = {word: i for i, word in enumerate(self.i2word)}
        self.i2dep = vocabulary["dep"]
        self.dep2i = {dep: i for i, dep in enumerate(self.i2dep)}
    
        self.i2actions = vocabulary["actions"]
        self.actions2i = {action: i for i, action in enumerate(self.i2actions)}
        
        self.SH_ID = self.actions2i[("SH", None)]
        self.LA_ID = [i for i, action in enumerate(self.i2actions) if action[0] == "LA"]
        self.LA_ID = [self.LA_ID[0], self.LA_ID[-1]]
        self.RA_ID = [i for i, action in enumerate(self.i2actions) if action[0] == "RA"]
        self.RA_ID = [self.RA_ID[0], self.RA_ID[-1]]
        self.RE_ID = self.actions2i[("RE", None)]
        # réseau de neurones
        
        self.word_embeddings = nn.Embedding(len(self.i2word), hp.dim_word_embeddings)
        self.sentence_lstm = nn.LSTM(input_size=hp.dim_word_embeddings,
                                     hidden_size=hp.dim_lstm // 2,
                                     num_layers=1,
                                     dropout=hp.dropout_lstm,
                                     batch_first=True,
                                     bidirectional=True)

        self.pos_tagger = nn.Linear(hp.dim_lstm, len(self.pos2i))
        self.action_tagger = nn.Linear(hp.dim_lstm * hp.n_features, len(self.actions2i))
        
        self.loss_fn = nn.CrossEntropyLoss()

        # 4 vectors:
        # 0: representation for the ROOT
        # 1: feature for 1st element of stack when stack is empty
        # 2: feature for 1st element of buffer when buffer is empty
        # 3: feature for 2nd element of buffer when buffer has only one (or is empty)
        self.default_values = nn.Parameter(torch.Tensor(4, hp.dim_lstm))
        torch.nn.init.uniform_(self.default_values, -0.01, 0.01)

    def filter_action(self, state, vector, minf=-10**10):
        """
        This function makes sure that impossible actions have a very low score
        """
        if not state.can_shift():
            vector[self.SH_ID] = minf
        if not state.can_reduce():
            vector[self.RE_ID] = minf
        if not state.can_left_arc():
            vector[self.LA_ID[0]:self.LA_ID[1]+1] = minf

        if not state.can_right_arc():
            vector[self.RA_ID[0]:self.RA_ID[1]+1] = minf


    def forward(self, sentence, oracle=None, gold_pos_tags=None):
        # Si  actions == None: on est en mode évaluation (on réalise une prédiction)
        # Sinon, on est en mode entraînement (on calcule la loss)

        sentence_ids = [self.word2i[w] if w in self.word2i else 0 for w in sentence]
        sentence_ids = torch.tensor(sentence_ids, dtype=torch.long)
        if oracle is not None:
            mask = torch.FloatTensor(len(sentence_ids)).uniform_() > 0.1
            sentence_ids = sentence_ids * mask

        
        if gold_pos_tags is not None:
            gold_pos_tags_ids = torch.tensor([self.pos2i[t] for t in gold_pos_tags], dtype=torch.long)
        
        embeddings = self.word_embeddings(sentence_ids)
        
        contextual_embeddings, _ = self.sentence_lstm(embeddings)
        pos_output = self.pos_tagger(contextual_embeddings)

        pred_pos_str = torch.argmax(pos_output, dim=1)
        pred_pos_str = [self.i2pos[i] for i in pred_pos_str]
        
        if oracle is not None:
            oracle_id = torch.tensor([self.actions2i[a] for a in oracle], dtype=torch.long)
            all_classifier_outputs = []
        
        N = 0 # number of actions taken so far

        output_dict = {}

        pred_derivation = []
        state = ArcEager(sentence)
        while not state.is_final():
            i, j1, j2 = state.get_features()
            classifier_input = []
            if i is None:
                classifier_input.append(self.default_values[1])
            elif i == 0:
                classifier_input.append(self.default_values[0])
            else:
                classifier_input.append(contextual_embeddings[i-1])
            if j1 is None:
                classifier_input.append(self.default_values[2])
            else:
                classifier_input.append(contextual_embeddings[j1-1])
            if j2 is None:
                classifier_input.append(self.default_values[3])
            else:
                classifier_input.append(contextual_embeddings[j2-1])

        
            classifier_input = torch.cat(classifier_input)
            classifier_output = self.action_tagger(classifier_input)
            
            
            if oracle is None:
                self.filter_action(state, classifier_output)
                predicted_action = torch.argmax(classifier_output)
                action_str = self.i2actions[predicted_action]
                state.do_action(*action_str)
                N +=1
                pred_derivation.append(action_str)
            else:
                all_classifier_outputs.append(classifier_output)
                state.do_action(*oracle[N])
                N += 1
        
        if oracle is not None:
            assert(len(all_classifier_outputs) == len(oracle))
            all_classifier_outputs = torch.stack(all_classifier_outputs)
            output_dict["parsing_loss"] = self.loss_fn(all_classifier_outputs, oracle_id)

        if gold_pos_tags is not None:
            output_dict["tagging_loss"] = self.loss_fn(pos_output, gold_pos_tags_ids)


        tree = state.get_tree()
        tree.set_pos_tags(pred_pos_str)
        output_dict["tree"] = tree
        output_dict["derivation"] = pred_derivation
        #print(pred_derivation)
        
        return output_dict

def read_treebank(filename):
    """Reads a conll corpus and returns it as a list of DependencyTree objects."""
    with open(filename, encoding="utf8") as f:
        conll_trees = f.read().split("\n\n")
    
    treebank = []
    for conll_tree in conll_trees:
        if not conll_tree.strip():
            continue
        
        lines = conll_tree.strip().split("\n")
        sentence_text = None
        
        # On vérifie si la ligne comme par "# sentence:"
        if lines[0].startswith("# sentence-text:"):
            # Si oui on la met de côté pour notre fichier Conll externe
            sentence_text = lines.pop(0).split(": ", 1)[1]

        lines = [line.split("\t") for line in lines if line.strip() and line[0] != "#"]
        tokens = [line[1] for line in lines]
        pos_tags = [line[4] for line in lines]
        heads = [int(line[6]) for line in lines]
        dep_labels = [line[7] for line in lines]
        
        treebank.append(DependencyTree(tokens, pos_tags, heads, dep_labels, sentence_text))

    return treebank


def train_dev_test_split(treebank):
    """returns a 80/10/10 train/dev/test split"""
    random.shuffle(treebank)
    N = len(treebank) // 10
    return treebank[2*N:], treebank[N:2*N], treebank[:N]

def get_vocabulary(treebank):
    """Extract the vocabulary of pos tags, dep labels and words from the treebank"""
    pos_tags = defaultdict(int)
    dep_labels = defaultdict(int)
    words = defaultdict(int)

    actions = defaultdict(int)

    for tree in treebank:
        for token in tree.tokens:
            words[token] += 1
        for tag in tree.pos_tags:
            pos_tags[tag] += 1
        for dep_label in tree.dep_labels:
            dep_labels[dep_label] += 1
        
        derivation = static_oracle_eager(tree)
        for action in derivation:
            actions[action] += 1
    
    
    return {"pos": sorted(pos_tags), 
            "words": sorted(words), 
            "dep": sorted(dep_labels), 
            "actions": sorted(actions, reverse=True),
            "stats": (words, pos_tags, dep_labels, actions)}

def static_oracle_eager(tree):
    """ 
    Détermine la séquence de transitions nécessaire pour construire l'arbre de dépendances donné, en suivant l'algorithme arc-eager tel que décrit par Goldberg & Nivre (2012).

    Args : 
        tree : objet de type DependencyTree représentant l'arbre de dépendances à construire.
    
    Sortie : 
        Une liste de tuples (action, label) représentant la séquence de transitions. 
    """

    state = ArcEager(tree.get_tokens())
    actions = []

    # Boucle principale : on continue tant que l'état n'est pas final
    while not state.is_final():
        # Vérifie si une transition LEFT-ARC est possible
        # Condition : le sommet de la pile est dépendant du sommet du buffer
        if state.can_left_arc() and tree.get_head(state.get_stack_top()) == state.get_buffer_top():
            # Action LEFT-ARC
            actions.append(("LA", tree.get_dep(state.get_stack_top())))
            # Transition LEFT-ARC
            state.left_arc(tree.get_dep(state.get_stack_top()))

        # Vérifie si une transition RIGHT-ARC est possible
        # Condition : le sommet du buffer est dépendant du sommet de la pile
        elif state.can_right_arc() and tree.get_head(state.get_buffer_top()) == state.get_stack_top():
            # Action RIGHT-ARC
            actions.append(("RA", tree.get_dep(state.get_buffer_top())))
            # Transition RIGHT-ARC
            state.right_arc(tree.get_dep(state.get_buffer_top()))

        # Vérifie si une transition REDUCE est possible
        # Condition : le sommet de la pile a reçu tous ses dépendants
        elif state.can_reduce() and all(tree.get_head(d) == state.get_stack_top()
                                      for d in range(1, len(tree) + 1)
                                      if tree.get_head(d) == state.get_stack_top()):
            # Action REDUCE
            actions.append(("RE", None))
            # Transition REDUCE
            state.reduce()

        # Sinon, on applique la transition SHIFT comme action par défaut
        else:
            actions.append(("SH", None))
            state.shift()

    return actions

def compute_evaluation(gold_trees, pred_trees):
    """
    Args : 
        gold_trees: liste d'objets DependencyTree
        pred_trees: liste d'objets DependencyTree
    
    Retourne un dictionnaire avec 3 clés: 
        {"pos": <pos_accuracy>, 
         "uas": <uas>,
         "las": <las> }
    """
    assert(len(gold_trees) == len(pred_trees))

    total_tokens = 0
    correct_las = 0
    correct_uas = 0
    correct_pos = 0
    
    # On boucle sur le deux arbres (le prédit et la référence)
    for gold_tree, pred_tree in zip(gold_trees, pred_trees):
        total_tokens += len(gold_tree)
        
        # Compare les têtes pour UAS
        gold_heads = gold_tree.get_heads()
        pred_heads = pred_tree.get_heads()
        # On compte le nombre d'éléments correctement prédits
        correct_uas += sum(1 for g, p in zip(gold_heads, pred_heads) if g == p)
        
        # Compare les têtes ET les labels pour LAS 
        gold_labels = gold_tree.get_deps()
        pred_labels = pred_tree.get_deps()
        # On compte le nombre d'éléments correctement prédits
        correct_las += sum(1 for g, p, gh, ph in zip(gold_labels, pred_labels, gold_heads, pred_heads) 
                          if g == p and gh == ph)
        
        # Compare les étiquettes POS
        gold_pos = gold_tree.get_pos_tags()
        pred_pos = pred_tree.get_pos_tags()
        # On compte le nombre d'éléments correctement prédits
        correct_pos += sum(1 for g, p in zip(gold_pos, pred_pos) if g == p)

    # On renvoie le ratio d'élements corrects
    return {
        "pos": (correct_pos / total_tokens)*100,
        "uas": (correct_uas / total_tokens)*100, 
        "las": (correct_las / total_tokens)*100,
    }

def export_treebank(treebank, filename):
    """
    Exporte une liste d'arbres vers un fichier au format CoNLL
    """
    with open(filename, 'w', encoding='utf8') as f:
        for tree in treebank:
            # On applique la fonction __str__ pour le formatage du fichier
            f.write(str(tree) + "\n\n")


if __name__ == "__main__":
    torch.manual_seed(10)
    random.seed(10)
    import argparse
    parser = argparse.ArgumentParser(description='Toy Bi-LSTM transition-based parser')
    parser.add_argument('config', help='Path to yaml config file')
    args = parser.parse_args()
    
    with open(args.config) as instream:
        hp = HyperParameters(yaml.load(instream, Loader=yaml.SafeLoader))

    # treebank = list of DependencyTree objects
    treebank = read_treebank(hp.treebank)

    # On garde seulement les arbres projectifs (car arc-eager ne peut pas produire d'abres non projectifs)
    treebank = [tree for tree in treebank if tree.is_projective()]
    
    # Train / dev / test split
    train, dev, test = train_dev_test_split(treebank)

    vocabulary = get_vocabulary(train)
    
    parser = BiLstmDependencyParser(vocabulary, hp)
    optimizer = AdamW(parser.parameters(), lr=hp.learning_rate)
    
    best_las_on_dev = 0
    # Boucle d'apprentissage
    for epoch in range(hp.epochs):
        epoch_loss_parse = 0
        epoch_loss_tag = 0
        parser.train()
        
        for tree in train:
            output_dict = parser(tree.get_tokens(), static_oracle_eager(tree), tree.get_pos_tags())
            
            loss = output_dict["parsing_loss"] + output_dict["tagging_loss"]
            epoch_loss_parse += output_dict["parsing_loss"].item()
            epoch_loss_tag += output_dict["tagging_loss"]
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

        parser.eval()
        pred_trees = []
        for tree in dev:
            output_tree = parser(tree.get_tokens())["tree"]
            pred_trees.append(output_tree)

        output_eval = compute_evaluation(dev, pred_trees)
        pos = output_eval["pos"]
        uas = output_eval["uas"]
        las = output_eval["las"]
        
        if las > best_las_on_dev:
            torch.save(parser, "best_model")
            best_las_on_dev = las
        
        print(f"Epoch {epoch}, parse loss: {epoch_loss_parse:.4f} tag loss: {epoch_loss_tag:.4f}   pos: {pos} uas: {uas} las: {las}")
    
    parser = torch.load("best_model")
    parser.eval()
    pred_trees = []
    for tree in test:
        output_tree = parser(tree.get_tokens())["tree"]
        # On ajoute la phrase complète analysée
        output_tree.sentence_text = " ".join(tree.get_tokens())
        pred_trees.append(output_tree)
    output_eval = compute_evaluation(test, pred_trees)
    pos = output_eval["pos"]
    uas = output_eval["uas"]
    las = output_eval["las"]

    print(f"Final evaluation on test set: pos: {pos} uas: {uas} las: {las}")

    # Export des arbres prédits dans un fichier
    export_treebank(pred_trees, "predicted_test_trees.conll")




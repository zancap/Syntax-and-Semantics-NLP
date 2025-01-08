from dependency_parser_canevas_salome import ArcEager


def print_configuration(state):
    """state: ArcEager Object"""
    print(f"stack : {state.stack}")
    print(f"buffer: {state.buffer}")
    print(f"arcs  : {state.arcs}")
    print(f"deps  : {state.labels}")


if __name__ == "__main__":

    phrase = "le chat mange une pomme".split()

    tokens = [f"{i+1}:{t}" for i, t in enumerate(phrase)]
    tokens = " ".join(tokens)
    print()
    print(f"tokens et leurs indices: {tokens}")
    print()


    state = ArcEager(phrase)

    print("Configuration initiale:")
    print()
    print_configuration(state)
    print()

    state.shift()
    
    print("Configuration après un shift:")
    print()
    print_configuration(state)
    print()
    
    state.left_arc("det")
    print("puis après un left-arc-det:")
    print("on a créé l'arc 1 -> 2 (le -> chat), donc state.arcs[1] contient 2")
    print("avec le label 'det'   donc state.labels[1] contient 'det'")
    print()
    print_configuration(state)
    print()


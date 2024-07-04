'''
During setting up the edge_types, if we want to include skill-skill or expert-expert connections,
the edge_type identifier "sm" or "stm" will have to be included as "sm.en" or "stm.en" [en = enhanced]
'''


settings = {
    'graph':{
        'edge_types':
            # ('member', 'm'),
            # ([('skill', 'to', 'member')], 'sm'),
            # ([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'member')], 'sm'), # sm enhanced
            # ([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm'),
            # ([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm') # stm enhanced,
            # [([('skill', 'to', 'member')], 'sm'), ([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm')],
            [([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'member')], 'sm.en'), ([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm.en')], # sm stm strongly connected

        'custom_supervision' : True, # if false, it will take all the forward edge_types as supervision edges
        # 'supervision_edge_types': [([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'member')], 'sm'), ([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm')], # sm stm strongly connected
        'supervision_edge_types': [([('skill', 'to', 'member')], 'sm.en'), ([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm.en')],
        'dir': False,
        'dup_edge': 'mean',         #None: keep the duplicates, else: reduce by 'add', 'mean', 'min', 'max', 'mul'
    },
    'model': {
        'd' : 8,                    # embedding dim array
        'b' : 128,                  # batch_size for loaders
        'e' : 100,                  # num epochs
        'ns' : 5,                   # number of negative samples
        'lr': 0.001,
        'loader_shuffle': True,
        'num_workers': 0,
        'save_per_epoch': False,
        'pt' : 0,                   # 1 -> use pretrained d2v skill vectors as initial node features of graph data
        'w2v': {
            'd' : 8,
            'max_epochs' : 100,
            'dm': 1,                # training algorithm. 1: distributed memory (PV-DM), 0: distributed bag of words (PV-DBOW)
            'dbow_words': 1,        # train word-vectors in skip-gram fashion; 0: no (default), 1: yes
            'window': 10,           # cooccurrence window
            'embtype': 'skill',     # embedding types : 'skill', 'member', 'joint', 'dt2v'
        },
        'gnn.n2v': {
            'walk_length': 5,
            'context_size': 2,
            'walks_per_node': 10,
            'num_negative_samples': 10,
            'p': 1.0,
            'q': 1.0,
        },
        'gnn.gcn': {
            'hidden_dim': 10,
            'p': 1.0,
            'q': 1.0,
        },
        'gnn.gs': {
            'e' : 5,                # number of epochs
            'b' : 128,              # batch size
            'd' : 8,                # embedding dimension
            'ns' : 2,               # number of negative samples
            'h' : 2,                # number of attention heads (if applicable)
            'nn' : [20, 10],        # number of neighbors in each hop ([20, 10] -> 20 neighbors in first hop, 10 neighbors in second hop)
            'graph_type' : 'stm',   # graph type used (stm -> ste -> skill-team-expert)
        },
        'gnn.gin': {
            'e': 5,
            'b': 128,
            'd': 8,
            'ns': 2,
            'h': 2,
            'nn': [20, 10],
            'graph_type': 'stm',
        },
        'gnn.gat': {
            'e': 5,
            'b': 128,
            'd': 8,
            'ns': 2,
            'h': 2,
            'nn': [20, 10],
            'graph_type': 'stm',
        },
        'gnn.gatv2': {
            'e': 5,
            'b': 128,
            'd': 8,
            'ns': 2,
            'h': 2,
            'nn': [20, 10],
            'graph_type': 'stm',
        },
        'gnn.han': {
            'e': 5,
            'b': 128,
            'd': 8,
            'ns': 2,
            'h': 2,
            'nn': [20, 10],
            'graph_type': 'stm',
        },
        'gnn.gine': {
            'e': 5,
            'b': 128,
            'd': 8,
            'ns': 2,
            'h': 2,
            'nn': [20, 10],
            'graph_type': 'stm',
        },
        'gnn.m2v': {
            'graph_type':'stm', # this value will be changed during runtime in each loop according to the graph_type and then be used in the embedding_output var
            'metapath' : {
                'sm' : [
                    ('member','rev_to','skill'),
                    ('skill', 'to', 'member'),
                ],
                'stm' : [
                    ('member','to','team'),
                    ('team', 'rev_to', 'skill'),
                    ('skill','to','team'),
                    ('team', 'rev_to', 'member'),
                ],
            },
            'walk_length': 10,
            'context_size': 10,
            'walks_per_node': 20,
            'ns' : 2,
            'd': 8,
            'b': 128,
            'e': 100,
        },
    },
    'data':{
        'dblp':{},
        'uspt':{},
        'imdb':{},
        'node_types': ['member'], #['id', 'skill', 'member'],
    },
    'cmd' : ['graph', 'emb'],
    'main':{
        'model': 'm2v',
        'domains': ['uspt','imdb','dblp'],
        'node_types': ['id', 'skill', 'member'],
        'edge_types': 'STE',
    },
}
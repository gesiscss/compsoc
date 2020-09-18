def elite_families_collection(path='data/elite_families/'):
    '''
    Description: Loads the normalized elite families data collection.
    
    Output: Four dataframes in this order: families, parties, relations, domains
    '''
    import pandas as pd
    families = pd.read_csv(path+'families.txt', sep='\t', encoding='utf-8')
    parties = pd.read_csv(path+'parties.txt', sep='\t', encoding='utf-8')
    relations = pd.read_csv(path+'relations.txt', sep='\t', encoding='utf-8')
    domains = pd.read_csv(path+'domains.txt', sep='\t', encoding='utf-8')
    
    return families, parties, relations, domains

def sns_collection(path='data/sns/'):
    '''
    Description: Loads the normalized Social Network Science data collection.
    
    Output: Six dataframes in this order: publications, subfields, authors, authorships, 
        words, usages
    '''
    
    import pandas as pd
    
    publications = pd.read_csv(path+'publications.txt', sep='\t', encoding='utf-8')
    subfields = pd.read_csv(path+'subfields.txt', sep='\t', encoding='utf-8')
    authors = pd.read_csv(path+'authors.txt', sep='\t', encoding='utf-8')
    authorships = pd.read_csv(path+'authorships.txt', sep='\t', encoding='utf-8')
    words = pd.read_csv(path+'words.txt', sep='\t', encoding='utf-8')
    usages = pd.read_csv(path+'usages.txt', sep='\t', encoding='utf-8')
    
    return publications, subfields, authors, authorships, words, usages

def copenhagen_collection(path='data/copenhagen/'):
    '''
    Description: Loads the normalized Copenhagen Networks Study data collection.
    
    Output: Six dataframes in this order: users, genders, bluetooth, calls, sms, facebook_friends
    '''
    # functions
    def replace_timestamp(df, timestamp):
        minute = (df[timestamp]/60).astype(int)
        hour = (minute/60).astype(int)
        day = (hour/24).astype(int).rename('day')
        hour = (hour-day*24).rename('hour')
        minute = (minute-day*24*60-hour*60).rename('minute')
        second = (df[timestamp]-day*24*60*60-hour*60*60-minute*60).rename('second')
        df.rename(columns={timestamp: 'time'}, inplace=True)
        df = pd.concat([df, day, hour, minute, second], axis=1)
        return df
    
    def replace_identifier(df, old_identifier, new_identifier):
        df = pd.merge(left=df, right=users[['user', 'user_id']], left_on=old_identifier, right_on='user')
        df.rename(columns={'user_id': new_identifier}, inplace=True)
        df.drop([old_identifier, 'user'], axis=1, inplace=True)
        return df
    
    # load data
    import os
    import pandas as pd
    bluetooth = pd.read_csv(os.path.join(path, 'bt_symmetric.csv'))
    calls = pd.read_csv(os.path.join(path, 'calls.csv'))
    sms = pd.read_csv(os.path.join(path, 'sms.csv'))
    facebook_friends = pd.read_csv(os.path.join(path, 'fb_friends.csv'))
    attributes = pd.read_csv(os.path.join(path, 'genders.csv'))
    
    # create ``users`` dataframe
    import itertools
    users = set(itertools.chain(*[
        bluetooth['user_a'].to_list(), 
        bluetooth['user_b'].to_list(), 
        calls['caller'].to_list(), 
        calls['callee'].to_list(), 
        sms['sender'].to_list(), 
        sms['recipient'].to_list(), 
        facebook_friends['# user_a'].to_list(), 
        facebook_friends['user_b'].to_list(), 
        attributes['# user'].to_list()
    ]))
    users = pd.DataFrame(list(users), columns=['user'])
    users = users[users['user'] >= 0]
    users = pd.merge(left=users, right=attributes, left_on='user', right_on='# user', how='left')
    users.fillna(2, inplace=True)
    users.rename(columns={'female': 'gender_id'}, inplace=True)
    users['gender_id'] = users['gender_id'].astype(int)
    users.drop(['# user'], axis=1, inplace=True)
    users['user_id'] = users.index
    users = users[['user_id', 'user', 'gender_id']]
    
    # create ``genders`` dataframe
    genders = pd.DataFrame([[0, 'male'], [1, 'female'], [2, 'unknown']], columns=['gender_id', 'gender'])
    
    # create ``bluetooth`` dataframe
    bluetooth = replace_timestamp(bluetooth, '# timestamp')
    bluetooth = bluetooth[bluetooth['rssi'] < 0]
    bluetooth['rssi'] = bluetooth['rssi']+100
    bluetooth.rename(columns={'rssi': 'strength'}, inplace=True)
    bluetooth = bluetooth[~bluetooth['user_b'].isin([-1, -2])]
    bluetooth = replace_identifier(bluetooth, 'user_a', 'user_id_from')
    bluetooth = replace_identifier(bluetooth, 'user_b', 'user_id_to')
    
    bluetooth_reversed = bluetooth.copy()[['time', 'strength', 'day', 'hour', 'minute', 'second', 'user_id_to', 'user_id_from']]
    bluetooth_reversed.columns = bluetooth.columns
    
    bluetooth = bluetooth.append(bluetooth_reversed)
    bluetooth.sort_values(['time', 'user_id_from', 'user_id_to'], inplace=True)
    bluetooth.reset_index(drop=True, inplace=True)
    bluetooth = bluetooth[['user_id_from', 'user_id_to', 'strength', 'time', 'day', 'hour', 'minute', 'second']]
    
    # create ``calls`` dataframe
    calls = replace_timestamp(calls, 'timestamp')
    calls = replace_identifier(calls, 'caller', 'user_id_from')
    calls = replace_identifier(calls, 'callee', 'user_id_to')
    calls['duration'].replace(-1, 0, inplace=True)
    calls.sort_values(['time', 'user_id_from', 'user_id_to'], inplace=True)
    calls.reset_index(drop=True, inplace=True)
    calls = calls[['user_id_from', 'user_id_to', 'duration', 'time', 'day', 'hour', 'minute', 'second']]
    
    # create ``sms`` dataframe
    sms = replace_timestamp(sms, 'timestamp')
    sms = replace_identifier(sms, 'sender', 'user_id_from')
    sms = replace_identifier(sms, 'recipient', 'user_id_to')
    sms.sort_values(['time', 'user_id_from', 'user_id_to'], inplace=True)
    sms.reset_index(drop=True, inplace=True)
    sms = sms[['user_id_from', 'user_id_to', 'time', 'day', 'hour', 'minute', 'second']]
    
    # create ``facebook_friends`` dataframe
    facebook_friends = replace_identifier(facebook_friends, '# user_a', 'user_id_from')
    facebook_friends = replace_identifier(facebook_friends, 'user_b', 'user_id_to')
    facebook_friends.sort_values(['user_id_from', 'user_id_to'], inplace=True)
    facebook_friends.reset_index(drop=True, inplace=True)
    
    return users, genders, bluetooth, calls, sms, facebook_friends

def weighted_edge_list_to_unlayered(
    edge_list, 
    function='sum'
):
    '''
    Description: Transforms a weighted layered to an unlayered edge list.
    
    Inputs:
        edge_list: Dataframe of a weighted layered edge list; first column must be 
            identifier of node u, second column must be identifier of node v, third column 
            must be edge weight w, fourth column must be the layer identifier; the layer 
            identifier must be an integer from 0 to n-1 where n is the number of layers; 
            all but the first four columns will be discarded.
        function: Function to chose edge weight after transformation; if 'min' the 
            smaller weight will be chosen, if 'max' the larger weight will be chosen, if 
            'sum' the weights of (u, v) and (v, u) will be summed; set to 'sum' by default.
    
    Output: Dataframe of a weighted unlayered edge list.
    '''
    
    df = edge_list.copy()
    if function == 'sum':
        df = df.groupby(df.columns[:2].tolist()).sum().reset_index()
    if function == 'min':
        df = df.groupby(df.columns[:2].tolist()).min().reset_index()
    if function == 'max':
        df = df.groupby(df.columns[:2].tolist()).max().reset_index()
    
    return df[df.columns[:3]]

def weighted_edge_list_to_undirected(
    edge_list, 
    reciprocal=False, 
    function='sum'
):
    '''
    Description: Transforms a directed to an undirected weighted edge list.
    
    Inputs:
        edge_list: Dataframe of a directed weighted edge list; first column must be 
            identifier of node u, second column must be identifier of node v, third column 
            must be edge weight w; all but the first three columns will be discarded.
        reciprocal: Boolean variable if only reciprocated ties should be kept; set to 
            False by default.
        function: Function to chose edge weight after transformation; if 'min' the 
            smaller weight will be chosen, if 'max' the larger weight will be chosen, if 
            'sum' the weights of (u, v) and (v, u) will be summed; set to 'sum' by default.
    
    Output: Dataframe of an undirected weighted edge list.
    '''
    
    import numpy as np
    import pandas as pd
    
    # order node tuples
    df = edge_list[edge_list.columns[:2]].copy()
    df = df.T
    df = df.transform(np.sort)
    df = df.T
    df = pd.concat([df, edge_list[edge_list.columns[2]]], axis=1)
    
    # flag reciprocal edges
    if reciprocal:
        df_count = df.groupby(df.columns[:2].tolist()).count().reset_index()
    
    # apply function
    if function == 'sum':
        df = df.groupby(df.columns[:2].tolist()).sum().reset_index()
    if function == 'min':
        df = df.groupby(df.columns[:2].tolist()).min().reset_index()
    if function == 'max':
        df = df.groupby(df.columns[:2].tolist()).max().reset_index()
    
    # keep only reciprocal edges
    if reciprocal:
        df = df[df_count[df_count.columns[2]] == 2].reset_index(drop=True)
    
    return df

def weighted_layered_edge_list_to_undirected(
    edge_list, 
    reciprocal=False, 
    function='sum'
):
    '''
    Description: Transforms a directed to an undirected weighted layered edge list.
    
    Inputs:
        edge_list: Dataframe of a directed weighted layered edge list; first column must 
            be identifier of node u, second column must be identifier of node v, third 
            column must be edge weight w, fourth column must be the layer identifier; the 
            layer identifier must be an integer from 0 to n-1 where n is the number of 
            layers; all but the first four columns will be discarded.
        reciprocal: Boolean variable if only reciprocated ties should be kept; set to 
            False by default.
        function: Function to chose edge weight after transformation; if 'min' the 
            smaller weight will be chosen, if 'max' the larger weight will be chosen, if 
            'sum' the weights of (u, v) and (v, u) will be summed; set to 'sum' by default.
    
    Output: Dataframe of an undirected weighted layered edge list.
    '''
    
    import compsoc as cs
    import pandas as pd
    
    edge_list_undirected = pd.DataFrame(columns=edge_list.columns[:4])
    for identifier in set(edge_list[edge_list.columns[3]]):
        df = edge_list[edge_list[edge_list.columns[3]] == identifier]
        df = cs.weighted_edge_list_to_undirected(edge_list=df, reciprocal=reciprocal, function=function)
        df[edge_list.columns[3]] = identifier
        edge_list_undirected = pd.concat([edge_list_undirected, df])
    
    return edge_list_undirected.reset_index(drop=True)

def project_selection_matrix(
    selections, 
    how, 
    transaction_id='transaction_id', 
    fact_id='fact_id', 
    norm=True, 
    remove_loops=True, 
    symmetrize=True
):
    '''
    Description: Projects a selection matrix to a transaction similarity matrix or a fact 
        co-selection matrix; computes fact attributes; computes cumulative co-selection 
        fractions for matrix filtering.
    
    Inputs:
        selections: Dataframe containing the selection matrix indices and data; must 
            contain a 'weight' column that contains the cell weights.
        how: String that specifies which projection is to be made; must be either 
            'transactions' or 'facts'.
        transaction_id: Name of the column of the dataframe ``selections`` that holds the 
            identifiers of the transactions selecting facts.
        fact_id: Name of the column of the dataframe ``selections`` that holds the 
            identifiers of the facts getting selected in transactions.
        norm: Boolean parameter specifying if matrix normalization should be performed.
        remove_loops: Boolean parameter specifying if the matrix diagonal should be 
            removed; if False, loops will be included in computing cumulative 
            co-selection fractions.
        symmetrize: Boolean parameter specifying if the lower portion of the matrix 
            should be removed.
    
    Output: A dataframe containing the projected matrices (enriched by cumulative 
        fractions in the case of a normalized projection to the fact mode); a dataframe 
        containing matrix-based attributes of transactions or facts (depending on the 
        type of projection)
    '''
    
    # function
    def get_unique(s):
        l = s.unique().tolist()
        return {identifier: index for index, identifier in enumerate(l)}
    
    # map identifiers of transactions and facts to unique integers
    import pandas as pd
    d_transactions_indices = get_unique(selections[transaction_id])
    d_facts_indices = get_unique(selections[fact_id])
    
    # construct selection matrix
    rows = [d_transactions_indices[transaction_id] for transaction_id in selections[transaction_id].values]
    columns = [d_facts_indices[fact_id] for fact_id in selections[fact_id].values]
    cells = selections['weight'].tolist()
    from scipy.sparse import csr_matrix, coo_matrix, triu
    G = coo_matrix((cells, (rows, columns))).tocsr()
    GT = csr_matrix.transpose(G)
    from sklearn.preprocessing import normalize
    GN = normalize(G, norm='l1', axis=1)
    
    # project selection matrix ...
    import numpy as np
    
    # ... to transaction similarity matrix
    if how == 'transactions':
        if norm == True:
            GNT = csr_matrix.transpose(GN)
            H = GN*GNT
        else:
            H = G*GT
        
        # derive transaction attributes dataframe
        H_nodiag = H.tolil()
        H_nodiag.setdiag(values=0)
        
        k = pd.Series([len(i) for i in H_nodiag.data.tolist()])
        w = pd.Series(np.array(H.diagonal()))
        if norm == True:
            w = (1/w).round(4)
        else:
            w = w.round(4)
        
        d_indices_transactions = {index: identifier for identifier, index in d_transactions_indices.items()}
        
        transaction_attributes = pd.concat([pd.Series(d_indices_transactions), k, w], axis=1)
        transaction_attributes.columns = [transaction_id, 'degree', 'weight']
        
        # construct similarities dataframe
        if remove_loops == True:
            H = H.tolil()
            H.setdiag(0)

        if symmetrize == True:
            H = triu(H.tocoo()).tocsr()
        else:
            H = H.tocsr()
        
        transaction_id_from = [d_indices_transactions[index] for index in H.nonzero()[0].tolist()]
        transaction_id_to = [d_indices_transactions[index] for index in H.nonzero()[1].tolist()]
        weight = H.data.tolist()
        
        similarities = pd.concat([pd.Series(transaction_id_from), pd.Series(transaction_id_to), pd.Series(weight)], axis=1)
        similarities.columns = [transaction_id+'_from', transaction_id+'_to', 'similarity']
        
        return similarities, transaction_attributes
    
    # ... to fact co-selection matrix
    if how == 'facts':
        if norm == True:
            I = GT*GN
        else:
            I = GT*G
        
        # derive fact attributes dataframe
        I_nodiag = I.tolil()
        I_nodiag.setdiag(values=0)
        
        k = pd.Series([len(i) for i in I_nodiag.data.tolist()])
        
        d_indices_facts = {index: identifier for identifier, index in d_facts_indices.items()}
        
        if norm == True:
            w = pd.Series(np.squeeze(np.array(I.sum(axis=1)))).round(4)
            a = pd.Series(np.array(I.diagonal())).round(4)
            e = (1-a/w).round(4)
            s = (k/w).round(4)
            
            fact_attributes = pd.concat([pd.Series(d_indices_facts), k, w, a, e, s], axis=1)
            fact_attributes.columns = [fact_id, 'degree', 'weight', 'autocatalysis', 'embeddedness', 'sociability']
            
        else:
            fact_attributes = pd.concat([pd.Series(d_indices_facts), k], axis=1)
            fact_attributes.columns = [fact_id, 'degree']
        
        # construct co-selections dataframe with cumulative co-selection fractions
        if remove_loops == True:
            I = I.tolil()
            I.setdiag(0)
        
        if symmetrize == True:
            I = triu(I.tocoo()).tocsr()
        else:
            I = I.tocsr()
                
        fact_id_from = [d_indices_facts[index] for index in I.nonzero()[0].tolist()]
        fact_id_to = [d_indices_facts[index] for index in I.nonzero()[1].tolist()]
        weight = I.data.tolist()
        
        co_selections = pd.concat([pd.Series(fact_id_from), pd.Series(fact_id_to), pd.Series(weight)], axis=1)
        co_selections.columns = [fact_id+'_from', fact_id+'_to', 'weight']
        
        co_selections_cumfrac = co_selections.copy()
        co_selections_cumfrac.index = co_selections_cumfrac.weight
        co_selections_cumfrac = co_selections_cumfrac['weight'].groupby(co_selections_cumfrac.index).sum()
        co_selections_cumfrac = co_selections_cumfrac.sort_index(ascending=False)
        co_selections_cumfrac = co_selections_cumfrac.cumsum()/sum(co_selections_cumfrac)
        co_selections_cumfrac = co_selections_cumfrac.round(4)
        co_selections_cumfrac.rename('cumfrac', inplace=True)
        
        co_selections = pd.merge(left=co_selections, right=co_selections_cumfrac, left_on='weight', right_on=co_selections_cumfrac.index)
        
        return co_selections, fact_attributes

def meaning_structures(
    selections, 
    transaction_id, 
    fact_id, 
    multiplex=False, 
    transactions=None, 
    domain_id=None, 
    facts=None, 
    norm=True, 
    remove_loops=True, 
    symmetrize=True
):
    '''
    Description: Projects a selection matrix to (multiplex) co-selection matrix.
    
    Inputs:
        selections: Dataframe containing the selection matrix indices and data; must 
            contain a 'weight' column that contains the cell weights.
        transaction_id: Name of the column of the dataframe ``selections`` that holds the 
            identifiers of the transactions selecting facts.
        fact_id: Name of the column of the dataframe ``selections`` that holds the 
            identifiers of the facts getting selected in transactions.
        multiplex: Boolean parameter specifying if selections occurr in multiple domains; 
            set to False by default.
        transactions: Dataframe containing the ``transaction_id`` identifiers of the 
            ``selections`` dataframe; must be specified if ``multiplex=True``; set to None 
            by default.
        domain_id: Name of the column of the dataframe ``transactions`` that holds the 
            identifiers of the domains the transactions belong to; must be an integer from 
            0 to d where d is the number of domains; must be specified if 
            ``multiplex=True``; set to None by default.
        facts: Dataframe containing the ``fact_id`` identifiers of the ``selections`` 
            dataframe; if specified, it will be enriched by fact attributes; set to None 
            by default.
        norm: Boolean parameter specifying if matrix normalization should be performed.
        remove_loops: Boolean parameter specifying if the matrix diagonal should be 
            removed; if False, loops will be included in computing cumulative 
            co-selection fractions.
        symmetrize: Boolean parameter specifying if the lower portion of the matrix 
            should be removed.
    
    Output: At least two dataframes will be returned: first, a dataframe containing the 
        co-selection matrix independent of domain; second, a dataframe containing fact 
        attributes (if no ``facts`` dataframe is provided), or an enriched ``facts`` 
        dataframe (if one is provided), independent of domain. When ``multiplex=True`` 
        two additional dataframes will be returned: third, a dataframe containing the 
        co-selection matrix for domains; fourth, a list of dataframes containing fact 
        attributes (if no ``facts`` dataframe is provided), or a list of enriched 
        ``facts`` dataframes (if a ``facts`` dataframe is provided), for domains.
    '''
    
    if multiplex == True:
        if transactions is None:
            print('A transactions dataframe must be specified.')
        else:
            if domain_id is None:
                print('The domain identifier for the transactions dataframe must be specified.')
            else:
                if domain_id not in transactions.columns:
                    print('The specified domain identifier is not a column in the transactions dataframe.')
                else:
                    domain_ids = set(transactions[domain_id])
                    if (len(domain_ids) > 1) & (min(domain_ids) == 0) & (max(domain_ids) == len(domain_ids)-1):
                        
                        # co-selections and fact attributes dataframes independent of domain
                        co_selections, fact_attributes = project_selection_matrix(selections=selections, how='facts', transaction_id=transaction_id, fact_id=fact_id, norm=norm, remove_loops=remove_loops, symmetrize=symmetrize)
                        
                        # co-selections and fact attributes dataframes for domains
                        import pandas as pd
                        co_selections_domain = pd.DataFrame(columns=[fact_id+'_from', fact_id+'_to', 'weight', 'cumfrac', domain_id])
                        fact_attributes_domain = []
                        facts_enriched_domain = []
                        for identifier in set(transactions[domain_id]):
                            df = selections[selections[transaction_id].isin(transactions[transactions[domain_id] == identifier][transaction_id])]
                            df_co_selections, df_fact_attributes = project_selection_matrix(selections=df, how='facts', transaction_id=transaction_id, fact_id=fact_id, norm=norm, remove_loops=remove_loops, symmetrize=symmetrize)
                            df_co_selections[domain_id] = identifier
                            co_selections_domain = pd.concat([co_selections_domain, df_co_selections])
                            if facts is None:
                                fact_attributes_domain.append(df_fact_attributes)
                            else:
                                df_facts_enriched = pd.merge(left=facts, right=df_fact_attributes, on=fact_id, how='left')
                                facts_enriched_domain.append(df_facts_enriched)
                        co_selections_domain.reset_index(drop=True, inplace=True)
                        if facts is None:
                            return co_selections, fact_attributes, co_selections_domain, fact_attributes_domain
                        else:
                            facts_enriched = pd.merge(left=facts, right=fact_attributes, on=fact_id, how='left')
                            
                            return co_selections, facts_enriched, co_selections_domain, facts_enriched_domain
                    else:
                        print('The specified domain identifier does not contain multiple domains or domains are not coded as integers starting with zero.')
    else:
        
        # co-selections and fact attributes dataframes independent of domain
        co_selections, fact_attributes = project_selection_matrix(selections=selections, how='facts', transaction_id=transaction_id, fact_id=fact_id, norm=norm, remove_loops=remove_loops, symmetrize=symmetrize)
        
        if facts is None:
            return co_selections, fact_attributes
        else:
            facts_enriched = pd.merge(left=facts, right=fact_attributes, on=fact_id, how='left')
            
            return co_selections, facts_enriched

def construct_graph(
    directed, 
    multiplex, 
    graph_name, 
    node_list, 
    edge_list, 
    node_pos=None, 
    node_size=None, 
    node_color=None, 
    node_shape=None, 
    node_border_color=None, 
    node_label=None, 
    attribute_shape={0: 's', 1: 'o', 2: '^', 3: '>', 4: 'v', 5: '<', 6: 'd', 7: 'p', 8: 'h', 9: '8'}, 
    layer_color={0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3', 4: '#ff7f00', 5: '#ffff33', 6: '#a65628', 7: '#f781bf', 8: '#999999'}
):
    '''
    Description: Constructs a graph from a node list and an edge list
    
    Inputs:
        directed: Boolean parameter specifying if graph should be directed.
        multiplex: Boolean parameter specifying if graph should be multiplex.
        graph_name: Name of the graph (string); must be specified.
        node_list: Dataframe containing the node properties; must contain a continuous 
            index from 0 to N-1 where N is the number of vertices; must contain a column 
            holding the name of each vertex; must contain a column holding an integer that 
            codes the class a vertex belongs to (used to color the vertices).
        node_pos: List of two columns of the dataframe ``node_list`` that hold the x and y 
            positions of each node; must be numerical variables; set to ``None`` by default.
        node_size: Name of the column of the dataframe ``node_list`` that holds the size 
            of each node; must be a numerical variable; set to ``None`` by default.
        node_color: Name of the column of the dataframe ``node_list`` that holds the color 
            of each node; must be a hexadecimal color variable; set to ``None`` by default.
        node_shape: Name of the column of the dataframe ``node_list`` that codes the shape 
            of each node; must be an integer between 0 and 9; set to ``None`` by default.
        node_border_color: Name of the column of the dataframe ``node_list`` that holds 
            the color of each node border; must be a hexadecimal color variable; set to 
            ``None`` by default.
        node_label: Name of the column of the dataframe ``node_list`` that holds the name 
            of each node; must be a string variable; set to ``None`` by default.
        attribute_shape: Dictionary containing the mapping from the integer stored in the 
            'node_shape' column to a shape; matplotlib.scatter markers 'so^>v<dph8' are 
            used by default.
        edge_list: Dataframe with exactly three columns (source node id, target node id, 
            edge weight; in that order) containing the edges of the graph; if the graph is 
            multiplex, a fourth column must contain an integer between 0 and N-1 where N 
            is the number of edge layers; must be specified.
        layer_color: Dictionary containing the mapping from the layer integer stored in 
            the fourth column of the ``edge_list`` to a hexadecimal color; a dictionary of 
            nine colors that are qualitatively distinguishable is used by default.
    
    Output: networkx graph object, potentially with graph, node, and edge attributes.
    '''
    # create graph object
    import networkx as nx
    if directed:
        if multiplex: g = nx.MultiDiGraph(name=graph_name)
        else: g = nx.DiGraph(name=graph_name)
    else:
        if multiplex: g = nx.MultiGraph(name=graph_name)
        else: g = nx.Graph(name=graph_name)
    
    # populate graph with vertices and their properties
    for i in node_list.index:
        g.add_node(i)
        if node_pos: g.nodes[i]['node_pos'] = node_list[node_pos].values[i]
        if node_size: g.nodes[i]['node_size'] = node_list[node_size][i]
        if node_color: g.nodes[i]['node_color'] = node_list[node_color][i]
        if node_shape: g.nodes[i]['node_shape'] = attribute_shape[node_list[node_shape][i]]
        if node_border_color: g.nodes[i]['node_border_color'] = node_list[node_border_color][i]
        if node_label: g.nodes[i]['node_label'] = node_list[node_label][i]
    
    # populate graph with edges and their properties
    if multiplex == True:
        for i in set(edge_list[edge_list.columns[3]]):
            df = edge_list[edge_list[edge_list.columns[3]] == i][edge_list.columns[:3]]
            g.add_weighted_edges_from(df.values, edge_color=layer_color[i])
    else:
        g.add_weighted_edges_from(edge_list[edge_list.columns[:3]].values)
    
    return g

def uniform_vertex_property(
    g, 
    vertex_property
):
    '''
    Description: Creates a uniform vertex property.
    
    Inputs:
        g: Graph for which the property should be created; must be a networkx graph object.
        vertex_property: Uniform property; can be anything from a hexadecimal color to a 
            string or numerical.
    
    Output: Dictionary with vertex identifiers as keys and properties as values.
    '''
    return dict(zip(g.nodes, g.number_of_nodes()*[vertex_property]))

def partition_to_vertex_property(
    partition, 
    _dict
):
    '''
    Description: Creates a vertex property dictionary.
    
    Inputs:
        partition: Dataframe column (series); indices must be integers from 0 to n-1 where 
            n is the number of vrtices in the graph for whch the vertext property is made; 
            values must be integers from 0 to m-1 where m is the number of partitions of 
            vertex partitions.
        _dict: Dictionary that maps partition identifiers (keys) to vertex properties 
            (values); properties can be anything from hexadecimal colors to strings and 
            numericals.
    
    Output: Dictionary with vertex identifiers as keys and properties as values.
    '''
    return {index: _dict[identifier] for index, identifier in partition.items()}

def node_attribute_to_list(g, node_attribute):
    '''
    Description: Returns a node attribute as a list.
    
    Inputs:
        g: networkx graph object.
        node_attribute: Name of the edge attribute; must be a string.
    
    Output: List.
    '''
    import networkx as nx
    return list(nx.get_node_attributes(g, node_attribute).values())

def edge_attribute_to_list(g, edge_attribute):
    '''
    Description: Returns an edge attribute as a list.
    
    Inputs:
        g: networkx graph object.
        edge_attribute: Name of the edge attribute; must be a string.
    
    Output: List.
    '''
    import networkx as nx
    return list(nx.get_edge_attributes(g, edge_attribute).values())

def draw_graph(
    g, 
    node_pos='internal', 
    node_size='internal', 
    node_size_factor=1, 
    node_color='internal', 
    node_shape='internal', 
    node_border_color='internal', 
    node_border_width=1, 
    #node_label='internal', 
    font_size='node_size', 
    font_size_factor=1, 
    font_color='black', 
    edge_width='internal', 
    edge_width_factor=1, 
    edge_color='internal', 
    edge_transparency=1, 
    curved_edges=False, 
    arrow_size=18, 
    labels=None, 
    label_transparency=1, 
    figsize='large', 
    margins=.1, 
    pdf=None, 
    png=None
):
    '''
    Description: Draws a graph with internal node and edge properties
    
    Inputs:
        g: Graph to be drawn; must be networkx graph object.
        node_pos: Node positions to be used for drawing; when set to 'internal' 
            (default), then the 'node_pos' attribute will be used, else a standard 
            spring layout is inferred; node color will be 'white'; parameter can take a 
            dictionary with node positions as values; when set to None, then a standard 
            spring layout is inferred.
        node_size: Node sizes to be used for drawing; when set to 'internal' (default), 
            then the 'node_size' attribute will be used, else node size will depend on 
            the number of nodes; parameter can take a dictionary with node sizes as 
            values; when set to None, node size will depend on the number of nodes.
        node_size_factor: Factor to change node size; set to 1 by default.
        node_color: Node colors to be used for drawing; when set to 'internal' (default), 
            then the 'node_color' attribute will be used, else node color will be 
            'white'; parameter can take a dictionary with hexadecimal or string node 
            colors as values; when set to None, node color will be 'white'.
        node_shape: Node shapes to be used for drawing; when set to 'internal' (default), 
            then the 'node_shape' attribute will be used, else node shape will be 'o'; 
            parameter can take a dictionary with node shapes as values; when set to None, 
            node shape will be 'o'.
        node_border_color: Node border colors to be used for drawing; when set to 
            'internal' (default), then the 'node_border_color' attribute will be used, 
            else node border color will be 'gray'; parameter can take a dictionary with 
            hexadecimal node colors as values; when set to None, node border color will 
            be 'gray'.
        node_border_width: Width of node border; set to 1 by default.
        node_label: Node labels to be used for drawing; when set to 'internal' (default), 
            then the 'node_label' attribute will be used.
        font_size: Font sizes to be used for drawing; when set to 'node_size' (default), 
            then the 'node_size' attribute will be used, else font size will be 12; 
            parameter can take a dictionary with font sizes as values; when set to None, 
            font size will be 12.
        font_size_factor: Factor to change font size; set to 1 by default.
        font_color: set to 'black' by default; parameter can take a hexadecimal or string 
            color.
        edge_width: Edge widths to be used for drawing; when set to 'internal' (default), 
            then the 'edge_width' attribute will be used, else edge width will be 1; 
            parameter can take a list of edge widths; when set to None, edge width will 
            be 1.
        edge_width_factor: Factor to change edge width; set to 1 by default.
        edge_color: Edge colors to be used for drawing; when set to 'internal' (default), 
            then the 'edge_color' attribute will be used, else edge width be 1; parameter 
            can take a list of edge colors; when set to None, edge width will be 1.
        edge_transparency: Alpha transparency of edge colors; set to 1 by default.
        curved_edges: Boolean parameter specifying if edges should be curved.
        arrow_size: Size of arrows; set to 18 by default; must be numerical.
        labels: If 'text', then the internal 'node_label' attribute will be used; if 
            'id', then the node identifier will be used.
        label_transparency: Alpha transparency of font color; set to 1 by default.
        figsize: Size of the figure; when set to 'small', then the plot will have size 
            (4, 4); when set to 'medium', then the plot will have size (8, 8); when set 
            to 'large', then the plot will have size (12, 12).
        margins: Margins of the figure; set to .1 by default; increase it if nodes extend 
            outside the drawing area.
        pdf: Name of a pdf file to be written.
        png: Name of a png file to be written.
    
    Output: ...
    '''
    
    # use internal node and edge attributes for drawing, otherwise use external attributes or none
    import networkx as nx
    if node_pos == 'internal':
        if bool(nx.get_node_attributes(g, 'node_pos')):
            vp_node_pos = nx.get_node_attributes(g, 'node_pos')
        else:
            vp_node_pos = nx.spring_layout(g)
    else:
        if node_pos:
            vp_node_pos = node_pos
        else:
            vp_node_pos = nx.spring_layout(g)
    
    if node_size == 'internal':
        if bool(nx.get_node_attributes(g, 'node_size')):
            vp_node_size = nx.get_node_attributes(g, 'node_size')
        else:
            vp_node_size = dict(zip(g.nodes, g.number_of_nodes()*[int(30000/g.number_of_nodes())]))
    else:
        if node_size:
            vp_node_size = node_size
        else:
            vp_node_size = dict(zip(g.nodes, g.number_of_nodes()*[int(30000/g.number_of_nodes())]))
    if node_size_factor != 1:
        vp_node_size = {key: node_size_factor*value for key, value in vp_node_size.items()}
    
    if node_color == 'internal':
        if bool(nx.get_node_attributes(g, 'node_color')):
            vp_node_color = nx.get_node_attributes(g, 'node_color')
        else:
            vp_node_color = dict(zip(g.nodes, g.number_of_nodes()*['white']))
    else:
        if node_color:
            vp_node_color = node_color
        else:
            vp_node_color = dict(zip(g.nodes, g.number_of_nodes()*['white']))
    
    if node_shape == 'internal':
        if bool(nx.get_node_attributes(g, 'node_shape')):
            vp_node_shape = nx.get_node_attributes(g, 'node_shape')
        else:
            vp_node_shape = dict(zip(g.nodes, g.number_of_nodes()*['o']))
    else:
        if node_shape:
            vp_node_shape = node_shape
        else:
            vp_node_shape = dict(zip(g.nodes, g.number_of_nodes()*['o']))
        
    if node_border_color == 'internal':
        if bool(nx.get_node_attributes(g, 'node_border_color')):
            vp_node_border_color = nx.get_node_attributes(g, 'node_border_color')
        else:
            vp_node_border_color = dict(zip(g.nodes, g.number_of_nodes()*['gray']))
    else:
        if node_border_color:
            vp_node_border_color = node_border_color
        else:
            vp_node_border_color = dict(zip(g.nodes, g.number_of_nodes()*['gray']))
    
    if font_size == 'node_size':
        if bool(nx.get_node_attributes(g, 'node_size')):
            vp_font_size = nx.get_node_attributes(g, 'node_size')
        else:
            vp_font_size = dict(zip(g.nodes, g.number_of_nodes()*[12]))
    else:
        if font_size:
            vp_font_size = font_size
        else:
            vp_font_size = dict(zip(g.nodes, g.number_of_nodes()*[12]))
    if font_size_factor != 1:
        vp_font_size = {key: font_size_factor*value for key, value in vp_font_size.items()}
    
    if edge_width == 'internal':
        if bool(nx.get_edge_attributes(g, 'weight')):
            ep_edge_width = list(nx.get_edge_attributes(g, 'weight').values())
        else:
            ep_edge_width = g.number_of_edges()*[1]
    else:
        if edge_width:
            ep_edge_width = edge_width
        else:
            ep_edge_width = g.number_of_edges()*[1]
    if edge_width_factor != 1:
        ep_edge_width = [edge_width_factor*x for x in ep_edge_width]
    
    if edge_color == 'internal':
        if bool(nx.get_edge_attributes(g, 'edge_color')):
            ep_edge_color = list(nx.get_edge_attributes(g, 'edge_color').values())
        else:
            ep_edge_color = g.number_of_edges()*['gray']
    else:
        if edge_color:
            ep_edge_color = edge_color
        else:
            ep_edge_color = g.number_of_edges()*['gray']
        
    # draw nodes
    import matplotlib.pyplot as plt
    if figsize == 'small':
        plt.figure(figsize=(4, 4))
    if figsize == 'medium':
        plt.figure(figsize=(8, 8))
    if figsize == 'large':
        plt.figure(figsize=(12, 12))
    for shape in set(vp_node_shape.values()):
        nodelist = [key for key, value in vp_node_shape.items() if value == shape]
        nx.draw_networkx_nodes(
            g, 
            pos=vp_node_pos, 
            nodelist=nodelist, 
            node_size=[vp_node_size[i] for i in nodelist], 
            node_color=[vp_node_color[i] for i in nodelist], 
            node_shape=shape, 
            linewidths=node_border_width, 
            edgecolors=[vp_node_border_color[i] for i in nodelist]
        )
    
    # draw edges
    nx.draw_networkx_edges(
        g, 
        pos=vp_node_pos, 
        width=ep_edge_width, 
        edge_color=ep_edge_color, 
        alpha=edge_transparency, 
        arrowstyle='->', 
        arrowsize=arrow_size, 
        connectionstyle='arc3, rad=.1' if curved_edges else 'arc3, rad=0', 
        #node_size=vp_node_size ###################
    )
    
    # label nodes if desired
    if labels == 'text':
        for node, attributes in g.nodes(data=True):
            nx.draw_networkx_labels(
                g, 
                pos=vp_node_pos, 
                labels={node: attributes['node_label']}, 
                font_size=vp_font_size[node], 
                font_color=font_color, 
                alpha=label_transparency, 
                nodelist=[node]
            )
    if labels == 'id':
        for node, attributes in g.nodes(data=True):
            nx.draw_networkx_labels(
                g, 
                pos=vp_node_pos, 
                labels={node: str(node)}, 
                font_size=vp_font_size[node], 
                font_color=font_color, 
                alpha=label_transparency, 
                nodelist=[node]
            )
    
    plt.axis('off')
    plt.margins(margins)
    if pdf:
        plt.savefig(pdf+'.pdf')
    if png:
        plt.savefig(png+'.png')

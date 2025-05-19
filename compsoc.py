def copenhagen_collection(
    path = 'data/copenhagen/'
):
    '''
    Description: Loads the normalized Copenhagen Networks Study data collection.
    
    Input:
        path: relative directory where the data is; set to 'data/copenhagen/' by default.
    
    Output: Six dataframes in this order: users, genders, bluetooth, calls, sms, fb_friends
    '''
    # load data
    import os
    import pandas as pd
    
    attributes = pd.read_csv(os.path.join(path, 'genders.csv'))
    bluetooth = pd.read_csv(os.path.join(path, 'bt_symmetric.csv'))
    calls = pd.read_csv(os.path.join(path, 'calls.csv'))
    sms = pd.read_csv(os.path.join(path, 'sms.csv'))
    fb_friends = pd.read_csv(os.path.join(path, 'fb_friends.csv'))
    
    # create users dataframe
    import itertools
    
    users = set(itertools.chain(*[
        bluetooth['user_a'].to_list(), 
        bluetooth['user_b'].to_list(), 
        calls['caller'].to_list(), 
        calls['callee'].to_list(), 
        sms['sender'].to_list(), 
        sms['recipient'].to_list(), 
        fb_friends['# user_a'].to_list(), 
        fb_friends['user_b'].to_list(), 
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
    map = users.set_index('user').to_dict()['user_id']
    
    # create genders dataframe
    genders = pd.DataFrame([[0, 'male'], [1, 'female'], [2, 'unknown']], columns=['gender_id', 'gender'])
    
    # create bluetooth dataframe
    bluetooth = bluetooth[~bluetooth['user_b'].isin([-1, -2])]
    bluetooth = bluetooth[bluetooth['rssi'] < 0]
    bluetooth['time'] = pd.to_datetime(bluetooth['# timestamp'], unit='s')
    bluetooth['time_bin'] = (bluetooth['# timestamp'] / 300).astype(int)
    bluetooth.loc[:, 'distance'] = 10**((-75 - bluetooth['rssi']) / 10**2)
    bluetooth['user_id_from'] = bluetooth['user_a'].map(map)
    bluetooth['user_id_to'] = bluetooth['user_b'].map(map)
    bluetooth.sort_values(['time', 'user_id_from', 'user_id_to'], inplace=True)
    bluetooth.reset_index(drop=True, inplace=True)
    bluetooth = bluetooth[['user_id_from', 'user_id_to', 'rssi', 'distance', 'time', 'time_bin']]
    
    # create calls dataframe
    calls['time'] = pd.to_datetime(calls['timestamp'], unit='s')
    calls['time_bin'] = (calls['timestamp'] / 300).astype(int)
    calls['user_id_from'] = calls['caller'].map(map)
    calls['user_id_to'] = calls['callee'].map(map)
    calls['duration'].replace(-1, 0, inplace=True)
    calls.sort_values(['time', 'user_id_from', 'user_id_to'], inplace=True)
    calls.reset_index(drop=True, inplace=True)
    calls = calls[['user_id_from', 'user_id_to', 'duration', 'time', 'time_bin']]
    
    # create sms dataframe
    sms['time'] = pd.to_datetime(sms['timestamp'], unit='s')
    sms['time_bin'] = (sms['timestamp'] / 300).astype(int)
    sms['user_id_from'] = sms['sender'].map(map)
    sms['user_id_to'] = sms['recipient'].map(map)
    sms.sort_values(['time', 'user_id_from', 'user_id_to'], inplace=True)
    sms.reset_index(drop=True, inplace=True)
    sms = sms[['user_id_from', 'user_id_to', 'time', 'time_bin']]
    
    # create fb_friends dataframe
    fb_friends['user_id_from'] = fb_friends['# user_a'].map(map)
    fb_friends['user_id_to'] = fb_friends['user_b'].map(map)
    fb_friends.sort_values(['user_id_from', 'user_id_to'], inplace=True)
    fb_friends.reset_index(drop=True, inplace=True)
    
    return users, genders, bluetooth, calls, sms, fb_friends

def sns_collection(
    path = 'data/sns/'
):
    '''
    Description: Loads the normalized Social Network Science data collection.
    
    Input:
        path: relative directory where the data is; set to 'data/sns/' by default.
    
    Output: Six dataframes in this order: publications, subfields, authors, authorships, 
        words, usages
    '''
    import os
    import pandas as pd
    
    publications = pd.read_csv(os.path.join(path, 'publications.txt'), sep='\t')
    subfields = pd.read_csv(os.path.join(path, 'subfields.txt'), sep='\t')
    authors = pd.read_csv(os.path.join(path, 'authors.txt'), sep='\t')
    authorships = pd.read_csv(os.path.join(path, 'authorships.txt'), sep='\t')
    words = pd.read_csv(os.path.join(path, 'words.txt'), sep='\t')
    usages = pd.read_csv(os.path.join(path, 'usages.txt'), sep='\t')
    
    return publications, subfields, authors, authorships, words, usages

def web_browsing_collection(
    path = 'data/web_browsing/'
):
    '''
    Description: ...
    
    Input:
        path: relative directory where the data is; set to 'data/web_browsing/' by default.
    
    Output: ...
    '''
    import os
    import pandas as pd
    
    browsing = pd.read_csv(os.path.join(path, 'browsing_with_gap.csv.gz'))
    browsing['category_names_top'] = browsing['category_names_top'].str.split(',')
    panelists = pd.DataFrame(browsing['panelist_id'].drop_duplicates()).reset_index(drop=True)
    browsing['top_level_domain'] = browsing['top_level_domain'].astype('category')
    top_level_domains = pd.DataFrame(browsing['top_level_domain'].cat.categories).reset_index()
    top_level_domains.columns = ['top_level_domain_id', 'top_level_domain']
    browsing['top_level_domain'] = browsing['top_level_domain'].cat.codes
    browsing.rename(columns={'top_level_domain': 'top_level_domain_id'}, inplace=True)
    #browsing_sum = browsing[['panelist_id', 'top_level_domain_id', 'active_seconds']].groupby(['panelist_id', 'top_level_domain_id']).sum().reset_index()
    browsing_cat = browsing.copy()
    browsing_cat = browsing_cat.explode('category_names_top').reset_index(drop=True)
    browsing_cat.rename(columns={'category_names_top': 'category_name_top'}, inplace=True)
    browsing_cat['category_name_top'] = browsing_cat['category_name_top'].astype('category')
    category_names_top = pd.DataFrame(browsing_cat['category_name_top'].cat.categories).reset_index()
    category_names_top.columns = ['category_name_top_id', 'category_name_top']
    browsing_cat['category_name_top'] = browsing_cat['category_name_top'].cat.codes
    browsing_cat.rename(columns={'category_name_top': 'category_name_top_id'}, inplace=True)
    #browsing_cat_sum = browsing_cat[['panelist_id', 'top_level_domain_id', 'category_name_top_id', 'active_seconds']].groupby(['panelist_id', 'top_level_domain_id', 'category_name_top_id']).sum().reset_index()
    
    return panelists, browsing_cat, top_level_domains, category_names_top

def construct_graph(
    node_list, 
    node_identifier, 
    edge_list, 
    directed = True, 
    graph_name = None, 
    node_properties = None, 
    node_property_types = None, 
    edge_properties = None, 
    edge_property_types = None
):
    '''
    Description: Constructs a graph, using graph-tool, from pandas node and edge lists.
    
    Inputs:
        node_list: pandas dataframe containing the node identifiers and properties; must 
            contain a continuous identifier from 0 to N-1 where N is the number of vertices; 
            node attributes must be in additional columns.
        node_identifier: Column name of the node list that contains the node identifier.
        edge_list: pandas dataframe containing a node identifier pair per edge; edge 
            attributes must be in additional columns.
        directed: If graph should be directed (boolean); set to True by default.
        graph_name: Name of the graph (string); set to None by default.
        node_properties: List of column names of the node list that contains node attributes 
            that are to be internalized in the graph; set to None by default.
        node_property_types: List of data types corresponding to the node properties list; 
            values must be from the set given at 
            https://graph-tool.skewed.de/static/doc/quickstart.html#property-maps; set to None 
            by default.
        edge_properties: List of column names of the edge_list that contains edge attributes 
            that are to be internalized in the graph; set to None by default.
        edge_property_types: List of data types corresponding to the edge properties list; 
            values must be from the set given at 
            https://graph-tool.skewed.de/static/doc/quickstart.html#property-maps; set to None 
            by default.
    
    Output: graph-tool graph object, with internal graph, vertex, and edge attributes, if 
        specified.
    '''
    # node list check
    node_list = node_list.sort_values(node_identifier, ascending=True).reset_index(drop=True)
    check = node_list[node_identifier] == node_list.index
    if (sum(check) == len(node_list)) == False:
        print('The node list dataframe does not meet the requirements. Check if the'+node_identifier+'column contains integers from zero to N-1 where N is the number of vertices in the graph.')
    
    # create graph
    import graph_tool.all as gt
    g = gt.Graph(directed=directed)
    
    if graph_name != None:
        # write graph name
        g.gp['name'] = g.new_gp('string')
        g.gp['name'] = graph_name
    
    # write vertices
    g.add_vertex(len(node_list))
    
    if node_properties != None:
        # create i internal vertex property maps
        for i in range(len(node_properties)):
            g.vp[node_properties[i]] = g.new_vp(node_property_types[i])
        
        # write scalar vertex attributes
        scalar_types = ['bool', 'uint8_t', 'int16_t', 'short', 'int32_t', 'int', 'int64_t', 'long', 'long long', 'double', 'float', 'long double']
        node_property_types_indices_scalar = [i for i, j in enumerate(node_property_types) if j in scalar_types]
        for i in node_property_types_indices_scalar:
            g.vp[node_properties[i]].a = node_list[node_properties[i]]
        
        # write non-scalar vertex attributes
        node_property_types_indices_nonscalar = [i for i, j in enumerate(node_property_types) if j not in scalar_types]
        for v in range(len(node_list)):
            for i in node_property_types_indices_nonscalar:
                g.vp[node_properties[i]][v] = node_list[node_properties[i]][v]
    
    columns = list(edge_list.columns[:2])
    if edge_properties != None:
        # create i internal edge property maps
        for i in range(len(edge_properties)):
            g.ep[edge_properties[i]] = g.new_ep(edge_property_types[i])
        
        # write edges with attributes 
        [columns.append(edge_property) for edge_property in edge_properties]
        eprops = [g.ep[edge_properties[i]] for i in range(len(edge_properties))]
        g.add_edge_list(edge_list[columns].values, eprops=eprops)
    else:
        # write edges
        g.add_edge_list(edge_list[columns].values)
    
    return g

def co_occurrence(
    node_list_u, 
    node_list_v, 
    edge_list, 
    node_u_identifier, 
    node_v_identifier, 
    weight, 
    norm, 
    category_identifier = None, 
    directed = False, 
    remove_self_loops = True, 
    enrich_node_list_v = False
):
    '''
    Description: Constructs a unipartite co-occurrence matrix from pandas node lists 
        representing the two sets U and V and an edge list E representing a bipartite 
        occurrence matrix by projecting E to the V side, possibly using normalization.
    
    Inputs:
        node_list_u: pandas dataframe containing the identifiers and properties of node in 
            the set U; must contain a continuous identifier from 0 to N-1 where N is the 
            number of vertices; node attributes must be in additional columns.
        node_list_v: pandas dataframe containing the identifiers and properties of node in 
            the set V; must contain a continuous identifier from 0 to N-1 where N is the 
            number of vertices; node attributes must be in additional columns.
        edge_list: pandas dataframe representing the occurrence matrix; must contain a U 
            and V node identifier pair per occurrence or edge.
        node_u_identifier: Column name of the U node list that contains the node identifier.
        node_v_identifier: Column name of the V node list that contains the node identifier.
        weight: Column name of the edge list that contains the weight of an occurrence; 
            if None, it is set to 1.
        norm: Kind of normalization to be used in projecting the co-occurrence matrix; 
            if None, no normalization is used; if 'partial', partial row normalization is 
            used; if 'full', full row normalization is used.
        category_identifier: Column name of the edge list that tells which category an 
            occurrence belongs to; set to None by default.
        directed: If co-occurrence should be directed (boolean); set to False by default.
        remove_self_loops: If values in the diagonal of the co-occurrence matrix should 
            be removed (boolean); set to True by default.
        enrich_node_list_v: If node attributes from matrix projection (e.g., occurrence, 
            self-occurrence) should be stored in the node list representing V; set to False 
            by default.
    
    Output: Unipartite matrix in form of a pandas dataframe or edge list giving weighted 
        co-occurrences.
    '''
    import pandas as pd
    from scipy.sparse import coo_matrix, csr_matrix
    
    if category_identifier:
        import compsoc as cs
        
        # call compsoc function
        if enrich_node_list_v:
            print("Node list can't be enriched because co-occurrences are computed for categories.")
        categories = edge_list[category_identifier].drop_duplicates().sort_values().tolist()
        edge_list_v = pd.DataFrame(columns=[node_v_identifier+'_i', node_v_identifier+'_j', category_identifier, 'weight', 'cumfrac'], dtype='int')
        for category in categories:
            edge_list_v_category = cs.co_occurrence(node_list_u=node_list_u, node_list_v=node_list_v, edge_list=edge_list[edge_list[category_identifier]==category], node_u_identifier=node_u_identifier, node_v_identifier=node_v_identifier, category_identifier=None, weight=weight, norm=norm, directed=directed, remove_self_loops=remove_self_loops, enrich_node_list_v=False)
            edge_list_v_category.loc[:, category_identifier] = category
            edge_list_v_category = edge_list_v_category[[node_v_identifier+'_i', node_v_identifier+'_j', category_identifier, 'weight', 'cumfrac']]
            edge_list_v = pd.concat([edge_list_v, edge_list_v_category])
    else:
        # node list checks
        node_list_u = node_list_u.sort_values(node_u_identifier, ascending=True).reset_index(drop=True)
        node_list_v = node_list_v.sort_values(node_v_identifier, ascending=True).reset_index(drop=True)
        check_u = node_list_u[node_u_identifier] == node_list_u.index
        check_v = node_list_v[node_v_identifier] == node_list_v.index
        if (sum(check_u) == len(node_list_u)) == False:
            print('The node list dataframe of set U does not meet the requirements. Check if the'+node_identifier+'column contains integers from zero to N-1 where N is the number of vertices in the graph.')
        if (sum(check_v) == len(node_list_v)) == False:
            print('The node list dataframe of set V does not meet the requirements. Check if the'+node_identifier+'column contains integers from zero to N-1 where N is the number of vertices in the graph.')
        
        # construct occurrence matrix
        rows = edge_list[node_u_identifier].tolist()
        columns = edge_list[node_v_identifier].tolist()
        if weight:
            weights = edge_list[weight].tolist()
        else:
            weights = [1] * len(edge_list)
        B = coo_matrix((weights, (rows, columns)), shape=(len(node_list_u), len(node_list_v))).tocsr()
        BT = csr_matrix.transpose(B)
        
        # construct co-occurrence matrix
        if norm == None:
            G = (BT * B).tocsr()
            if enrich_node_list_v:
                print('Enriching...')
                node_list_v.loc[:, 'occurrence'] = G.diagonal().tolist()
        elif norm == 'partial':
            from sklearn.preprocessing import normalize
            BN = normalize(B, norm='l1', axis=1)
            G = (BT * BN).tocsr()
            if enrich_node_list_v:
                print('Enriching...')
                node_list_v.loc[:, 'occurrence'] = G.sum(axis=0).tolist()[0]
                node_list_v.loc[:, 'self-occurrence'] = G.diagonal().tolist()
                node_list_v.loc[:, 'self-sufficiency'] = node_list_v['self-occurrence'] / node_list_v['occurrence']
                node_list_v.loc[:, 'embeddedness'] = 1 - node_list_v['self-sufficiency']
        elif norm == 'full':
            from sklearn.preprocessing import normalize
            BN = normalize(B, norm='l1', axis=1)
            BNT = csr_matrix.transpose(BN)
            G = (BNT * BN).tocsr()
            if enrich_node_list_v:
                print('Enriching...')
                node_list_v.loc[:, 'self-occurrence'] = G.sum(axis=0).tolist()[0]
        
        # construct co-occurrence edge list with cumulative fractions
        edge_list_v = pd.concat([pd.Series(G.nonzero()[0]), pd.Series(G.nonzero()[1]), pd.Series(G.data)], axis=1)
        edge_list_v.columns = [node_v_identifier+'_i', node_v_identifier+'_j', 'weight']
        if directed == False:
            edge_list_v = edge_list_v[edge_list_v[node_v_identifier+'_j'] >= edge_list_v[node_v_identifier+'_i']]
        if remove_self_loops == True:
            edge_list_v = edge_list_v[edge_list_v[node_v_identifier+'_j'] != edge_list_v[node_v_identifier+'_i']]
        edge_list_v_cumfrac = edge_list_v['weight'].groupby(edge_list_v['weight']).sum().sort_index(ascending=False)
        edge_list_v_cumfrac = (edge_list_v_cumfrac.cumsum() / sum(edge_list_v_cumfrac)).round(6)
        edge_list_v_cumfrac.rename('cumfrac', inplace=True)
        edge_list_v = pd.merge(left=edge_list_v, right=edge_list_v_cumfrac, left_on='weight', right_on=edge_list_v_cumfrac.index)
    
    return edge_list_v
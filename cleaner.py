import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score


def get_distinct_colors(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i) for i in range(n)]


def get_community_dict(G):
    communities = greedy_modularity_communities(G, resolution=1)

    node_to_community = {}
    for community_id, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_id
    return node_to_community


def generate_negative_samples(graph, num_samples):
    negative_samples = []
    nodes = list(graph.nodes)
    while len(negative_samples) < num_samples:
        node1, node2 = random.sample(nodes, 2)
        if not graph.has_edge(node1, node2):
            negative_samples.append((node1, node2))
    return pd.DataFrame(negative_samples, columns=['i', 'j'])


def getFeatures(G, df, features=None):
    """
    Compute a variety of features for multiple node pairs in the graph G at once, including community and attribute
    one-hot encoding if specified, and return a DataFrame containing only the requested features.

    Parameters:
    G (networkx.Graph): The input graph with nodes and edges.
    df (pd.DataFrame): A DataFrame containing the node pairs with columns 'i' and 'j'.
    com_dict (dict): Dictionary mapping nodes to their community IDs (optional).
    features (list): A list of feature names to include in the output. If None, returns all features.

    Returns:
    pd.DataFrame: A DataFrame containing only the specified features for all node pairs.
    """

    df = df.copy()
    # Ensure the DataFrame has the required 'i' and 'j' columns
    if not {'i', 'j'}.issubset(df.columns):
        raise ValueError("Input DataFrame must have 'i' and 'j' columns representing node pairs.")

    com_dict = get_community_dict(G)

    node_pairs = [(i, j) for (i, j) in zip(df['i'], df['j'])]
    # Calculate features based on the requested list
    if features is None or 'degree_i' in features:
        df['degree_i'] = [G.degree(node) for node in df['i']]

    if features is None or 'degree_j' in features:
        df['degree_j'] = [G.degree(node) for node in df['j']]

    if features is None or 'pa_score' in features:
        df['pa_score'] = [score[2] for score in list(nx.preferential_attachment(G, node_pairs))]

    if features is None or 'common_neighbors' in features:
        df['common_neighbors'] = df.apply(lambda row: len(list(nx.common_neighbors(G, row['i'], row['j']))), axis=1)

    if features is None or 'jaccard_score' in features:
        df['jaccard_score'] = [score[2] for score in list(nx.jaccard_coefficient(G, node_pairs))]

    if features is None or 'adamic_adar_score' in features:
        df['adamic_adar_score'] = [score[2] for score in nx.adamic_adar_index(G, node_pairs)]

    if features is None or 'resource_allocation_score' in features:
        df['resource_allocation_score'] = [score[2] for score in nx.resource_allocation_index(G, node_pairs)]
    if features is None or 'triadic_score' in features:
        df['triadic_score'] = [len(set(G.neighbors(i)).intersection(set(G.neighbors(j)))) for (i, j) in node_pairs]
    if features is None or 'attribute_i' in features:
        df['attribute_i'] = df['i'].map(lambda node: G.nodes[node].get('attribute'))
    if features is None or 'attribute_j' in features:
        df['attribute_j'] = df['j'].map(lambda node: G.nodes[node].get('attribute'))
    if features is None or 'attribute_score' in features:
        df['attribute_score'] = (df['i'].map(lambda node: G.nodes[node].get('attribute')) == df['j'].map(
            lambda node: G.nodes[node].get('attribute'))).astype(int)
    if features is None or 'community_score' in features:
        comm_i = df['i'].map(com_dict)
        comm_j = df['j'].map(com_dict)
        df['community_score'] = (comm_i == comm_j).astype(int)
    if features is None or 'soundarajan':
        df['soundarajan'] = [score[2] for score in
                             list(nx.cn_soundarajan_hopcroft(G, node_pairs, community='community'))]

    if 'i' and 'j' not in features:
        df.drop(columns=['i', 'j'], inplace=True)

    df = df.select_dtypes(include=['int64', 'float64'])
    return df


def preprocess(G, positive, negative=None, training=False, features=None):
    X = []
    y = []

    if negative is not None and training is True:
        print('preprocessing training')
        # Add positive examples where edge exist
        X_positive = getFeatures(G, positive, features)
        X_negative = getFeatures(G, negative, features)

        y_positive = pd.Series([1] * X_positive.shape[0], name="link")
        y_negative = pd.Series([0] * X_negative.shape[0], name="link")

        X = pd.concat([X_positive, X_negative], axis=0)
        y = pd.concat([y_positive, y_negative], axis=0)


    elif training is False:
        print('preprocessing testing')
        X = getFeatures(G, positive, features)

    if training:
        return X, y
    else:
        return X


def split_graph(G, train_size=0.8):
    edges = list(G.edges())
    n_edges_remove = int((1 - train_size) * len(edges))
    remove_edges = random.sample(edges, n_edges_remove)

    G_train = G.copy()
    G_train.remove_edges_from(remove_edges)

    return G_train, remove_edges


def evaluate_model(model, X_test, y_test, X_train, y_train):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print("\nBest model performance:")

    scores_train = cross_val_score(model, X_train, y_train, cv=5)
    print("%0.3f Train accuracy with a standard deviation of %0.2f" % (scores_train.mean(), scores_train.std()))

    scores_test = cross_val_score(model, X_test, y_test, cv=5)
    print("%0.3f Test accuracy with a standard deviation of %0.2f" % (scores_test.mean(), scores_test.std()))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    cm_disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Reds', normalize='true')
    return roc_disp, cm_disp


def print_feature_importances(model, feature_names):
    """
    Prints the feature importances of a trained RandomForest model.

    Parameters:
    model : Trained RandomForest model
    feature_names : List of feature names (column names)
    """
    # Get feature importances from the model
    importances = model.feature_importances_

    # Create a dataframe for better visualization
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Print the results
    print(feature_importances)
import argparse
from utils import *
from models import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file",type=str,default="dm_test.sdp")
    args = parser.parse_args()

    nodes = get_nodes_for_singeleton_classifier("dm.sdp")
    SingleClassifier =  SingletonClassifier(nodes)
    SingleClassifier.load_from("singletonCls-123.pkl")

    edges = get_edges_for_edge_classifier("dm.sdp")
    #EdgeClassifier = EdgeEnsembleClassifier(edges,["edgeCls999.pkl"]) 
    EdgeClassifier = EdgeEnsembleClassifier(edges,["edgeCls123.pkl","edgeCls666.pkl","edgeCls789.pkl","edgeCls888.pkl","edgeCls999.pkl"]) 
    samples,tokens,lemmas,pos_tags,relations = read_raw_samples("dm_test.sdp")
    target_samples = getAnnotationsForSamples(samples,SingleClassifier,EdgeClassifier)
    get_metrics_from_samples(target_samples,samples)
    


    

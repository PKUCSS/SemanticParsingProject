import pickle
import numpy
import copy
import random
import matplotlib.pyplot as plt 
from tqdm import tqdm
from models import *

def towID2key(a,b):
    return "#".join([str(a),str(b)])

def key2towID(s):
    x,y = s.split("#")
    return int(x),int(y)


def get_sample_idlist(test_file):
    lis = []
    with open(test_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                lis.append(line.strip())
    return lis 

def write_results(samples,test_file,target_file,output_lemmas=False):
    id_list = get_sample_idlist(test_file)
    with open(target_file,'w') as fw:
        for i,id in enumerate(id_list):
            sample = samples[id]
            fw.write(str(id)+'\n')
            tokens = sample['tokens']
            lemmas = sample['lemmas']
            if output_lemmas == False:
                lemmas = ['-' for i in range(len(tokens))]
            pos_tags = sample['pos_tags']
            preds = sample['preds']
            tops = sample['tops']
            edges = sample['edges']
            pred_cnt = 0 
            pred_ids = []
            for j,p in enumerate(preds):
                if p == '+':
                    pred_cnt += 1
                    pred_ids.append(j)
            for i in range(len(tokens)):
                fw.write("{}\t{}\t{}\t{}\t{}\t{}".format(i+1,tokens[i],lemmas[i],pos_tags[i],tops[i],preds[i]))
                for j in range(pred_cnt):
                    fw.write("\t{}".format(edges.get(towID2key(pred_ids[j],i),'_')))
                fw.write("\n")
            fw.write("\n")
    return 

         

def get_metrics_from_samples(target_samples,golden_samples):
    target_edges_cnt = 0
    golden_edges_cnt = 0
    common_edges_cnt = 0
    graph_cnt = len(golden_samples)
    exact_match = 0 
    for sample_id in golden_samples.keys():
        target_edges = target_samples[sample_id]['edges']
        golden_edges = golden_samples[sample_id]['edges']
        target_edges_cnt += len(target_edges)
        golden_edges_cnt += len(golden_edges)
        tmp = 0
        for edge_id in golden_edges.keys():
            if golden_edges[edge_id] == target_edges.get(edge_id,"xxx"):
                tmp += 1
        common_edges_cnt += tmp
        if tmp == len(golden_edges) and tmp == len(target_edges):
            exact_match +=1 
    precision = common_edges_cnt/target_edges_cnt
    recall = common_edges_cnt/golden_edges_cnt
    F1 = 2.0*precision*recall/(precision+recall)
    match_ratio = exact_match/graph_cnt
    print("precision: {},recall: {},F1: {},exact match:{}".format(precision,recall,F1,match_ratio))
    return {"precision":precision,"recall":recall,"F1":F1,"exact_match":match_ratio}




def read_raw_samples(sdp_filename):
    print("Getting samples from {} ...".format(sdp_filename))
    samples = {}
    tokens = set()
    lemmas = set()
    pos_tags = set()
    Relations = set()
    Relations.add("NoEdge")
    with open(sdp_filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        line_idx = 0
        while line_idx < len(lines):
            s = line_idx 
            while  line_idx < len(lines) and lines[line_idx] != '\n':
                line_idx += 1
            e = line_idx 
            if e >= len(lines):
                break
            if lines[line_idx] == "\n":
                line_idx += 1    
            sample_id = lines[s].strip()
            tmp_lines = lines[s+1:e]
            pred_num = 0
            pred2linenum = {}
            for idx,line in enumerate(tmp_lines):
                id,word,lemma,pos,top,pred = line.split("\t")[:6]
                tokens.add(word)
                lemmas.add(lemma)
                pos_tags.add(pos)
                if pred == "+":
                    pred2linenum[pred_num] = idx
                    pred_num += 1
            sample = {'id':sample_id,'len':len(tmp_lines),"tokens":[],"lemmas":[],"pos_tags":[],"tops":[],"preds":[],"edges":{}}
            for i,line in enumerate(tmp_lines):
                # print(pred_num)
                # print(line)
                line = line.strip()
                assert len(line.split("\t")) == 6 + pred_num
                id,word,lemma,pos,top,pred = line.split("\t")[:6]
                sample["tokens"].append(word)
                sample["lemmas"].append(lemma)
                sample["pos_tags"].append(pos)
                sample["tops"].append(top)
                sample["preds"].append(pred)
                relations = line.split("\t")[6:]
                for j,rel in enumerate(relations):
                    if rel != "_":
                        Relations.add(rel)
                        sample["edges"][towID2key(pred2linenum[j],i)] = rel
            samples[sample_id] = sample
        return samples,tokens,lemmas,pos_tags,Relations



def draw_edge_distribution(sdp_filename):
    samples,tokens,lemmas,pos_tags,relations = read_raw_samples(sdp_filename)
    edge_lens = []
    for sample in samples.values():
        for edge_id in sample['edges'].keys():
            x,y = key2towID(edge_id)
            edge_lens.append(abs(y-x))
    plt.hist(edge_lens,bins=30,range=(0,30))
    plt.xlabel("The distance of two tokens in golden edges")
    plt.ylabel("Number")
    plt.savefig("draw_edge_distribution.png")
    plt.show()


def get_nodes_for_singeleton_classifier(sdp_filename):
    samples,tokens,lemmas,pos_tags,relations = read_raw_samples(sdp_filename)
    nodes = []
    for sample in samples.values():
        edge_ids = set()
        for edge_id in sample['edges'].keys():
            x,y = key2towID(edge_id)
            edge_ids.add(x)
            edge_ids.add(y)
        tokens = sample['tokens']
        pos_tags = sample['pos_tags']
        lemmas = sample['lemmas']
        tops = sample['tops']
        for i in range(len(tokens)):
            if tops[i] == '+':
                top = 1
            else:
                top = 0
            if i not in edge_ids:
                nodes.append({'token':tokens[i], 'pos':pos_tags[i],'lemma':lemmas[i],'single':0,'top':top})
            else:
                nodes.append({'token':tokens[i], 'pos':pos_tags[i],'lemma':lemmas[i],'single':1,'top':top})
    return nodes 
        
 
def test_singleton_exp(train_sdp_filename="dm.sdp",test_sdp_filename="dm_test.sdp"):
    train_nodes = get_nodes_for_singeleton_classifier(train_sdp_filename)
    classifier =  SingletonClassifier(train_nodes)
    test_nodes = get_nodes_for_singeleton_classifier(test_sdp_filename)
    best_acc = 0
    random.seed(123)
    for i in range(10):
        print("Epoch:{}".format(i))
        random.shuffle(train_nodes)
        classifier.train(train_nodes,batch_size=1000)
        print("training accuracy: {}".format(classifier.test(train_nodes)))
        test_acc = classifier.test(test_nodes)
        print("test accuracy: {}".format(test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            classifier.save("singletonCls-123.pkl")
    return classifier


def get_edges_for_edge_classifier(sdp_filename="dm.sdp",singleton_classifier_path="singletonCls-123.pkl"):
    nodes = get_nodes_for_singeleton_classifier(sdp_filename)
    classifier =  SingletonClassifier(nodes)
    classifier.load_from(singleton_classifier_path)
    samples,tokens,lemmas,pos_tags,relations = read_raw_samples(sdp_filename)
    Edges = []
    for sample in samples.values():
        tokens = sample['tokens']
        pos_tags = sample['pos_tags']
        lemmas = sample['lemmas']
        edges = sample['edges']
        nodes = [{'token':tokens[i], 'pos':pos_tags[i],"lemma":lemmas[i]} for i in range(len(tokens))]
        isSingletons = classifier.predict(nodes)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if abs(i-j) > 10 or i == j: 
                    continue
                if isSingletons[i] == 0 or isSingletons[j] == 0:
                    continue 
                edge_id = towID2key(i,j)
                rel = edges.get(edge_id,"_")
                neighbor_pos_tags = []
                if i == 0:
                    neighbor_pos_tags.append("Left")
                else:
                    neighbor_pos_tags.append(pos_tags[i-1])
                if i + 1 >= len(tokens):
                    neighbor_pos_tags.append("Right")
                else:
                    neighbor_pos_tags.append(pos_tags[i+1])
                if j == 0:
                    neighbor_pos_tags.append("Left")
                else:
                    neighbor_pos_tags.append(pos_tags[j-1])
                if j + 1 >= len(tokens):
                    neighbor_pos_tags.append("Right")
                else:
                    neighbor_pos_tags.append(pos_tags[j+1])
                Edges.append({"tokens":[tokens[i],tokens[j]],
                            "pos_tags":[pos_tags[i],pos_tags[j]],
                            "lemmas":[lemmas[i],lemmas[j]],
                            "idxs":[i,j],
                            "neighbor_pos_tags":neighbor_pos_tags,
                            "rel":rel }
                        )
    return Edges
                


def train_edge_classifier(save_path="edgeCls.pkl",seed=123):
    edges = get_edges_for_edge_classifier()
    print("Total Edge Numder:",len(edges))
    test_edges = get_edges_for_edge_classifier("dm_test.sdp")
    cnt = 0
    for edge in edges:
        if edge['rel'] != "_":
            cnt += 1
    print(cnt/len(edges))
    classifier = EdgeClassifier(edges)
    random.seed(seed)
    random.shuffle(edges)
    train_edges = edges 
    best_acc = 0.0
    for i in range(10):
        print("Epoch:{}".format(i))
        random.shuffle(train_edges)
        classifier.train(train_edges,batch_size=100000)
        print("training accuracy: {}".format(classifier.test(train_edges)))
        test_acc = classifier.test(test_edges)
        print("test accuracy: {}".format(test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            classifier.save(save_path)
    return classifier
        



class Lemmaziation(object):
    def __init__(self,sdp_filename,lemma_filename):
        dic = {}
        nodes = get_nodes_for_singeleton_classifier(sdp_filename)
        for node in nodes:
            dic[node['token']] = node['lemma']
        with open('lemma.txt','r') as f:
            lines = f.readlines()
            for line in lines:
                word1,word2 = line.split()
                word2 = word2.strip()
                if word2 not in dic.keys():
                    dic[word2] = word1
        self.dic = dic 
    
    def get_lemma(self,token):
        return self.dic.get(token,token)

lemma_tool = Lemmaziation("dm.sdp","lemma.txt")

def read_raw_samples_from_test_data(sdp_filename):
    print("Getting samples from {} ...".format(sdp_filename))
    samples = {}
    tokens = set()
    lemmas = set()
    pos_tags = set()
    with open(sdp_filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        line_idx = 0
        while line_idx < len(lines):
            s = line_idx 
            while lines[line_idx] != '\n' and line_idx < len(lines):
                line_idx += 1
            e = line_idx 
            if lines[line_idx] == "\n":
                line_idx += 1    
            sample_id = lines[s].strip()
            tmp_lines = lines[s+1:e]
            pred_num = 0
            pred2linenum = {}
            for idx,line in enumerate(tmp_lines):
                id,word,lemma,pos = line.split("\t")[:4]
                tokens.add(word)
                lemmas.add(lemma)
                pos_tags.add(pos)
            sample = {'id':sample_id,'len':len(tmp_lines),"tokens":[],"lemmas":[],"pos_tags":[]}
            for i,line in enumerate(tmp_lines):
                # print(pred_num)
                # print(line)
                line = line.strip()
                id,word,lemma,pos = line.split("\t")[:4]
                sample["tokens"].append(word)
                sample["lemmas"].append(lemma_tool.get_lemma(word))
                sample["pos_tags"].append(pos)
            samples[sample_id] = sample
        return samples,tokens,lemmas,pos_tags



def getAnnotationsForSamples(samples,SingletonClassifier,EdgeClassifier,TopClassifier=None):
    annotations = {}
    id2class = EdgeClassifier.id2class
    for sample_id in tqdm(samples.keys()):
        sample = samples[sample_id]
        tokens = sample['tokens']
        pos_tags = sample['pos_tags']
        lemmas = sample['lemmas']
        edges = sample['edges']
        nodes = [{'token':tokens[i], 'pos':pos_tags[i],"lemma":lemmas[i]} for i in range(len(tokens))]
        isSingletons = SingletonClassifier.predict(nodes)
        tmp_edges = {}
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if abs(i-j) > 10 or i == j: 
                    continue
                if isSingletons[i] == 0 or isSingletons[j] == 0:
                    continue 
                edge_id = towID2key(i,j)
                rel = edges.get(edge_id,"_")
                neighbor_pos_tags = []
                if i == 0:
                    neighbor_pos_tags.append("Left")
                else:
                    neighbor_pos_tags.append(pos_tags[i-1])
                if i + 1 >= len(tokens):
                    neighbor_pos_tags.append("Right")
                else:
                    neighbor_pos_tags.append(pos_tags[i+1])
                if j == 0:
                    neighbor_pos_tags.append("Left")
                else:
                    neighbor_pos_tags.append(pos_tags[j-1])
                if j + 1 >= len(tokens):
                    neighbor_pos_tags.append("Right")
                else:
                    neighbor_pos_tags.append(pos_tags[j+1])
                tmp_edges[edge_id] = {"tokens":[tokens[i],tokens[j]],
                            "pos_tags":[pos_tags[i],pos_tags[j]],
                            "lemmas":[lemmas[i],lemmas[j]],
                            "idxs":[i,j],
                            "neighbor_pos_tags":neighbor_pos_tags,
                             }
        tmp_sample = copy.deepcopy(sample)
        tmp_sample['edges'] = {}

        for edge_id in tmp_edges.keys():
            tmp_edge = [0]
            #print("#####",edge_id,tmp_edges[edge_id])
            tmp_edge[0] = tmp_edges[edge_id]
            res = id2class[EdgeClassifier.predict(tmp_edge)[0]]
            if  res != '_':
                tmp_sample['edges'][edge_id] = res
        annotations[sample_id] = tmp_sample
    return annotations


def get_nodes_for_top_classifier(sdp_filename,SingletonClassifier,EdgeClassifier):
    samples,tokens,lemmas,pos_tags,relations = read_raw_samples(sdp_filename)
    samples = getAnnotationsForSamples(samples,SingletonClassifier,EdgeClassifier)
    nodes = []
    for sample in samples.values():
        incoming_edge_ids = set()
        outcoming_edge_ids = set()
        edge_ids = set()
        for edge_id in sample['edges'].keys():
            x,y = key2towID(edge_id)
            outcoming_edge_ids.add(x)
            incoming_edge_ids.add(y)
            edge_ids.add(x)
            edge_ids.add(y)
        tokens = sample['tokens']
        pos_tags = sample['pos_tags']
        lemmas = sample['lemmas']
        tops = sample['tops']
        for i in range(len(tokens)):
            if tops[i] == '+':
                top = 1
            else:
                top = 0
            if i not in outcoming_edge_ids:
                continue
            if i not in incoming_edge_ids:
                nodes.append({'index':i,'token':tokens[i], 'pos':pos_tags[i],'lemma':lemmas[i],'top':top,'root':0})
            else:
                nodes.append({'index':i,'token':tokens[i], 'pos':pos_tags[i],'lemma':lemmas[i],'top':top,'root':1})
    return nodes 


def test_top_exp(train_sdp_filename="dm.sdp",test_sdp_filename="dm_test.sdp"):
    nodes = get_nodes_for_singeleton_classifier(train_sdp_filename)
    SingleClassifier =  SingletonClassifier(nodes)
    SingleClassifier.load_from("singletonCls-123.pkl")
    edges = get_edges_for_edge_classifier(train_sdp_filename)
    EdgeClassifier = EdgeEnsembleClassifier(edges,["edgeCls123.pkl","edgeCls666.pkl","edgeCls789.pkl","edgeCls888.pkl","edgeCls999.pkl"]) 
    train_nodes = get_nodes_for_top_classifier(train_sdp_filename,SingleClassifier,EdgeClassifier)
    classifier =  TopClassifier(train_nodes)
    test_nodes = get_nodes_for_top_classifier(test_sdp_filename,SingleClassifier,EdgeClassifier)
    best_acc = 0
    random.seed(123)
    for i in range(10):
        print("Epoch:{}".format(i))
        random.shuffle(train_nodes)
        classifier.train(train_nodes,batch_size=1000)
        print("training accuracy: {}".format(classifier.test(train_nodes)))
        test_acc = classifier.test(test_nodes)
        print("test accuracy: {}".format(test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            classifier.save("topCls-123.pkl")
    return classifier

def getAnnotationsForTestSamples(samples,SingletonClassifier,EdgeClassifier,TopClassifier):
    annotations = {}
    id2class = EdgeClassifier.id2class
    for sample_id in tqdm(samples.keys()):
        sample = samples[sample_id]
        tokens = sample['tokens']
        pos_tags = sample['pos_tags']
        lemmas = sample['lemmas']
        nodes = [{'token':tokens[i], 'pos':pos_tags[i],"lemma":lemmas[i]} for i in range(len(tokens))]
        isSingletons = SingletonClassifier.predict(nodes)
        tmp_edges = {}
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if abs(i-j) > 10 or i == j: 
                    continue
                if isSingletons[i] == 0 or isSingletons[j] == 0:
                    continue 
                edge_id = towID2key(i,j)
                neighbor_pos_tags = []
                if i == 0:
                    neighbor_pos_tags.append("Left")
                else:
                    neighbor_pos_tags.append(pos_tags[i-1])
                if i + 1 >= len(tokens):
                    neighbor_pos_tags.append("Right")
                else:
                    neighbor_pos_tags.append(pos_tags[i+1])
                if j == 0:
                    neighbor_pos_tags.append("Left")
                else:
                    neighbor_pos_tags.append(pos_tags[j-1])
                if j + 1 >= len(tokens):
                    neighbor_pos_tags.append("Right")
                else:
                    neighbor_pos_tags.append(pos_tags[j+1])
                tmp_edges[edge_id] = {"tokens":[tokens[i],tokens[j]],
                            "pos_tags":[pos_tags[i],pos_tags[j]],
                            "lemmas":[lemmas[i],lemmas[j]],
                            "idxs":[i,j],
                            "neighbor_pos_tags":neighbor_pos_tags,
                            }
        tmp_sample = copy.deepcopy(sample)
        tmp_sample['edges'] = {}

        for edge_id in tmp_edges.keys():
            tmp_edge = [0]
            #print("#####",edge_id,tmp_edges[edge_id])
            tmp_edge[0] = tmp_edges[edge_id]
            res = id2class[EdgeClassifier.predict(tmp_edge)[0]]
            if  res != '_':
                tmp_sample['edges'][edge_id] = res
        preds = ['-' for i in range(len(tmp_sample['tokens']))]
        outcoming_edge_ids = set()
        incoming_edge_ids = set()
        for edge_id in tmp_sample['edges'].keys():
            x,y = key2towID(edge_id)
            outcoming_edge_ids.add(x)
            incoming_edge_ids.add(y)
            preds[x] = '+'
        tops = ['-' for i in range(len(tmp_sample['tokens']))]
        max_top_score = -1e10
        top_position = -1
        for i in range(len(tokens)):
            if i not in outcoming_edge_ids or isSingletons[i] == 0:
                continue
            node = {}
            node['token'] = tokens[i]
            node['pos'] = pos_tags[i]
            node['lemma'] = lemmas[i]
            node['index'] = i 
            if i in incoming_edge_ids:
                node['root'] = 0
            else:
                node['root'] = 1
            label,score  = TopClassifier.predict_one_node(node)
            if type(score) == list:
                score = score[0]
            if score > max_top_score:
                max_top_score = score
                top_position = i 
        
        if top_position != -1:
            tops[top_position] = '+'
        tmp_sample['tops'] = tops 
        tmp_sample['preds'] = preds
        annotations[sample_id] = tmp_sample
    return annotations

def make_final_submission(train_sdp_filename="dm.sdp",test_file_name="test_data/esl.input",output_path="test_results.sdp"):
    nodes = get_nodes_for_singeleton_classifier(train_sdp_filename)
    SingleClassifier =  SingletonClassifier(nodes)
    SingleClassifier.load_from("singletonCls-123.pkl")
    edges = get_edges_for_edge_classifier(train_sdp_filename)
    EdgeClassifier = EdgeEnsembleClassifier(edges,["edgeCls123.pkl","edgeCls666.pkl","edgeCls789.pkl","edgeCls888.pkl","edgeCls999.pkl"]) 
    #nodes = get_nodes_for_top_classifier(train_sdp_filename,SingleClassifier,EdgeClassifier)
    topClassifier = TopClassifier([])
    topClassifier.load_from("topCls-123.pkl")
    test_samples,tokens,lemmas,pos_tags = read_raw_samples_from_test_data(test_file_name)
    annotations = getAnnotationsForTestSamples(test_samples,SingleClassifier,EdgeClassifier,topClassifier)
    write_results(annotations,test_file_name,output_path)
    return 

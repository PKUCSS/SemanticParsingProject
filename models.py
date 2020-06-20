import math
import random
import pickle
import numpy as np
from tqdm import tqdm
from scipy.special import expit



# I reused the log-linear mdoel implemented in Project 1 as the backbone classifier


class LogLinearModel:
    '''
    The Log-Linear Model for text categorization
    '''
    def __init__(self,lr=0.1,n_classes=2,n_features=20000,lemma =1e-4):
        '''
        lr:learning rate
        n_classes:number of classes
        n_features:the demension of features
        lemma:regularization coefficient
        '''
        self.w = [ [0.00001 for i in range(n_features)] for i in range(n_classes)]
        self.n_classes = n_classes
        self.n_features = n_features
        self.lemma = lemma
        self.lr = lr


    def predict(self,samples):
        '''
        Get predict result of given samples 
        '''
        labels = []
        scores_list = []
        for sample in samples:
                scores = []
                max_tmp = 0 
                for clazz in range(self.n_classes):
                    tmp_score = sum([id*self.w[clazz][id] for id in sample])
                    max_tmp = max(max_tmp,tmp_score)
                    scores.append(tmp_score)
                scores = [ score - max_tmp for score in scores]
                scores  = [math.exp(score) for score in scores]
                s = sum(scores)
                scores = [score/s for score in scores]
                label = 0
                max_score = scores[0]
                for i in range(1,self.n_classes):
                    if scores[i] > max_score:
                        max_score = scores[i]
                        label = i
                labels.append(label)
                scores_list.append(scores)
        return labels, scores_list

    def get_update(self,samples,labels):
        '''
        Get coefficients updated for a mini-batch of samples
        '''
        _,scores_list = self.predict(samples)
        gradient = [ [0.0 for _ in range(self.n_features)] for _ in range(self.n_classes)]
        for id,sample in enumerate(samples):
            for word_id in sample:
                for clazz in range(self.n_classes):
                    if clazz == labels[id]:
                        gradient[clazz][word_id] += 1.0 - 1.0*scores_list[id][clazz]
                    else:
                        gradient[clazz][word_id] += 0.0 - 1.0*scores_list[id][clazz]
        for i in range(self.n_classes):
            for j in range(self.n_features):
                self.w[i][j] += self.lr*gradient[i][j] - self.lemma*self.w[i][j]
        return 

    def train(self,samples,labels,batch_size=100):
        '''
        training for given data and the batch size
        '''
        idx = 0 
        for _ in tqdm(range(len(samples) // batch_size )):
            samples_batch = samples[idx:idx+batch_size]
            labels_batch = labels[idx:idx+batch_size]
            idx += batch_size 
            self.get_update(samples_batch,labels_batch)
            if idx >= len(samples):
                break
        return
    
    def test(self,samples,labels):
        '''
        test on given samples and labels
        '''
        pred,_ = self.predict(samples)
        correct_num = 0 
        for i in range(len(samples)):
            if pred[i] == labels[i]:
                correct_num = correct_num + 1
        #print(confusion_matrix(labels,pred))
        return correct_num/len(samples)

    def save(self,path):
        pickle.dump(self.w,open(path, 'wb'))

    def load_from(self,path):
        w = pickle.load(open(path, 'rb'))
        self.w = w

class SingletonClassifier(object):
	'''
	The binary classifier for singleton classification 
	'''
	def __init__(self,init_nodes,lr=0.001,Lemma=0.0):
		tokens2id = {"#MISSED":0}
		next_id = 1
		for node in init_nodes:
			token = node["token"]
			pos_tag = node["pos"]
			lemma = node["lemma"]
			pos_tag = "pos#"+pos_tag
			lemma = "lemma#"+lemma
			if token not in tokens2id.keys():
				tokens2id[token] = next_id
				next_id += 1
			if pos_tag not in tokens2id.keys():
				tokens2id[pos_tag] = next_id
				next_id += 1
			if lemma not in tokens2id.keys():
				tokens2id[lemma] = next_id
				next_id += 1
		self.tokens2id = tokens2id
		self.classifier = LogLinearModel(lr=lr,n_classes=2,n_features=len(tokens2id),lemma=Lemma)
	

	def nodes2samples(self,nodes):
		samples = []
		for node in nodes:
			sample = []
			sample.append(self.tokens2id.get(node['token'],0))
			sample.append(self.tokens2id.get("pos#"+node['pos'],0))
			sample.append(self.tokens2id.get("lemma#"+node['lemma'],0))
			samples.append(sample)
		return samples
	
	def predict(self,nodes):
		samples = self.nodes2samples(nodes)
		return self.classifier.predict(samples)[0]

	def train(self,nodes,batch_size=100):
		samples = self.nodes2samples(nodes)
		labels = [node['single'] for node in nodes]
		self.classifier.train(samples,labels,batch_size=batch_size)
	
	def test(self,nodes):
		samples = self.nodes2samples(nodes)
		labels = [node['single'] for node in nodes]
		return self.classifier.test(samples, labels)

	def save(self,path):
		records = {"tokens2id":self.tokens2id,"w":self.classifier.w}
		pickle.dump(records,open(path, 'wb'))

	def load_from(self,path,lr=0.001,Lemma=0.0):
		records = pickle.load(open(path, 'rb'))
		self.tokens2id = records["tokens2id"]
		self.classifier = LogLinearModel(lr=lr,n_classes=2,n_features=len(self.tokens2id),lemma=Lemma)
		self.classifier.w = records["w"]



class EdgeClassifier(object):
	'''
	The classifier for edge classification 
	'''
	def __init__(self,init_edges,lr=0.001,Lemma=0.0):
		tokens2id = {"#MISSED":0}
		next_id = 1
		rels = set()
		for edge in init_edges:
			rels.add(edge['rel'])
			tokens = edge["tokens"]
			pos_tags = edge["pos_tags"]
			lemmas = edge["lemmas"]
			neighbor_pos_tags = edge["neighbor_pos_tags"]
			left_token = "left_token#"+tokens[0]
			right_token = "right_token#"+tokens[1]
			left_pos = "left_pos#"+pos_tags[0]
			right_pos = "right_pos#"+pos_tags[1]
			left_lemma = "left_lemma#"+lemmas[0]
			right_lemma = "right_lemma#"+lemmas[1]
			distance = "distance#"+str(edge["idxs"][0]-edge['idxs'][1])
			pos_tags_f1 = "posf1#"+neighbor_pos_tags[0]+"#"+pos_tags[0]+"#"+neighbor_pos_tags[1] + "#" + neighbor_pos_tags[2]+"#"+pos_tags[1]+"#"+neighbor_pos_tags[3]
			pos_tags_f2 = "posf2#"+neighbor_pos_tags[0]+"#"+pos_tags[0]+"#"+neighbor_pos_tags[2]+"#"+pos_tags[1]
			pos_tags_f3 = "posf3#"+neighbor_pos_tags[1]+"#"+pos_tags[0]+"#"+neighbor_pos_tags[3]+"#"+pos_tags[1]
			features = [left_token,right_token,left_lemma,right_lemma,left_pos,right_pos,distance,pos_tags_f1,pos_tags_f2,pos_tags_f3]
			if edge['idxs'][0] > edge['idxs'][1]:
				features.append("###reverse")
			for f in features:
				if f not in tokens2id.keys():
					tokens2id[f] = next_id
					next_id += 1

		self.tokens2id = tokens2id
		self.class2id = {}
		self.id2class = {}
		self.rels = rels
		for id,rel in enumerate(list(rels)):
			self.class2id[rel] = id
			self.id2class[id] = rel 
		self.classifier = LogLinearModel(lr=lr,n_classes=len(rels),n_features=len(tokens2id),lemma=Lemma)
	

	def edges2samples(self,edges):
		samples = []
		#print(len(edges))
		for edge in edges:
			sample = []
			#print(edge)
			#print(type(edge))
			tokens = edge["tokens"]
			pos_tags = edge["pos_tags"]
			lemmas = edge["lemmas"]
			neighbor_pos_tags = edge["neighbor_pos_tags"]
			left_token = "left_token#"+tokens[0]
			right_token = "right_token#"+tokens[1]
			left_pos = "left_pos#"+pos_tags[0]
			right_pos = "right_pos#"+pos_tags[1]
			left_lemma = "left_lemma#"+lemmas[0]
			right_lemma = "right_lemma#"+lemmas[1]
			distance = "distance#"+str(edge["idxs"][0]-edge['idxs'][1])
			pos_tags_f1 = "posf1#"+neighbor_pos_tags[0]+"#"+pos_tags[0]+"#"+neighbor_pos_tags[1] + "#" + neighbor_pos_tags[2]+"#"+pos_tags[1]+"#"+neighbor_pos_tags[3]
			pos_tags_f2 = "posf2#"+neighbor_pos_tags[0]+"#"+pos_tags[0]+"#"+neighbor_pos_tags[2]+"#"+pos_tags[1]
			pos_tags_f3 = "posf3#"+neighbor_pos_tags[1]+"#"+pos_tags[0]+"#"+neighbor_pos_tags[3]+"#"+pos_tags[1]
			features = [left_token,right_token,left_lemma,right_lemma,left_pos,right_pos,distance,pos_tags_f1,pos_tags_f2,pos_tags_f3]
			if edge['idxs'][0] > edge['idxs'][1]:
				features.append("###reverse")
			for f in features:
				sample.append(self.tokens2id.get(f,0))
			samples.append(sample)
		return samples
	
	def predict(self,edges):
		samples = self.edges2samples(edges)
		return self.classifier.predict(samples)[0]


	def train(self,edges,batch_size=100):
		samples = self.edges2samples(edges)
		labels = [self.class2id[edge['rel']] for edge in edges]
		self.classifier.train(samples,labels,batch_size=batch_size)
	
	def test(self,edges):
		samples = self.edges2samples(edges)
		labels = [self.class2id[edge['rel']] for edge in edges]
		return self.classifier.test(samples, labels)

	def save(self,path):
		record = {"w":self.classifier.w,"tokens2id":self.tokens2id,'class2id':self.class2id,'id2class':self.id2class,"rels":self.rels}
		pickle.dump(record,open(path, 'wb'))

	def load_from(self,path,lr=0.001,Lemma=0.0):
		records = pickle.load(open(path, 'rb'))
		self.tokens2id = records['tokens2id']
		self.class2id = records['class2id']
		self.id2class = records['id2class']
		self.rels = records['rels']
		self.classifier = LogLinearModel(lr=lr,n_classes=len(self.rels),n_features=len(self.tokens2id),lemma=Lemma)
		self.classifier.w = records['w']


class EdgeEnsembleClassifier(object):
	'''
	Ensemble Model by simple voting 
	'''
	def __init__(self,init_edges,model_list):
		self.models = []
		for path in model_list:
			classifier = EdgeClassifier(init_edges)
			classifier.load_from(path)
			self.models.append(classifier)
		self.class2id = self.models[0].class2id
		self.id2class = self.models[0].id2class

	def predict(self,edges):
		predictions = [[ self.class2id[model.id2class[class_id]] for class_id in model.predict(edges)] for model in self.models]
		predictions = np.array(predictions)
		predictions = np.transpose(predictions)
		labels = []
		for line in predictions:
			labels.append(int(np.argmax(np.bincount(line))))
		return labels



class TopClassifier(object):
	'''
	The binary classifier for top classification 
	'''
	def __init__(self,init_nodes,lr=0.001,Lemma=0.0):
		tokens2id = {"#MISSED":0}
		next_id = 1
		for node in init_nodes:
			token = node["token"]
			pos_tag = node["pos"]
			lemma = node["lemma"]
			index = node["index"]
			root = node["root"]
			pos_tag = "pos#"+pos_tag
			lemma = "lemma#"+lemma
			index = "index#"+str(index)
			root= "root#"+str(root)
			if token not in tokens2id.keys():
				tokens2id[token] = next_id
				next_id += 1
			if pos_tag not in tokens2id.keys():
				tokens2id[pos_tag] = next_id
				next_id += 1
			if lemma not in tokens2id.keys():
				tokens2id[lemma] = next_id
				next_id += 1
			if index not in tokens2id.keys():
				tokens2id[index] = next_id
				next_id += 1
			if root not in tokens2id.keys():
				tokens2id[root] = next_id
				next_id += 1
		self.tokens2id = tokens2id
		self.classifier = LogLinearModel(lr=lr,n_classes=2,n_features=len(tokens2id),lemma=Lemma)
	

	def nodes2samples(self,nodes):
		samples = []
		for node in nodes:
			sample = []
			sample.append(self.tokens2id.get(node['token'],0))
			sample.append(self.tokens2id.get("pos#"+node['pos'],0))
			sample.append(self.tokens2id.get("lemma#"+node['lemma'],0))
			sample.append(self.tokens2id.get("root#"+str(node['root']),0))
			sample.append(self.tokens2id.get("index#"+str(node['index']),0))
			samples.append(sample)
		return samples
	
	def predict(self,nodes):
		samples = self.nodes2samples(nodes)
		return self.classifier.predict(samples)[0]

	def predict_one_node(self,node):
		sample = []
		sample.append(self.tokens2id.get(node['token'],0))
		sample.append(self.tokens2id.get("pos#"+node['pos'],0))
		sample.append(self.tokens2id.get("lemma#"+node['lemma'],0))
		sample.append(self.tokens2id.get("root#"+str(['root']),0))
		sample.append(self.tokens2id.get("index#"+str(node['index']),0))
		labels,scores_list = self.classifier.predict([sample])
		return labels[0],scores_list[0]

	def train(self,nodes,batch_size=100):
		samples = self.nodes2samples(nodes)
		labels = [node['top'] for node in nodes]
		self.classifier.train(samples,labels,batch_size=batch_size)
	
	def test(self,nodes):
		samples = self.nodes2samples(nodes)
		labels = [node['top'] for node in nodes]
		return self.classifier.test(samples, labels)

	def save(self,path):
		records = {"tokens2id":self.tokens2id,"w":self.classifier.w}
		pickle.dump(records,open(path, 'wb'))

	def load_from(self,path,lr=0.001,Lemma=0.0):
		records = pickle.load(open(path, 'rb'))
		self.tokens2id = records["tokens2id"]
		self.classifier = LogLinearModel(lr=lr,n_classes=2,n_features=len(self.tokens2id),lemma=Lemma)
		self.classifier.w = records["w"]



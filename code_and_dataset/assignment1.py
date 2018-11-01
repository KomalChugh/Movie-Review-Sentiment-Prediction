import random
import math
import sys
import copy

class Node(object):
	value=0     		# attribute 
	left_child=None
	right_child=None
	label=0
	instances=0		# number of instances at that node
	right_inst=0		# number of instances at its right child node
	left_inst=0		# number of instances at its left child node
	pos=0			# number of positive instances at that node
	neg=0			# number of negative instances at that node
	
	def __init__(self,val,left,right,label,instances,right_inst,left_inst,pos,neg):
		self.value=val
		self.left_child=left
		self.right_child=right
		self.label=label
		self.instances=instances
		self.right_inst=right_inst
		self.left_inst=left_inst
		self.pos=pos
		self.neg=neg

	

def last(n):
	return n[0]


# computes entropy for a node
def compute_entropy(pos_eg,neg_eg):
	if(pos_eg+neg_eg==0):
		return 0
	prob_pos= float(pos_eg)/(pos_eg+neg_eg)
	prob_neg= float(neg_eg)/(pos_eg+neg_eg)
	if(prob_pos==0 or prob_neg==0):
		entropy = 0
	else:
		entropy = -((prob_pos)*(math.log(prob_pos,2))+(prob_neg)*(math.log(prob_neg,2)))
	return entropy
	

# returns label for a given review and decision tree	
def get_label(tree_node,feature_list):
	if(tree_node.label!=0):
		return tree_node.label
	if(tree_node.value in feature_list):
		return get_label(tree_node.left_child,feature_list)
	else:
		return get_label(tree_node.right_child,feature_list)
		

# returns root node of a decision tree built by ID3 algorithm 
def build_tree(pos_eg,neg_eg,node_list,train_output_list,word_indices,early_stopping,min_instances,depth,max_depth):
	global tree_depth
	if(tree_depth<depth):
		tree_depth= depth
	if(len(pos_eg)==0):
		node = Node(0,None,None,-1,len(pos_eg)+len(neg_eg),0,0,0,len(neg_eg))
		return node
		
	if(len(neg_eg)==0):
		node = Node(0,None,None,1,len(pos_eg)+len(neg_eg),0,0,len(pos_eg),0)
		return node
	
	if(early_stopping=="instances"):
		if(len(pos_eg)+len(neg_eg)<min_instances):
			if(len(pos_eg)>=len(neg_eg)):
				label = 1
			else:
				label= -1
			node = Node(0,None,None,label,len(pos_eg)+len(neg_eg),0,0,len(pos_eg),len(neg_eg))
			return node
			
	if(early_stopping=="depth"):
		if(depth==max_depth):
			if(len(pos_eg)>=len(neg_eg)):
				label = 1
			else:
				label= -1
			node = Node(0,None,None,label,len(pos_eg)+len(neg_eg),0,0,len(pos_eg),len(neg_eg))
			return node
		
	info_gain=0
	present_pos_list=[]
	present_neg_list=[]
	not_present_pos_list=[]
	not_present_neg_list=[]
	max_entro_word = -1
	
	# compute information gain for all attributes not present in node_list and choose best attribute	
	for j in word_indices:
		if(j in node_list):
			continue
		entro_parent = compute_entropy(len(pos_eg),len(neg_eg))
		yes_pos_list=[]
		yes_neg_list=[]
		no_pos_list=[]
		no_neg_list=[]
		yes_pos_eg=0
		yes_neg_eg=0
		no_pos_eg=0
		no_neg_eg=0
		for i in pos_eg:
			if(j in train_feature_list[i]):
				if(train_output_list[i]==1):
					yes_pos_eg = yes_pos_eg+1
					yes_pos_list.append(i)
				else:
					yes_neg_eg = yes_neg_eg+1
					yes_neg_list.append(i)
			else:
				if(train_output_list[i]==1):
					no_pos_eg = no_pos_eg+1
					no_pos_list.append(i)
				else:
					no_neg_eg = no_neg_eg+1
					no_neg_list.append(i)
					
		for i in neg_eg:
			if(j in train_feature_list[i]):
				if(train_output_list[i]==1):
					yes_pos_eg = yes_pos_eg+1
					yes_pos_list.append(i)
				else:
					yes_neg_eg = yes_neg_eg+1
					yes_neg_list.append(i)
			else:
				if(train_output_list[i]==1):
					no_pos_eg = no_pos_eg+1
					no_pos_list.append(i)
				else:
					no_neg_eg = no_neg_eg+1
					no_neg_list.append(i)
					
		entro_yes = compute_entropy(yes_pos_eg,yes_neg_eg)
		entro_no = compute_entropy(no_pos_eg,no_neg_eg)
		prob_yes = float(yes_pos_eg+yes_neg_eg)/(yes_pos_eg+yes_neg_eg+no_pos_eg+no_neg_eg)
		prob_no = float(no_pos_eg+no_neg_eg)/(yes_pos_eg+yes_neg_eg+no_pos_eg+no_neg_eg)
		
		if(float(entro_parent) - float(prob_yes*entro_yes) - float(prob_no*entro_no) > info_gain):
			info_gain = entro_parent - (prob_yes*entro_yes) - (prob_no*entro_no)
			present_pos_list= yes_pos_list
			present_neg_list= yes_neg_list
			not_present_pos_list= no_pos_list
			not_present_neg_list= no_neg_list
			max_entro_word = j
	
	# if there is no increase in information gain
	if(max_entro_word==-1):
		if(len(pos_eg)>=len(neg_eg)):
			label=1
		else:
			label=-1
		node = Node(0,None,None,label,len(pos_eg)+len(neg_eg),0,0,len(pos_eg),len(neg_eg))
		return node
		
	parent_node = Node(max_entro_word,None,None,0,len(pos_eg)+len(neg_eg),len(not_present_pos_list)+len(not_present_neg_list),len(present_pos_list)+len(present_neg_list),len(pos_eg),len(neg_eg))
	child_list = []
	for m in node_list:
		child_list.append(m)
	child_list.append(max_entro_word)
	parent_node.left_child = build_tree(present_pos_list,present_neg_list,child_list,train_output_list,word_indices,early_stopping,min_instances,depth+1,max_depth)
	parent_node.right_child = build_tree(not_present_pos_list,not_present_neg_list,child_list,train_output_list,word_indices,early_stopping,min_instances,depth+1,max_depth)
	
	
	return parent_node
			


# counts number of internal nodes and leaves while traversing the tree					
def print_tree(node):
	global internal_nodes
	global leaves
	global count_occ_list
	if(node==None):
		return
	print_tree(node.left_child)
	print_tree(node.right_child)
	
	if(node.label==0):
		internal_nodes = internal_nodes + 1
		if(node.value in count_occ_list):
			count_occ_list[node.value] = count_occ_list[node.value] + 1
			
		else:
			count_occ_list[node.value]=1;
		
	else:
		leaves = leaves + 1

		
# computes test and train accuracy  		
def get_accuracy(tree,test_output_list,train_output_list):
	
	wrong_label_count=0
	
	for i in range(1000):
		output_label = get_label(tree,test_feature_list[i])
		if(output_label!=test_output_list[i]):
			wrong_label_count = wrong_label_count + 1
		
	sample_size=1000
		
	test_accuracy = float((sample_size-wrong_label_count)*100)/sample_size
	

	wrong_label_count=0
		 
	for i in range(1000):
		output_label = get_label(tree,train_feature_list[i])
		if(output_label!=train_output_list[i]):
			wrong_label_count = wrong_label_count + 1

		
	train_accuracy = float((sample_size-wrong_label_count)*100)/sample_size
	return test_accuracy,train_accuracy
	

# adds random noise to train data and builds a decision tree
def add_noise(percent,sample_size):
	num = int((percent*sample_size)/100)
	flip_indices = random.sample(range(sample_size),num)
	noised_output_list=[]
	pos_review_list=[]
	neg_review_list=[]
	node_list=[]
	for i in range(sample_size):
		if(i in flip_indices):
			true_label = train_output_list[i]
			if(true_label==1):
				noised_output_list.append(int(-1))
				neg_review_list.append(i)
			else:
				noised_output_list.append(int(1))
				pos_review_list.append(i)
		else:
			noised_output_list.append(train_output_list[i])
			true_label = train_output_list[i]
			if(true_label==1):
				pos_review_list.append(i)
			else:
				neg_review_list.append(i)
	
	global tree_depth
	tree_depth = 0			
	noised_tree = build_tree(pos_review_list,neg_review_list,node_list,noised_output_list,word_indices,"no",0,0,200)

	global internal_nodes
	global leaves
	global count_occ_list
	count_occ_list = {}
	internal_nodes = 0
	leaves = 0
	print_tree(noised_tree)
	print '%-15s  --->  %-6s' %("internal nodes",str(internal_nodes))
	print '%-15s  --->  %-6s' %("leaves",str(leaves))
	print '%-15s  --->  %-6s' %("tree depth",str(tree_depth))
	
	
	test_accuracy,train_accuracy = get_accuracy(noised_tree,test_output_list,noised_output_list)
	print '%-15s  --->  %-6s' %("test accuracy",str(test_accuracy))
	print '%-15s  --->  %-6s' %("train accuracy",str(train_accuracy))
	

# builds a decision tree with or without early stopping based on the parameter passed
def decision_tree(early_stopping,min_instances,max_depth,frequency):
	pos_eg_list=[]
	neg_eg_list=[]

	for i in range(500):
		pos_eg_list.append(i)

	for i in range(500,1000):
		neg_eg_list.append(i)
	
	node_list = []

	global tree_depth
	tree_depth = 0

	tree = build_tree(pos_eg_list,neg_eg_list,node_list,train_output_list,word_indices,early_stopping,min_instances,0,max_depth)
	
	
	global internal_nodes
	global leaves
	internal_nodes = 0
	leaves = 0
	global count_occ_list
	count_occ_list = {}
	print_tree(tree)
	print '%-15s  --->  %-6s' %("internal nodes",str(internal_nodes))
	print '%-15s  --->  %-6s' %("leaves",str(leaves))
	print '%-15s  --->  %-6s' %("tree depth",str(tree_depth))
	
	test_accuracy,train_accuracy = get_accuracy(tree,test_output_list,train_output_list)
	print '%-15s  --->  %-6s' %("test accuracy",str(test_accuracy))
	print '%-15s  --->  %-6s' %("train accuracy",str(train_accuracy))
	
	count = 0
	if(frequency=="yes"):
		print("\n\n Number of times an attribute is used as the splitting function\n")
		print '%-15s  --->  %-6s' %("attribute","frequency")
		sorted_indices = sorted(count_occ_list.items(), key=lambda x:x[1], reverse=True)
		for key,value in sorted_indices:
			if(count==10):
				break
			word = vocab_list[key]
			print '%-15s  --->  %-6s' %(word,str(value))
			count = count+1
	return tree	
	
# builds a decision forest using feature bagging		
def decision_forest(feature_size,num_trees,num_nodes):
	
	k = 0
	pos_eg_list=[]
	neg_eg_list=[]
	pos_test_output_labels= [0 for i in range(1000)]
	neg_test_output_labels= [0 for i in range(1000)]
	pos_train_output_labels= [0 for i in range(1000)]
	neg_train_output_labels= [0 for i in range(1000)]

	for i in range(500):
		pos_eg_list.append(i)

	for i in range(500,1000):
		neg_eg_list.append(i)
	
	
	for i in range(num_trees):
		feature_bagging_list = []
		random_list = random.sample(range(feature_size),num_nodes)
		for j in random_list:
			feature_bagging_list.append(word_indices[j])
		node_list = []

		global tree_depth
		tree_depth = 0

		tree = build_tree(pos_eg_list,neg_eg_list,node_list,train_output_list,feature_bagging_list,"no",0,0,200)
		for i in range(1000):
			label = get_label(tree,test_feature_list[i])
			if(label==1):
				pos_test_output_labels[i] = pos_test_output_labels[i] + 1
			else:
				neg_test_output_labels[i] = neg_test_output_labels[i] + 1
			
		for i in range(1000):
			label = get_label(tree,train_feature_list[i])
			if(label==1):
				pos_train_output_labels[i] = pos_train_output_labels[i] + 1
			else:
				neg_train_output_labels[i] = neg_train_output_labels[i] + 1
				
		
	wrong_label_count = 0			
	for i in range(1000):
		if(pos_test_output_labels[i]>=neg_test_output_labels[i]):
			label = 1
		else:
			label = -1
		if(label!=test_output_list[i]):
			wrong_label_count = wrong_label_count + 1
			
	sample_size=1000
	test_accuracy = float((sample_size-wrong_label_count)*100)/sample_size
	
	wrong_label_count = 0			
	for i in range(1000):
		if(pos_train_output_labels[i]>=neg_train_output_labels[i]):
			label = 1
		else:
			label = -1
		if(label!=train_output_list[i]):
			wrong_label_count = wrong_label_count + 1
			
	sample_size=1000
	train_accuracy = float((sample_size-wrong_label_count)*100)/sample_size
	print '%-20s  --->  %-20s  --->  %-20s' %(num_trees,train_accuracy,test_accuracy)


# given a tree, this function removes a subtree for which the resultant tree has maximum increase in test accuracy  
def remove_subtree(tree,root):
	global pruned_tree
	global original_accuracy
	if(root.label!=0):
		return
	remove_subtree(tree,root.left_child)
	remove_subtree(tree,root.right_child)
	if(root.pos>=root.neg):
		label=1
	else:
		label=-1
	temp_left = copy.deepcopy(root.left_child)
	temp_right = copy.deepcopy(root.right_child)
	temp_value = root.value
	root.value = 0
	root.left_child = None
	root.right_child = None
	root.label= label
	test_accuracy,train_accuracy = get_accuracy(tree,test_output_list,train_output_list)
	if(test_accuracy>original_accuracy):
		original_accuracy = test_accuracy
		pruned_tree = copy.deepcopy(tree)
	root.value = temp_value
	root.left_child = temp_left
	root.right_child = temp_right
	root.label = 0
	
	
# prunes a tree recursively until there is an increase in test accuracy
def prune_tree(tree):
	global pruned_tree
	global original_accuracy
	test_accuracy,train_accuracy = get_accuracy(tree,test_output_list,train_output_list)
	original_accuracy = test_accuracy
	while(1):
		pruned_tree = None
		remove_subtree(tree,tree)
		if(pruned_tree==None):
			break
		new_test_accuracy,new_train_accuracy = get_accuracy(pruned_tree,test_output_list,train_output_list)
		print("\n\n")
		global internal_nodes
		global leaves
		global count_occ_list
		internal_nodes = 0
		leaves = 0
		count_occ_list = {}
		print_tree(pruned_tree)
		print '%-20s  --->  %-6s' %("internal nodes",str(internal_nodes))
		print '%-20s  --->  %-6s' %("leaves",str(leaves))
		print '%-20s  --->  %-6s' %("new test accuracy",str(new_test_accuracy))
		print '%-20s  --->  %-6s' %("new train accuracy",str(new_train_accuracy))
	
	
		tree = copy.deepcopy(pruned_tree)
		
		
# extract 2500 most positive sentimental and 2500 most negative sentimental words from imdbEr.txt file

filename= "aclImdb/imdbEr.txt"
file = open(filename,"r")
word_list = []
i=0
for line in file:
	word_list.append((float(line),i))
	i=i+1

word_list = sorted(word_list, key=last)
word_indices=[]
k=0
for j in word_list:
	word_indices.append(j[1])
	k = k+1
	if(k==2500):
		break

word_list.reverse()
k=0
for j in word_list:
	word_indices.append(j[1])
	k = k+1
	if(k==2500):
		break
		
# put vocabulary words present in imdb.vocab in a list

vocab_file = open("aclImdb/imdb.vocab",'r')
vocab_list = []
for line in vocab_file:
	vocab_list.append(str(line[:-1]))


# read data from file or sample again on the basis of argument 

if(sys.argv[1]=='-f'):
	train_reviews = list(open("train_data_file.txt"))
	test_reviews = list(open("test_data_file.txt"))

elif(sys.argv[1]=='-r'):
	train_reviews = random.sample(list(open("aclImdb/train/labeledBow.feat"))[:12500],500)
	train_reviews = train_reviews + random.sample(list(open("aclImdb/train/labeledBow.feat"))[12500:],500)

	train_data_file = open("train_data_file.txt","w")
	for line in train_reviews:
		train_data_file.write(str(line))
		
	test_reviews = random.sample(list(open("aclImdb/test/labeledBow.feat"))[:12500],500)
	test_reviews = test_reviews + random.sample(list(open("aclImdb/test/labeledBow.feat"))[12500:],500)

	test_data_file = open("test_data_file.txt","w")
	for line in test_reviews:
		test_data_file.write(str(line))
		

# pre-process train data
train_feature_list=[]
train_output_list=[]


for line in train_reviews:
	data = line.split(" ")
	indices = []
	frequency = []
	for j in range(1,len(data)):
		x = data[j].split(":")
		indices.append(int(x[0]))
		frequency.append(int(x[1]))
	lst=[]
	for j in word_indices:
		if(j in indices):
			lst.append(j)
	train_feature_list.append(lst)
	

for i in range(500):
	train_output_list.append(int(1))

for i in range(500):
	train_output_list.append(int(-1))
	

# pre-process test data	
test_feature_list=[]
test_output_list=[]

for line in test_reviews:
	data = line.split(" ")
	indices = []
	frequency = []
	for j in range(1,len(data)):
		x = data[j].split(":")
		indices.append(int(x[0]))
		frequency.append(int(x[1]))
	lst=[]
	for j in word_indices:
		if(j in indices):
			lst.append(j)
	test_feature_list.append(lst)
	
for i in range(500):
	test_output_list.append(int(1))

for i in range(500):
	test_output_list.append(int(-1))
	
	
# global variables used	
tree_depth = 0
internal_nodes = 0
leaves = 0
count_occ_list = {}
original_accuracy = 0
pruned_tree = None


# different experiments

# decision tree with early stopping and number of times an attribute is used for splitting
if(sys.argv[2]=='2'):
	print("Decision tree without early stopping \n")
	tree = decision_tree("no",0,200,"yes")
	print("\n\n Early stopping with instances less than 50 \n")
	early_stopped_tree_50 = decision_tree("instances",50,150,"yes")
	print("\n\n Early stopping with instances less than 70 \n")
	early_stopped_tree_70 = decision_tree("instances",70,150,"yes")
	print("\n\n Early stopping with instances less than 100 \n")
	early_stopped_tree_100 = decision_tree("instances",100,150,"yes")
	print("\n\n Early stopping with instances less than 150 \n")
	early_stopped_tree_150 = decision_tree("instances",150,150,"yes")
	print("\n\n Early stopping with maximum depth upto 50 \n")
	early_stopped_depth_50 = decision_tree("depth",150,50,"yes")
	
		

# decision tree with some noise added 
elif(sys.argv[2]=='3'):
	print("Decision tree without any noise\n")
	tree = decision_tree("no",0,200,"no")
	noise_percent = 0.5
	print("\n 0.5% noise added to tree\n")
	add_noise(noise_percent,1000)
	noise_percent = 1
	print("\n 1% noise added to tree\n")
	add_noise(noise_percent,1000)
	noise_percent = 5
	print("\n 5% noise added to tree\n")
	add_noise(noise_percent,1000)
	noise_percent = 10
	print("\n 10% noise added to tree\n")
	add_noise(noise_percent,1000)
	noise_percent = 20
	print("\n 20% noise added to tree\n")
	add_noise(noise_percent,1000)
	
# pruned decision tree 	
elif(sys.argv[2]=='4'):
	print("Decision tree without pruning\n")
	tree = decision_tree("no",0,200,"no")
	print("\n\nDecision tree after pruning\n")
	prune_tree(tree)
	
# decision forest
elif(sys.argv[2]=='5'):
	feature_size = 5000
	print("Effect of number of trees in the forest on train and test accuracies\n")
	print '%20s  --->  %20s  --->  %20s' %("Number of trees","Train accuracy","Test accuracy")
	decision_forest(feature_size,1,2000)
	decision_forest(feature_size,3,2000)
	decision_forest(feature_size,5,2000)
	decision_forest(feature_size,10,2000)
	decision_forest(feature_size,15,2000)
	decision_forest(feature_size,20,2000)
	decision_forest(feature_size,25,2000)





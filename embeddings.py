import pandas as pd
import numpy as np
import ast_helper
import tree_lstm


df = pd.read_csv('labels.csv')

lst=[]
for sub1, sub2 in zip(df["sub1"],df["sub2"]):
    lst.append(sub1)
    lst.append(sub2)
    #print(sub1," ",sub2)

lst=list(set(lst))

file_contents = {}
lst1=[]
for f in lst:
    with open("./template-free/"+f+".java", "r") as file:
        try:
            t=file.read()
        except:
            continue
        file_contents[f]=ast_helper.remove_comments_and_whitespace(t)
        lst1.append(f)

model = tree_lstm.TreeLSTM(input_dim=128, hidden_dim=256)
embeddings = {}
j=0
for i in lst1:
    ast_tree = ast_helper.parse_java_code(file_contents[i])
    if(ast_tree is None):
      print("Error in: ",i)
      continue

    root_node = tree_lstm.parse_javalang_ast(ast_tree,None)
    embedding = model.encode(root_node).detach().numpy()
    codebert_embedding = tree_lstm.get_codebert_embedding(file_contents[i])
    final_embedding = np.concatenate((embedding, codebert_embedding))

    embeddings[i] = final_embedding
    print(j)
    j+=1

    if(j==20):
        break

np.savez("embeddings/test.npz", **embeddings)
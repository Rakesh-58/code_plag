from javalang.tree import LocalVariableDeclaration, ClassCreator, StatementExpression, ArrayCreator, ArrayInitializer
from javalang.tree import MethodInvocation, MemberReference, Import
from javalang.tree import Node

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

NODE_TYPES = ['MethodInvocation', 'MemberReference', 'Literal', 
    'BinaryOperation', 'VariableDeclarator', 'IfStatement',
    'ForStatement', 'WhileStatement', 'ReturnStatement',
    'Assignment', 'Parameter', 'Type', 'MethodDeclaration',
    'CompilationUnit', 'BlockStatement', 'ExpressionStatement',
    'FieldDeclaration', 'ClassDeclaration', 'ConstructorDeclaration'
]

NODE_TYPE_TO_INDEX = {node_type: idx for idx, node_type in enumerate(NODE_TYPES, start=1)}
UNKNOWN_NODE_INDEX = 0

def parse_javalang_ast(node,parent):
   
    if node is None:
        return None
    
    # If node is a javalang AST Node
    if isinstance(node, Node):

        if(isinstance(node,Import)):
            return None

        if isinstance(node,LocalVariableDeclaration) or isinstance(node, ClassCreator) or isinstance(node, StatementExpression) or isinstance(node, ArrayInitializer) or isinstance(node, ArrayCreator):
            for child in getattr(node, 'children', []):
                if isinstance(child, list):
                    for sub_child in child:
                        sub_tree = parse_javalang_ast(sub_child,parent)
                        if sub_tree:
                            parent.children.append(sub_tree)
                elif child is not None:
                    sub_tree = parse_javalang_ast(child,parent)
                    if sub_tree:
                        parent.children.append(sub_tree)

            return None

        value = type(node).__name__
        
        tree_node = TreeNode(value)
        # Add variable names, literals, and operators to the label
        if hasattr(node, 'name') and node.name:
            val_node = TreeNode(node.name)
            tree_node.children.append(val_node)
        elif hasattr(node, 'value') and node.value and not isinstance(node.value,Node):
            val_node = TreeNode(node.value)
            tree_node.children.append(val_node)
        elif hasattr(node, 'operator') and node.operator:
            val_node = TreeNode(node.operator)
            tree_node.children.append(val_node)           
        
        if isinstance(node, MethodInvocation) or isinstance(node, MemberReference):
            if hasattr(node, 'qualifier') and node.qualifier:
                val_node = TreeNode(node.qualifier)
                tree_node.children.append(val_node)
            elif hasattr(node, 'member') and node.member:
                val_node = TreeNode(node.member)
                tree_node.children.append(val_node)
                

        for child in getattr(node, 'children', []):
            if isinstance(child, list):
                for sub_child in child:
                    sub_tree = parse_javalang_ast(sub_child,tree_node)
                    if sub_tree:
                        tree_node.children.append(sub_tree)
            elif child is not None:
                sub_tree = parse_javalang_ast(child,tree_node)
                if sub_tree:
                    tree_node.children.append(sub_tree)
        
        return tree_node    
    return None

class TreeLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=input_dim)  
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, node: TreeNode) -> torch.Tensor:
        node_type_id = NODE_TYPE_TO_INDEX.get(node.value, UNKNOWN_NODE_INDEX)
        if(node_type_id == 0):
            if 'var' in node.value and node.value[3:].isnumeric():
                node_type_id = int(node.value[3:])
                node_type_id = node_type_id + 19
                node_type_id = node_type_id % 100
            else:
                node_type_id = hash(node.value) % 100
        
        x = self.embedding(torch.tensor(node_type_id))
             
        x = x / torch.norm(x, p=2)  

        if not node.children:
            h_sum = torch.zeros(self.hidden_dim)  
            c_avg = torch.zeros(self.hidden_dim)
        else:
            child_h, child_c = zip(*[self.forward(child) for child in node.children])
            h_sum = torch.sum(torch.stack(child_h), dim=0)
            c_avg = torch.mean(torch.stack(child_c), dim=0)

        self.dropout = nn.Dropout(0.2)  # 20% dropout

        h, c = self.cell(self.dropout(x), (h_sum, c_avg))
        return h, c

    def encode(self, root: TreeNode) -> torch.Tensor:
        h, _ = self.forward(root)
        return h

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
codebert = RobertaModel.from_pretrained("microsoft/codebert-base")

def get_codebert_embedding(code):
    tokens = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = codebert(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract CLS token representation
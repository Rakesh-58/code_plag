from javalang.parse import parse
from javalang.tree import MethodDeclaration, VariableDeclarator, MethodInvocation, MemberReference, CompilationUnit, ClassDeclaration
import re

def remove_comments_and_whitespace(java_code):
    java_code = re.sub(r'/\*[^*]*\*+(?:[^/*][^*]*\*+)*/', '', java_code, flags=re.DOTALL) # Remove multiline comments (/* ... */)
    java_code = re.sub(r'//.*', '', java_code) # Remove single-line comments (// ...)    
    java_code = re.sub(r'\s+', ' ', java_code) # Remove extra whitespace (replace multiple spaces/newlines with a single space)
    return java_code.strip()

class CodeReorderer:
    def reorder(self,tree):
        self._visit(tree)
        return tree
    
    def _visit(self, node):
        if isinstance(node, CompilationUnit):            
            child_list = getattr(node,'types',[])
            if(child_list):
                child_list.sort(key = lambda x: getattr(x,'name',' '))
                node.types = child_list
        
        elif isinstance(node, ClassDeclaration):
            child_list = getattr(node,'body',[])
            if(child_list):
                child_list.sort(key = lambda x: getattr(x,'name',' '))
                node.body = child_list

class JavaASTNormalizer:
    def __init__(self):
        self.global_name_counter = 0
        self.global_name_mapping = {}
        self.function_name_mapping = {}
        self.local_name_mapping = {}

    def _get_generic_name(self, original_name, mapping):
        if original_name not in mapping:
            self.global_name_counter += 1
            mapping[original_name] = f"var{self.global_name_counter}"
        return mapping[original_name]

    def normalize(self, tree):
        self._visit(tree)
        return tree

    def _visit(self, node):
        
        if isinstance(node, MethodDeclaration):
            # Rename function name
            new_name = self._get_generic_name(node.name, self.function_name_mapping)
            node.name = new_name

        elif isinstance(node, MethodInvocation):
            # Rename function calls
            if node.member in self.function_name_mapping:
                node.member = self.function_name_mapping[node.member]
            else:
                new_name = self._get_generic_name(node.member, self.function_name_mapping)
                node.member = new_name
            if node.qualifier in self.local_name_mapping:
                node.qualifier = self.local_name_mapping[node.qualifier]
            else:
                new_name = self._get_generic_name(node.qualifier, self.function_name_mapping)
                node.qualifier = new_name

        elif isinstance(node, VariableDeclarator):
            # Rename variables
            node.name = self._get_generic_name(node.name, self.local_name_mapping)
        
        elif isinstance(node, MemberReference):
            node.member = self._get_generic_name(node.member, self.local_name_mapping)

        # Recursively visit children
        for child in getattr(node, 'children', []):
            if isinstance(child, list):
                for sub_child in child:
                    self._visit(sub_child)
            elif child is not None:
                self._visit(child)

def parse_java_code(code: str):
    try:
        tree = parse(code)
        reorder = CodeReorderer()
        tree = reorder.reorder(tree)
        normalizer = JavaASTNormalizer()
        normalized_tree = normalizer.normalize(tree)
        return normalized_tree
    except Exception as e:
        print(f"Syntax error in Java code: {e}")
        return None
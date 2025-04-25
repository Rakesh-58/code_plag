import numpy as np
import tree_lstm
import ast_helper
import torch
import torch.nn as nn


class SiamesePlagiarismClassifier(nn.Module):
    def __init__(self, input_size=1024):  # Updated input size
        super(SiamesePlagiarismClassifier, self).__init__()

        self.shared_fc = nn.Sequential(
            nn.Linear(input_size, 512),  # Reduce from 896 to 512
            nn.BatchNorm1d(512),  # Batch Normalization for stability
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),  
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        
        )

        self.out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb1, emb2):
        emb1 = self.shared_fc(emb1)
        emb2 = self.shared_fc(emb2)
        similarity = torch.abs(emb1 - emb2)
        return self.sigmoid(self.out(similarity))
    

def print_tree(node, level=0):
    """Recursively prints the tree structure with indentation."""
    if node is None:
        return
    
    # Print the current node with indentation based on the tree level
    print(" " * (level * 2) + f"- {node.value}")
    
    # Recursively print all children
    for child in node.children:
        print_tree(child, level + 1)


def get_embedding(code):
    model = tree_lstm.TreeLSTM(input_dim=128, hidden_dim=256)
    t = ast_helper.remove_comments_and_whitespace(code)
    ast_tree = ast_helper.parse_java_code(t)
    if(ast_tree is None):
      print("Error")
      return

    root_node = tree_lstm.parse_javalang_ast(ast_tree,None)
    #print_tree(root_node)
    embedding = model.encode(root_node).detach().numpy()
    codebert_embedding = tree_lstm.get_codebert_embedding(t)
    final_embedding = np.concatenate((embedding, codebert_embedding))

    return final_embedding

def predict(s1,s2):
    e1 = get_embedding(s1)
    e1.shape
    e2 = get_embedding(s2)
    e2.shape

    smodel = SiamesePlagiarismClassifier()
    smodel.load_state_dict(torch.load(f'models/10_hyb.pth'))
    smodel.eval()
    embedding1 = torch.tensor(e1, dtype=torch.float32)
    embedding2 = torch.tensor(e2, dtype=torch.float32)

    with torch.no_grad():
        output = smodel(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).squeeze()
        prediction = (output > 0.5).float().item()

    return (prediction, output.item())

if __name__ == '__main__':
    s1 = '''
    // Java program for coin change problem.
    // using recursion

    class GfG {

        static int countRecur(int[] c, int num, int s) {

            // If sum is 0 then there is 1 solution
            // (do not include any coin)
            if (s == 0) return 1;

            // 0 ways in the following two cases
            if (s < 0 || num == 0) return 0;

            // count is sum of solutions (i)
            // including coins[n-1] (ii) excluding coins[n-1]
            return countRecur(c, num, s - c[n - 1]) +
                    countRecur(c, num - 1, s);
        }

        static int count(int[] c, int s) {
            return countRecur(c, c.length, s);
        }

    }

    '''
    s2 = '''
    // Java program for coin change problem.
    // using recursion

    class GfG {

        static int countRecur(int[] coins, int n, int sum) {

            // If sum is 0 then there is 1 solution
            // (do not include any coin)
            if (sum == 0) return 1;

            // 0 ways in the following two cases
            if (sum < 0 || n == 0) return 0;

            // count is sum of solutions (i)
            // including coins[n-1] (ii) excluding coins[n-1]
            return countRecur(coins, n, sum - coins[n - 1]) +
                    countRecur(coins, n - 1, sum);
        }

        static int count(int[] coins, int sum) {
            return countRecur(coins, coins.length, sum);
        }

        public static void main(String[] args) {
            int[] coins = {1, 2, 3};
            int sum = 5;
            System.out.println(count(coins, sum));
        }
    }

    '''
    s3 = '''

    import java.util.*;

    class TUF {
        // Function to count the ways to make change
        static long countWaysToMakeChange(int[] arr, int n, int T) {
            // Create an array to store results of subproblems for the previous element
            long[] prev = new long[T + 1];

            // Initialize base condition for the first element of the array
            for (int i = 0; i <= T; i++) {
                if (i % arr[0] == 0)
                    prev[i] = 1;
                // Else condition is automatically fulfilled, as prev array is initialized to zero
            }

            // Fill the prev array using dynamic programming
            for (int ind = 1; ind < n; ind++) {
                // Create an array to store results of subproblems for the current element
                long[] cur = new long[T + 1];
                for (int target = 0; target <= T; target++) {
                    long notTaken = prev[target];

                    long taken = 0;
                    if (arr[ind] <= target)
                        taken = cur[target - arr[ind]];

                    cur[target] = notTaken + taken;
                }
                prev = cur;
            }

            return prev[T];
        }

        public static void main(String args[]) {
            int arr[] = { 1, 2, 3 };
            int target = 4;
            int n = arr.length;

            // Call the countWaysToMakeChange function and print the result
            System.out.println("The total number of ways is " + countWaysToMakeChange(arr, n, target));
        }
    }



    '''
    predict(s1,s3)
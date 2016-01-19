

# import pandas as pd

import numpy as np
from numpy import random

from sklearn.tree import DecisionTreeClassifier as DTree, export_graphviz as ExportDtree
from sklearn.datasets import make_blobs

from collections import deque
import json


class UDSpace():

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def fit(self, X):
        """
        accept
        """
        self.X = X
        self.shape = X.shape
        self.max_row = np.amax(X, axis=0)
        self.min_row = np.amin(X, axis=0)
        self.gap_row = self.max_row - self.min_row

        return self

    def transform(self, ratio=None):
        shape = list(self.shape)

        if ratio is None:
            ratio = self.ratio

        # if n_rows is None:
        #     ratio = 0.5
        # elif n_rows < 1:
        #     ratio = n_rows
        # else:
        #     ratio = 1.0*n_rows/shape[0]
        
        shape[0] *= ratio
        
        ud_X = random.sample(shape) * self.gap_row + self.min_row
        X = self.X

        Y = np.zeros(len(X), dtype=np.int)
        ud_Y = np.ones(len(ud_X), dtype=np.int)
        
        data_X = np.concatenate((X, ud_X), axis=0)
        target_Y = np.concatenate((Y, ud_Y), axis=0)

        return data_X, target_Y


class DTreePlus_C():
    """
    two question remain:
    - why exp_score is much worser than leaf_sibling_values
    - how to make dtree mse smoother,
    above questions lead to the task of dtree space
    """

    def __init__(self, tree):
        """
        tree = DTree.tree_
        """
        self.tree = tree
        self.parent_arr = self.gen_tree_parent(tree)
        self.sibling_arr = self.gen_tree_sibling(tree)
        self.sibling_values = self.gen_sibling_values(tree)

        self.leaf_values = self.gen_leaf_values()
        self.leaf_sibling_values = self.gen_leaf_sibling_values()

    def gen_tree_parent(self, tree):
        node_count = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        parent_arr = np.zeros(node_count)
        for i in xrange(0, node_count):
            child_left = children_left[i]
            child_right = children_right[i]

            if child_left >0:
                parent_arr[child_left] = i
                parent_arr[child_right] = i

        parent_arr[0] = -1
        return parent_arr

    def gen_tree_sibling(self, tree):
        node_count = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        sibling_arr = np.zeros(node_count)

        for sibling_left, sibling_right in zip(children_left, children_right):
            if sibling_left >0:
                sibling_arr[sibling_left] = sibling_right
                sibling_arr[sibling_right] = sibling_left

        sibling_arr[0] = -1
        return sibling_arr

    def gen_sibling_values(self, tree):
        sibling_arr = self.sibling_arr
        sibling_values = np.zeros(tree.node_count)
        node_values = tree.value
        for node_index, sibling_index in enumerate(sibling_arr):
            if sibling_index>0:
                sibling_values[node_index] = node_values[sibling_index]

        sibling_values[0] = -1
        return sibling_values

    def get_leaf_all(self, arr):
        tree = self.tree
        return arr[tree.children_left==-1]

    def gen_leaf_values(self):
        tree = self.tree
        return self.get_leaf_all(tree.value).flatten()

    def gen_leaf_sibling_values(self):
        tree = self.tree
        return self.get_leaf_all(self.sibling_values)

class CLTree():

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def fit(self, X):
        X = np.asarray(X)
        self.shape = X.shape
        ud_space = UDSpace(self.ratio)
        tX, tY = ud_space.fit(X).transform()
        dtree = DTree()
        dtree.fit(tX, tY)

        # output rules in json format, rules can be used in pandas
        tree_ = dtree.tree_
        children_left = tree_.children_left
        children_right = tree_.children_right
        feature = tree_.feature
        impurity = tree_.impurity
        threshold = tree_.threshold
        n_node_samples = tree_.n_node_samples

        root_node = {"index":0}
        node_queue = deque()
        node_queue.append(root_node)

        while len(node_queue)>0:
            cur_node = node_queue.popleft()
            cur_node_index = cur_node["index"]

            # fill node
            cur_node["feature"] = feature[cur_node_index]
            cur_node["impurity"] = impurity[cur_node_index]
            cur_node["threshold"] = threshold[cur_node_index]
            cur_node["n_samples"] = n_node_samples[cur_node_index]

            cur_node["leaf_flag"] = False

            # append left and right child, when it has children
            left_child_index = children_left[cur_node_index]
            right_child_index = children_right[cur_node_index]
            
            if left_child_index >0:
                left_child = {"index":left_child_index}
                right_child = {"index":right_child_index}
                cur_node["left"] = left_child
                cur_node["right"] = right_child
                node_queue.append(left_child)
                node_queue.append(right_child)
            else:
                cur_node["leaf_flag"] = True

        self.root_node = root_node
        self.dtree = dtree

        return self

    def get_rules(self):
        return self.root_node

    def export_dtree(self, filepath=None, mode="graphvz"):
        dtree = self.dtree

        if filepath is None:
            return self.root_node

        if mode == "graphvz":
            ExportDtree(dtree, out_file=filepath, leaves_parallel=True, filled=True, node_ids=False, label=True)
        elif mode == "json":
            with open(filepath, "w") as F:
                json.dump(self.root_node, F, indent=2, ensure_ascii=False)

    def predict(self, X):
        """
        return leaf index as cluster
        """

def test_CLTree():
    tX, tY = make_blobs(n_samples=100, n_features=5, centers=3, cluster_std=1.0, center_box=(1.0, 10.0), shuffle=True)
    # print tX, tY
    clt = CLTree(ratio=0.9)
    clt.fit(tX)
    clt.export_dtree("./export_dtree/graphvz.dot", mode="graphvz")
    clt.export_dtree("./export_dtree/dtree.json", mode="json")




if __name__ == '__main__':
    test_CLTree()



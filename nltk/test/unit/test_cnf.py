# -*- coding: utf-8 -*-
"""
Unit tests for tree.chomsky_normal_form()
"""

import unittest
import nltk
from nltk.tree import Tree

class TestTreeCNF(unittest.TestCase):

    @staticmethod
    def is_chomsky_normal_form(tree):
        # a chonsky normal form only contains
        # A -> BC or A -> terminal rules
        if len(tree) == 1 and isinstance(tree[0], str):
            return True
        elif len(tree) == 2 and isinstance(tree[0], Tree) \
                            and isinstance(tree[1], Tree):
            return TestTreeCNF.is_chomsky_normal_form(tree[0]) and \
                   TestTreeCNF.is_chomsky_normal_form(tree[1])
        else:
            return False

    @staticmethod
    def encodes_same_leaves(treeA, treeB):
        return treeA.leaves() == treeB.leaves()

    def test_tree_cnf_0(self):
        gr = nltk.CFG.fromstring("""
        S -> '1' | '2' 
        S -> S OP S 
        OP -> '+' | '-'
        """)
        parser = nltk.ChartParser(gr)
        tree = next(parser.parse('1 + 2'.split()))
        treeCopy = tree.copy(deep=True)
        treeCopy.chomsky_normal_form()
        assert(TestTreeCNF.is_chomsky_normal_form(treeCopy))
        assert(TestTreeCNF.encodes_same_leaves(tree, treeCopy))

    
    def test_tree_cnf_1(self):
        gr = nltk.CFG.fromstring("""
        S -> '1' | '2' 
        S -> S '+' S | S '-' S
        """)
        parser = nltk.ChartParser(gr)
        parse_obj = parser.parse('1 + 2'.split())
        tree = next(parse_obj)
        treeCopy = tree.copy(deep=True)
        treeCopy.chomsky_normal_form()
        assert(TestTreeCNF.is_chomsky_normal_form(treeCopy))
        assert(TestTreeCNF.encodes_same_leaves(tree, treeCopy))
    
    def test_tree_cnf_3(self):
        gr = nltk.CFG.fromstring("""
        S -> L
        S -> S '+' S | S '-' S
        L -> '1' | '2' 
        """)
        parser = nltk.ChartParser(gr)
        parse_obj = parser.parse('1 + 2'.split())
        tree = next(parse_obj)
        treeCopy = tree.copy(deep=True)
        treeCopy.chomsky_normal_form()
        assert(TestTreeCNF.is_chomsky_normal_form(treeCopy))
        assert(TestTreeCNF.encodes_same_leaves(tree, treeCopy))
    
    def test_tree_cnf_4(self):
        gr = nltk.CFG.fromstring("""
        L -> '1' | '2' 
        S -> L
        S -> S '+' S | S '-' S
        """)
        parser = nltk.ChartParser(gr)
        parse_obj = parser.parse('1 + 2'.split())
        tree = next(parse_obj)
        treeCopy = tree.copy(deep=True)
        treeCopy.chomsky_normal_form()
        assert(TestTreeCNF.is_chomsky_normal_form(treeCopy))
        assert(TestTreeCNF.encodes_same_leaves(tree, treeCopy))

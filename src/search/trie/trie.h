#pragma once

/**
 * Source: https://github.com/akshitgrover/trie
 */

#include <vector>
#include <cassert>

#include "trie_node.h"

enum SearchRule { supersets, subsets, samesets };
SearchRule getRule(std::string rule) {
  if (rule=="vu_u" || rule=="supersets")
    return SearchRule::supersets;
  else if (rule=="v_vu" || rule=="subsets")
    return SearchRule::subsets;
  else if (rule=="v_v" || rule=="samesets")
    return SearchRule::samesets;
  assert(false);
  return SearchRule::samesets;
}

namespace trie {
  int MAX_CHILDREN = 128;

  template <typename T>
  class trie {
  public:
    using KeyType = std::vector<int>;

    trie();
    void insert(KeyType, T);
    bool exists(KeyType);
    bool empty();
    void find_all_compatible(KeyType key, SearchRule rule, std::vector<T>& values) const;

    bool has_superset(KeyType key) const {
      return has_superset(key, 0, root);
    }

    bool has_subset(KeyType key) const {
      return has_subset(key, 0, root);
    }

  private:
    void find_samesets (const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) const;
    void find_supersets(const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) const;
    void find_subsets  (const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) const;
    bool has_superset  (const KeyType& key, unsigned pos, tnode<T>* n) const;
    bool has_subset    (const KeyType& key, unsigned pos, tnode<T>* n) const;

    void adjust_key(KeyType& key) const;

    tnode<T> *root;
    bool is_empty;
  };

  template <typename T>
  trie<T>::trie(): is_empty(true) {
    root = new tnode<T>(T());
  }

  template <typename T>
  void trie<T>::insert(KeyType key, T val) {
    tnode<T>* node = root;
    adjust_key(key);
    for (int v : key) {
      auto child = node->getChild(v);
      if (child == nullptr) {
	T flag = T();
	tnode<T>* _node = new tnode<T>(flag);
	node->addChild(_node, v);
      }
      node = node->getChild(v);
    }
    is_empty = false;
    node->update(val);
  }

  template <typename T>
  bool trie<T>::exists(KeyType key) {
    tnode<T>* node = root;
    adjust_key(key);
    for (int v : key) {
      auto child = node->getChild(v);
      if (child == nullptr)
	return false;
      node = child;
    }
    return true;
  }

  template <typename T>
  bool trie<T>::empty() {
    return is_empty;
  }

  // Our use case has -1, so its increments to get the values in the range (0..MAX_CHILDREN-1)
  template<typename T>
  void trie<T>::adjust_key(KeyType& key) const {
    for (int& v : key) {
      v++;
      assert(v >= 0 && v < MAX_CHILDREN);
    }
  }

  template <typename T>
  void trie<T>::find_all_compatible(KeyType key, SearchRule rule, std::vector<T>& values) const {
    adjust_key(key);
    switch(rule) {
    case SearchRule::samesets:
      find_samesets(key, 0, root, values);
      break;
    case SearchRule::supersets:
      find_supersets(key, 0, root, values);
      break;
    case SearchRule::subsets:
      find_subsets(key, 0, root, values);
      break;
    }
  }

  template <typename T>
  void trie<T>::find_samesets(const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) const {
    if (n==nullptr)
      return;

    if (pos==key.size()) {
      values.push_back(n->get());
      return;
    }
    find_samesets(key, pos + 1, n->getChild(key[pos]), values);
  }

  template <typename T>
  void trie<T>::find_supersets(const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) const {
    if (n==nullptr)
      return;

    if (pos==key.size()) {
      values.push_back(n->get());
      return;
    }

    find_supersets(key, pos + 1, n->getChild(key[pos]), values);
    if (key[pos] != 0)
      find_supersets(key, pos + 1, n->getChild(0), values);
  }

  template <typename T>
  bool trie<T>::has_superset(const KeyType& key, unsigned pos, tnode<T>* n) const {
    if (n==nullptr)
      return false;

    if (pos==key.size())
      return true;

    if (has_superset(key, pos+1, n->getChild(key[pos])))
      return true;

    if (key[pos] != 0)
      return has_superset(key, pos+1, n->getChild(0));

    return false;
  }

  template <typename T>
  void trie<T>::find_subsets(const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) const {
    if (n==nullptr)
      return;

    if (pos==key.size()) {
      values.push_back(n->get());
      return;
    }

    find_subsets(key, pos + 1, n->getChild(key[pos]), values);

    if (key[pos] == 0)
      for(auto& [i,cnode] : n->children)
	find_subsets(key, pos + 1, cnode, values);
  }

  template <typename T>
  bool trie<T>::has_subset(const KeyType& key, unsigned pos, tnode<T>* n) const {
    if (n==nullptr)
      return false;

    if (pos==key.size())
      return true;

    if (has_subset(key, pos+1, n->getChild(key[pos])))
      return true;

    if (key[pos] == 0)
      for(auto& [i,cnode] : n->children)
	 return has_subset(key, pos + 1, cnode);

    return false;
  }
} // namespace trie

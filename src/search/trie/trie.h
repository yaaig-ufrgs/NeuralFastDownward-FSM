#ifndef TRIE_TRIE_H
#define TRIE_TRIE_H

/**
 * Source: https://github.com/akshitgrover/trie
 */

#include <vector>
#include <assert.h>

#include "trie_node.h"
#include "trie_iterator.h"

enum SearchRule { supersets, subsets, samesets };
SearchRule getRule(std::string rule) {
  if (rule=="vu_u")
    return SearchRule::supersets;
  else if (rule=="v_vu")
    return SearchRule::subsets;
  else if (rule=="v_v")
    return SearchRule::samesets;
  assert(false);
  return SearchRule::samesets;
}

namespace trie {
  template <typename T>
  class trie {
  public:
    using iterator = trie_iterator<T>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using KeyType = std::vector<int>;

    trie();
    void insert(KeyType, T);
    bool exist(KeyType);
    bool empty();
    iterator begin();
    iterator end();
    reverse_iterator rbegin();
    reverse_iterator rend();
    iterator find(KeyType);
    void find_all_compatible(KeyType key, SearchRule rule, std::vector<T>& values);

  private:
    void find_samesets (const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values);
    void find_supersets(const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values);
    void find_subsets  (const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values);

    void adjust_key(KeyType& key);

    tnode<T> *root;
    int size;
    bool is_empty;
  };

  template <typename T>
  trie<T>::trie(): size(0) {
    T flag = T();
    root = new tnode<T>(flag, nullptr, -1);
    size = 0;
    is_empty = true;
  }

  template <typename T>
  void trie<T>::insert(KeyType key, T val) {
    tnode<T>* node = this->root;
    adjust_key(key);
    for (int v : key) {
      if (node->getChild(v) != nullptr) {
	node = node->getChild(v);
      } else {
	T flag = T();
	tnode<T>* _node = new tnode<T>(flag, node, v);
	node->addChild(_node, v);
	node = node->getChild(v);
      }
    }
    if (!node->isEnd()) {
      this->size += 1;
    }
    this->is_empty = false;
    node->update(val);
  }

  template <typename T>
  bool trie<T>::exist(KeyType key) {
    bool res = true;
    tnode<T>* node = this->root;
    adjust_key(key);
    for (int v : key) {
      if (node->getChild(v) == nullptr) {
	res = false;
	break;
      } else {
	node = node->getChild(v);
      }
    }
    if (!node->isEnd()) {
      res = false;
    }
    return res;
  }

  template <typename T>
  bool trie<T>::empty() {
    // return this->size == 0;
    return this->is_empty;
  }

  template <typename T>
  typename trie<T>::iterator trie<T>::begin() {
    trie_iterator<T> it = *(new trie_iterator<T>(this->root));
    return ++it;
  }

  template <typename T>
  tnode<T>* rbrecur(tnode<T>* n, int offset = MAX_CHILDREN-1, tnode<T>* r = nullptr);

  template <typename T>
  typename trie<T>::iterator trie<T>::end() {
    T flag;
    tnode<T>* r = nullptr;
    if (!this->empty()) {
      r = rbrecur(this->root);
    }
    tnode<T>* t = new tnode<T>(flag, r, 1516);
    return *(new trie_iterator<T>(t));
  }

  template <typename T>
  tnode<T>* rbrecur(tnode<T>* n, int offset, tnode<T>* r) {
    tnode<T>* it = nullptr;
    for (int i = offset; i > -1; i--) {
      it = n->getChild(i);
      if (it == nullptr) {
	if (i == 0) {
	  return r;
	}
	continue;
      }
      if (it->isEnd()) {
	r = it;
      }
      return rbrecur(it, MAX_CHILDREN-1, r);
    }
    return nullptr;
  }

  template <typename T>
  typename trie<T>::reverse_iterator trie<T>::rbegin() {
    return *(new trie<T>::reverse_iterator(trie<T>::end()));
  }

  template <typename T>
  typename trie<T>::reverse_iterator trie<T>::rend() {
    return *(new trie<T>::reverse_iterator(trie<T>::begin()));
  }

  template <typename T>
  typename trie<T>::iterator trie<T>::find(KeyType key) {
    tnode<T>* n = this->root;
    adjust_key(key);
    for (int v : key) {
      n = n->getChild(v);
      if (n == nullptr)
	return this->end();
    }
    if (!n->isEnd()) {
      return this->end();
    }
    trie_iterator<T> it = *(new trie_iterator<T>(n));
    return it;
  }


  // Our use case has -1, so its increments to get the values in the range (0..MAX_CHILDREN-1)
  template<typename T>
  void trie<T>::adjust_key(KeyType& key) {
    for (int& v : key) {
      v++;
      assert(v >= 0 && v < MAX_CHILDREN);
    }
  }
  
  template <typename T>
  void trie<T>::find_all_compatible(KeyType key, SearchRule rule, std::vector<T>& values) {
    adjust_key(key);
    switch(rule) {
    case SearchRule::samesets:
      find_samesets(key, 0, this->root, values);
      break;
    case SearchRule::supersets:
      find_supersets(key, 0, this->root, values);
      break;
    case SearchRule::subsets:
      find_subsets(key, 0, this->root, values);
      break;
    }
  }

  template <typename T>
  void trie<T>::find_samesets(const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) {
    if (n==nullptr)
      return;
  
    if (pos==key.size()) {
      values.push_back(n->get());
      return;
    }
    find_samesets(key, pos + 1, n->getChild(key[pos]), values);
  }

  template <typename T>
  void trie<T>::find_supersets(const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) {
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
  void trie<T>::find_subsets(const KeyType& key, unsigned pos, tnode<T>* n, std::vector<T>& values) {
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
    // for (int i = 1; i < MAX_CHILDREN-1; i++)
    //   find_subsets(key, pos + 1, n->getChild(i), values);
  }
} // namespace trie

#endif
